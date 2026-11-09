import os
import json
import hashlib
import hmac
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional


class SecureModelStore:
    """
    Encrypts model weights at rest using AES-256-GCM.
    Verifies model integrity on every load via HMAC-SHA256.
    """

    def __init__(self, encryption_key: str, store_dir: str = ".nervsharp/models"):
        if not isinstance(encryption_key, str) or not encryption_key:
            raise ValueError("encryption_key must be a nonempty string")
        self._key = self._derive_key(encryption_key)
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    @staticmethod
    def _validate_id(model_id):
        import re
        if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", model_id):
            raise ValueError("Invalid model identifier")

    @staticmethod
    def _atomic_write(destination: Path, data: bytes) -> None:
        temporary = None
        try:
            with NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _derive_key(self, key_str: str) -> bytes:
        return hashlib.sha256(key_str.encode()).digest()

    def encrypt_model(self, model_path: str, model_id: str) -> str:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise ImportError("cryptography is required: pip install nervsharp[privacy]")

        self._validate_id(model_id)
        data = Path(model_path).read_bytes()
        nonce = os.urandom(12)
        aesgcm = AESGCM(self._key)
        encrypted = aesgcm.encrypt(nonce, data, None)

        mac = hmac.new(self._key, encrypted, hashlib.sha256).digest()

        out_path = self.store_dir / f"{model_id}.enc"
        self._atomic_write(out_path, nonce + mac + encrypted)

        manifest = {
            "model_id": model_id,
            "original_size": len(data),
        }
        self._atomic_write(
            self.store_dir / f"{model_id}.json",
            (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
        )
        return str(out_path)

    def decrypt_model(self, model_id: str, output_path: str) -> str:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise ImportError("cryptography is required: pip install nervsharp[privacy]")

        self._validate_id(model_id)
        enc_path = self.store_dir / f"{model_id}.enc"
        if not enc_path.exists():
            raise FileNotFoundError(f"encrypted model not found: {model_id}")

        raw = enc_path.read_bytes()
        nonce = raw[:12]
        stored_mac = raw[12:44]
        encrypted = raw[44:]

        expected_mac = hmac.new(self._key, encrypted, hashlib.sha256).digest()
        if not hmac.compare_digest(stored_mac, expected_mac):
            raise ValueError("model integrity check failed: HMAC mismatch")

        aesgcm = AESGCM(self._key)
        data = aesgcm.decrypt(nonce, encrypted, None)
        # Replace the destination only after authentication and a complete write.
        destination = Path(output_path)
        temporary = None
        try:
            with NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return output_path

    def list_models(self) -> list[dict]:
        return [
            json.loads((self.store_dir / f).read_text())
            for f in self.store_dir.iterdir()
            if f.suffix == ".json"
        ]


