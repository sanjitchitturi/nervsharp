import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_hardware_detection():
    from nervsharp.core.hardware import detect_hardware
    info = detect_hardware()
    assert "device_name" in info
    assert "has_gpu" in info
    assert "cpu_count" in info
    assert isinstance(info["cpu_count"], int)
    assert info["cpu_count"] > 0


def test_benchmark_uses_runtime_tokenizer_counts():
    from nervsharp.engine import NervSharp

    engine = NervSharp.__new__(NervSharp)
    engine._llm = MagicMock()
    engine._llm.return_value = {"choices": [{"text": "generated"}], "usage": {}}
    engine._llm.tokenize.side_effect = lambda text, add_bos=False: list(text)
    engine._device_info = {"device_name": "test-cpu"}

    result = engine.benchmark(n_tokens=4)

    assert result["generated_tokens"] == len("generated")
    assert result["prompt_tokens"] > result["generated_tokens"]
    assert result["count_source"] == "tokenizer"
    assert "generated_tokens_per_second" in result


def test_benchmark_prefers_runtime_usage_and_validates_limit():
    from nervsharp.engine import NervSharp

    engine = NervSharp.__new__(NervSharp)
    engine._llm = MagicMock(return_value={
        "choices": [{"text": "generated"}],
        "usage": {"prompt_tokens": 21, "completion_tokens": 4},
    })
    engine._device_info = {"device_name": "test-cpu"}
    result = engine.benchmark(n_tokens=4)
    assert result["prompt_tokens"] == 21
    assert result["generated_tokens"] == 4
    assert result["count_source"] == "runtime_usage"
    engine._llm.tokenize.assert_not_called()
    with pytest.raises(ValueError, match="positive integer"):
        engine.benchmark(n_tokens=0)


def test_quantizer_unsupported_type():
    from nervsharp.core.quantizer import ModelQuantizer
    q = ModelQuantizer()
    with pytest.raises(ValueError, match="unsupported quantization type"):
        q.quantize("model.bin", "out.gguf", quantization_type="float16")


def test_quantizer_missing_model():
    from nervsharp.core.quantizer import ModelQuantizer
    q = ModelQuantizer()
    with pytest.raises(FileNotFoundError):
        q.quantize("/nonexistent/model.bin", "out.gguf")


def test_quantizer_estimate(tmp_path):
    from nervsharp.core.quantizer import ModelQuantizer
    model = tmp_path / "model.bin"
    model.write_bytes(b"x" * 1_000_000)
    q = ModelQuantizer()
    result = q.estimate_quantized_size(str(model), "int4")
    assert result["original_mb"] == pytest.approx(1.0, abs=0.1)
    assert result["estimated_quantized_mb"] < result["original_mb"]
    assert result["estimated_compression_ratio"] > 1


def test_secure_model_store_encrypt_decrypt(tmp_path):
    pytest.importorskip("cryptography")
    from nervsharp.privacy.secure import SecureModelStore

    store = SecureModelStore("test-key-1234", store_dir=str(tmp_path / "store"))
    model_data = b"fake model weights " * 100
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(model_data)

    store.encrypt_model(str(model_path), "test-model")
    out_path = str(tmp_path / "decrypted.gguf")
    store.decrypt_model("test-model", out_path)
    assert Path(out_path).read_bytes() == model_data


def test_secure_model_store_tamper_detection(tmp_path):
    pytest.importorskip("cryptography")
    from nervsharp.privacy.secure import SecureModelStore

    store = SecureModelStore("test-key-1234", store_dir=str(tmp_path / "store"))
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"weights" * 50)
    store.encrypt_model(str(model_path), "test-model")

    enc_path = tmp_path / "store" / "test-model.enc"
    data = bytearray(enc_path.read_bytes())
    data[50] ^= 0xFF
    enc_path.write_bytes(bytes(data))

    with pytest.raises(Exception):
        store.decrypt_model("test-model", str(tmp_path / "out.gguf"))


def test_store_rejects_path_traversal(tmp_path):
    from nervsharp.privacy.secure import SecureModelStore
    store = SecureModelStore("test-key", str(tmp_path))
    with pytest.raises(ValueError, match="identifier"):
        store.decrypt_model("../outside", str(tmp_path / "out"))


def test_decrypted_output_is_private_and_replaces_symlink(tmp_path):
    from nervsharp.privacy.secure import SecureModelStore

    store = SecureModelStore("test-key", str(tmp_path / "vault"))
    source = tmp_path / "weights"
    source.write_bytes(b"model")
    store.encrypt_model(str(source), "example")
    other = tmp_path / "unrelated"
    other.write_bytes(b"keep")
    output = tmp_path / "output"
    output.symlink_to(other)
    store.decrypt_model("example", str(output))
    assert output.read_bytes() == b"model"
    assert other.read_bytes() == b"keep"
    assert output.stat().st_mode & 0o077 == 0


def test_encrypted_output_is_private_and_replaces_symlink(tmp_path):
    from nervsharp.privacy.secure import SecureModelStore

    store = SecureModelStore("test-key", str(tmp_path / "vault"))
    source = tmp_path / "weights"
    source.write_bytes(b"model")
    other = tmp_path / "unrelated"
    other.write_bytes(b"keep")
    encrypted = tmp_path / "vault" / "example.enc"
    encrypted.symlink_to(other)

    store.encrypt_model(str(source), "example")

    assert other.read_bytes() == b"keep"
    assert encrypted.is_file() and not encrypted.is_symlink()
    assert encrypted.stat().st_mode & 0o077 == 0


def test_store_rejects_empty_key_and_nonstring_identifier(tmp_path):
    from nervsharp.privacy.secure import SecureModelStore

    with pytest.raises(ValueError, match="encryption_key"):
        SecureModelStore("", str(tmp_path))
    store = SecureModelStore("test-key", str(tmp_path))
    with pytest.raises(ValueError, match="identifier"):
        store.decrypt_model(None, str(tmp_path / "out"))
