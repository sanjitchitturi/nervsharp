import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class QuantizationResult:
    input_path: str
    output_path: str
    quantization_type: str
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    elapsed_s: float


class ModelQuantizer:
    """
    Converts full-precision models to INT4 or INT8 GGUF format for edge deployment.
    Wraps llama.cpp's quantization toolchain with a clean Python interface.
    """

    SUPPORTED_TYPES = ["int4", "int8", "int4_k_m", "int4_k_s", "int8_0"]

    LLAMA_CPP_TYPE_MAP = {
        "int4": "Q4_0",
        "int4_k_m": "Q4_K_M",
        "int4_k_s": "Q4_K_S",
        "int8": "Q8_0",
        "int8_0": "Q8_0",
    }

    def quantize(
        self,
        model_path: str,
        output_path: str,
        quantization_type: Literal["int4", "int8", "int4_k_m", "int4_k_s"] = "int4",
        calibration_data: Optional[str] = None,
    ) -> QuantizationResult:
        if quantization_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"unsupported quantization type: {quantization_type}. choose from {self.SUPPORTED_TYPES}")

        input_path = Path(model_path)
        if not input_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        q_type = self.LLAMA_CPP_TYPE_MAP[quantization_type]
        original_size_mb = input_path.stat().st_size / 1e6

        start = time.perf_counter()
        self._run_quantization(str(input_path), output_path, q_type, calibration_data)
        elapsed = time.perf_counter() - start

        out_path = Path(output_path)
        quantized_size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0.0

        return QuantizationResult(
            input_path=str(input_path),
            output_path=output_path,
            quantization_type=quantization_type,
            original_size_mb=round(original_size_mb, 2),
            quantized_size_mb=round(quantized_size_mb, 2),
            compression_ratio=round(original_size_mb / quantized_size_mb, 2) if quantized_size_mb else 0.0,
            elapsed_s=round(elapsed, 2),
        )

    def _run_quantization(self, input_path: str, output_path: str, q_type: str, calibration_data: Optional[str]):
        import subprocess
        import shutil

        quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
        if quantize_bin:
            cmd = [quantize_bin, input_path, output_path, q_type]
            if calibration_data:
                cmd += ["--calibration-data", calibration_data]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"quantization failed: {result.stderr}")
        else:
            try:
                from llama_cpp import llama_model_quantize_params
                raise NotImplementedError(
                    "llama-quantize binary not found. install llama.cpp and ensure it is on your PATH. "
                    "see: https://github.com/ggerganov/llama.cpp"
                )
            except ImportError:
                raise RuntimeError(
                    "quantization requires llama-quantize binary on PATH. "
                    "build from: https://github.com/ggerganov/llama.cpp"
                )

    def estimate_quantized_size(self, model_path: str, quantization_type: str = "int4") -> dict:
        size_mb = Path(model_path).stat().st_size / 1e6
        ratios = {"int4": 0.25, "int4_k_m": 0.27, "int4_k_s": 0.24, "int8": 0.5, "int8_0": 0.5}
        ratio = ratios.get(quantization_type, 0.25)
        return {
            "original_mb": round(size_mb, 2),
            "estimated_quantized_mb": round(size_mb * ratio, 2),
            "estimated_compression_ratio": round(1 / ratio, 2),
        }

