import time
import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from .core.quantizer import ModelQuantizer
from .core.hardware import detect_hardware


@dataclass
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    seed: int = -1


@dataclass
class NervSharpConfig:
    model_path: str
    device: Literal["auto", "cpu", "gpu", "npu"] = "auto"
    n_threads: int = field(default_factory=lambda: os.cpu_count() or 4)
    n_gpu_layers: int = 0
    context_length: int = 2048
    verbose: bool = False


class NervSharp:
    """
    On-device inference engine for local GGUF language models.

    Runs GGUF-format models locally using llama-cpp-python, with optional
    Quantization and encrypted storage are separate utilities.
    """

    def __init__(self, model_path: str, device: str = "auto", **kwargs):
        self.config = NervSharpConfig(model_path=model_path, device=device, **kwargs)
        self._llm = None
        self._device_info = detect_hardware()
        self._load_model()

    def _load_model(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python is required: pip install nervsharp[inference]")

        n_gpu_layers = self.config.n_gpu_layers
        if self.config.device == "auto" and self._device_info["has_gpu"]:
            n_gpu_layers = -1
        elif self.config.device == "gpu":
            n_gpu_layers = -1

        if self.config.verbose:
            print(f"loading model: {self.config.model_path}")
            print(f"device: {self._device_info['device_name']}")

        self._llm = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.context_length,
            n_threads=self.config.n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=self.config.verbose,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[list[str]] = None,
    ) -> str:
        if self._llm is None:
            raise RuntimeError("model not loaded")

        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
        )
        return output["choices"][0]["text"]

    def generate_stream(self, prompt: str, **kwargs):
        if self._llm is None:
            raise RuntimeError("model not loaded")
        for chunk in self._llm(prompt, stream=True, **kwargs):
            yield chunk["choices"][0]["text"]

    def benchmark(self, n_tokens: int = 128) -> dict:
        if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 1:
            raise ValueError("n_tokens must be a positive integer")
        if self._llm is None:
            raise RuntimeError("model not loaded")
        prompt = "The quick brown fox jumps over the lazy dog. " * 4
        start = time.perf_counter()
        completion = self._llm(prompt, max_tokens=n_tokens, temperature=0.0)
        elapsed = time.perf_counter() - start
        usage = completion.get("usage", {})
        if isinstance(usage.get("prompt_tokens"), int) and isinstance(usage.get("completion_tokens"), int):
            prompt_tokens = usage["prompt_tokens"]
            generated_tokens = usage["completion_tokens"]
            count_source = "runtime_usage"
        else:
            prompt_tokens = self._count_tokens(prompt)
            generated_tokens = self._count_tokens(completion["choices"][0]["text"])
            count_source = "tokenizer" if generated_tokens is not None else "unavailable"
        result = {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "elapsed_s": round(elapsed, 3),
            "device": self._device_info["device_name"],
            "count_source": count_source,
        }
        if generated_tokens is not None and elapsed > 0:
            result["generated_tokens_per_second"] = round(generated_tokens / elapsed, 2)
        return result

    def _count_tokens(self, text: str) -> int | None:
        """Return the runtime tokenizer count when llama.cpp exposes it."""
        tokenize = getattr(self._llm, "tokenize", None)
        if not callable(tokenize):
            return None
        return len(tokenize(text.encode("utf-8"), add_bos=False))

    @property
    def device_info(self) -> dict:
        return self._device_info

