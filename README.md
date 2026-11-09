# NervSharp

Run a local GGUF text model, inspect available hardware, or estimate the size of a quantized model. Hardware inspection works without downloading weights.

## Quick start

```sh
python -m pip install -e '.[privacy,dev]'
python -c 'from nervsharp import detect_hardware; print(detect_hardware())'
python -m pytest
```

For text generation, install the inference extra and supply a trusted local GGUF file:

```python
from nervsharp import NervSharp

engine = NervSharp(model_path="models/model.gguf", device="cpu")
print(engine.generate("Explain cosine similarity.", max_tokens=64))
```

Quantization requires the llama.cpp quantizer on PATH. Size estimates are heuristics. The encrypted store operates in memory and requires a high-entropy secret kept outside the repository. Hardware and store tests run without a model. Text generation requires a separate model and compatible llama.cpp build. Install `.[inference]` to add that runtime; PyTorch is optional and used only for additional hardware detection.
Inference loads a local GGUF path through llama.cpp. The engine has no privacy
toggle or differential privacy implementation. `SecureModelStore` encrypts files
at rest as a separate operation; loading a model for inference requires a local
decrypted GGUF file under your control.

`benchmark()` prefers token counts returned in the llama.cpp completion response.
When usage counts are absent, it uses the runtime tokenizer without a beginning-of-sequence
token. The `count_source` field reports which path was used, and counts are `null` if neither
is available. Its elapsed time covers the complete request;
it is not a separate prefill/decode benchmark and should not be compared across runtimes
without recording the model, quantization, prompt, and hardware.

## Verification

Run `python -m pytest` to exercise the model store, tamper detection, identifier validation, and hardware inspection.

## License

Read the [changelog](CHANGELOG.md) for release history.

[MIT License](LICENSE) for the project code. Third-party software, models, and datasets retain their own terms.
