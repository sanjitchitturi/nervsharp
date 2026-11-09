from nervsharp import NervSharp

engine = NervSharp(
    model_path="models/phi-2-int4.gguf",
    device="auto",
)

prompts = [
    "Summarize the following support ticket in one sentence:\nTicket: My laptop won't turn on after the update.\nSummary:",
    "Answer the following question concisely:\nQ: What is the difference between RAM and storage?\nA:",
    "Generate a Python function that reverses a string:\n",
]

for prompt in prompts:
    print(f"prompt: {prompt[:60]}...")
    response = engine.generate(prompt, max_tokens=128, temperature=0.3)
    print(f"response: {response.strip()}\n")

print("benchmark:")
stats = engine.benchmark(n_tokens=64)
for k, v in stats.items():
    print(f"  {k}: {v}")

