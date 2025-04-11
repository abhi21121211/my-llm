# scripts/run_tiny_gpt2.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_PATH = Path("models/tiny-gpt2")

def main():
    print("🔁 Loading Tiny-GPT2 model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    while True:
        prompt = input("\n📝 Enter prompt (or 'exit'): ")
        if prompt.lower() == "exit":
            break

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs, max_length=100, do_sample=True, temperature=0.8
        )

        print("📣 Output:")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
