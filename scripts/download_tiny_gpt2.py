# scripts/download_tiny_gpt2.py

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "sshleifer/tiny-gpt2"
SAVE_DIR = Path("models/tiny-gpt2")

def download_model():
    print(f"Downloading {MODEL_NAME} to {SAVE_DIR}...")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)
    
    print("✅ Download complete.")

if __name__ == "__main__":
    download_model()
