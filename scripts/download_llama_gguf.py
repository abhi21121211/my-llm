# scripts/download_llama_gguf.py
import os
import requests
import sys
from tqdm import tqdm
from pathlib import Path

# Choose a smaller GGUF model that's more manageable on CPU
MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
SAVE_DIR = "models/llama-gguf"
SAVE_PATH = os.path.join(SAVE_DIR, "llama-2-7b-chat.Q4_K_M.gguf")

def download_file(url, filepath):
    """Download a file with progress bar"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(filepath, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)

def main():
    print(f"Downloading Llama 2 7B GGUF model (CPU-optimized)...")
    
    if os.path.exists(SAVE_PATH):
        print(f"Model already exists at {SAVE_PATH}")
        sys.exit(0)
    
    print("This will download approximately 4GB of data.")
    choice = input("Continue? (y/n): ").lower()
    if choice != 'y':
        print("Download canceled.")
        sys.exit(0)
    
    download_file(MODEL_URL, SAVE_PATH)
    print(f"Model downloaded to {SAVE_PATH}")

if __name__ == "__main__":
    main()

