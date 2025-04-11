import os
import sys
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Load token from .env if available
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

MODEL_URL = "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
SAVE_DIR = "models/llama3-8b-gguf"
SAVE_PATH = os.path.join(SAVE_DIR, "llama-3-8b-instruct.Q4_K_M.gguf")
ENV_FILE = ".env"

def ask_and_save_token():
    token = input("🔐 Enter your Hugging Face Access Token (starting with 'hf_'): ").strip()
    if token.startswith("hf_"):
        with open(ENV_FILE, "a") as f:
            f.write(f"\nHUGGINGFACE_TOKEN={token}")
        print("✅ Token saved to .env")
        return token
    else:
        print("❌ Invalid token format.")
        sys.exit(1)

def download_file(url, filepath, token):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Authorization': f'Bearer {token}'
    }

    response = requests.get(url, stream=True, headers=headers)
    if response.status_code == 401:
        raise Exception("Unauthorized: Invalid or expired token.")

    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024

    with open(filepath, 'wb') as f, tqdm(
        desc="📦 Downloading model",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)

    file_size = os.path.getsize(filepath)
    if total_size > 0 and file_size != total_size:
        os.remove(filepath)
        raise Exception("Download incomplete or corrupted.")

def main():
    print("== LLaMA 3 GGUF Model Downloader ==")

    global HF_TOKEN
    if not HF_TOKEN:
        HF_TOKEN = ask_and_save_token()

    if os.path.exists(SAVE_PATH):
        choice = input(f"Model already exists at {SAVE_PATH}. Re-download? (y/n): ")
        if choice.lower() != 'y':
            print("Using existing model file.")
            return
        else:
            os.remove(SAVE_PATH)

    try:
        download_file(MODEL_URL, SAVE_PATH, HF_TOKEN)
        print(f"\n✅ Model downloaded to {SAVE_PATH}")

        file_size_gb = os.path.getsize(SAVE_PATH) / (1024**3)
        if file_size_gb < 3:
            print(f"⚠️ File size ({file_size_gb:.2f} GB) seems small. Might be corrupted.")
            retry = input("Try downloading again? (y/n): ")
            if retry.lower() == 'y':
                os.remove(SAVE_PATH)
                download_file(MODEL_URL, SAVE_PATH, HF_TOKEN)
                print(f"✅ Re-downloaded model to {SAVE_PATH}")
        else:
            print(f"✅ File size looks good: {file_size_gb:.2f} GB")

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)
        sys.exit(1)

if __name__ == "__main__":
    main()
