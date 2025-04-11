# scripts/download_phi3_cpu.py
import os
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
SAVE_DIR = "models/phi3-mini-cpu"

def main():
    print(f"Downloading {MODEL_NAME} for CPU usage...")
    
    # Create directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Download tokenizer first
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_DIR)
    
    # Download model with float16 precision for CPU
    print("Downloading model (this may take a while)...")
    try:
        # First try with fp16 for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",  # Will use float16 if available
            low_cpu_mem_usage=True,
            device_map="cpu"  # Force CPU
        )
    except Exception as e:
        print(f"Error with float16 precision: {e}")
        print("Trying with default precision...")
        # Fall back to default precision if needed
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True,
            device_map="cpu"  # Force CPU
        )
    
    model.save_pretrained(SAVE_DIR)
    
    print(f"Model and tokenizer saved to {SAVE_DIR}")
    print("Note: This is a standard CPU model without 4-bit quantization.")
    print("      It will use more memory but doesn't require CUDA.")

if __name__ == "__main__":
    main()

