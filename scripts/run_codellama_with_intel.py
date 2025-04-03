import logging
import os
from pathlib import Path
import gc
import sys
import ctypes
from typing import List, Optional
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if Intel GPU is available through PyTorch
def check_intel_gpu():
    try:
        if torch.xpu.is_available():
            logger.info("Intel XPU (GPU) is available!")
            return True
        else:
            logger.info("Intel XPU (GPU) is not available through PyTorch")
            return False
    except:
        logger.info("Intel XPU support not found in PyTorch")
        return False

# Try to use Intel GPU via OneAPI if available
use_intel_gpu = check_intel_gpu()
if use_intel_gpu:
    logger.info("Will attempt to use Intel GPU acceleration")
    torch.set_num_threads(8)  # Use more CPU threads when offloading to GPU
else:
    logger.info("Using CPU only")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(4)  # Adjust based on your CPU cores

def list_directory_contents(path):
    """List contents of a directory for debugging"""
    path = Path(path)
    logger.info(f"Contents of {path}:")
    
    if not path.exists():
        logger.error(f"Directory {path} does not exist!")
        return
    
    for item in path.iterdir():
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            logger.info(f"  File: {item.name} ({size_mb:.2f} MB)")
        else:
            logger.info(f"  Dir: {item.name}")

def generate_code(model_path, prompt, max_tokens=500, temperature=0.7):
    """Generate code using llama-cpp-python"""
    try:
        # Import here to avoid loading if just checking requirements
        from llama_cpp import Llama
        
        logger.info(f"Loading model from {model_path}")
        
        # Try to use Intel GPU offloading if available
        # Note: n_gpu_layers determines how many layers to offload to GPU
        n_gpu_layers = 1 if use_intel_gpu else 0
        
        # Initialize the model with appropriate parameters
        model = Llama(
            model_path=str(model_path),
            n_ctx=2048,           # Context window size
            n_batch=512,          # Batch size for prompt processing
            n_threads=8 if use_intel_gpu else 4,  # More threads when using GPU offloading
            n_gpu_layers=n_gpu_layers  # Try to use Intel GPU for some layers
        )
        
        logger.info(f"Generating with prompt: '{prompt}'")
        
        # Generate text
        output = model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repeat_penalty=1.1,
            top_k=40,
            stop=["```", "\n\n\n"],  # Stop generation at these tokens
            echo=False              # Don't include prompt in the output
        )
        
        # Get generated text
        if output and "text" in output:
            generated_text = output["text"]
            logger.info(f"Generated {len(generated_text)} characters")
            return generated_text
        else:
            logger.error("No text was generated")
            return ""
            
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        raise

def cleanup():
    """Clean up memory"""
    gc.collect()
    
def main():
    # Get the path to the model
    script_dir = Path(__file__).parent.parent
    model_dir = script_dir / "models" / "codellama-7b-python-4bit"
    model_path = model_dir / "codellama-7b-python.Q4_K_M.gguf"
    
    # Check if model exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please run 'python scripts/download_codellama.py' first")
        sys.exit(1)
    
    # Verify model file
    file_size_gb = model_path.stat().st_size / (1024 * 1024 * 1024)
    logger.info(f"Model file size: {file_size_gb:.2f} GB")
    
    try:
        # Get user's prompt for code generation
        user_prompt = input("Enter a coding task (or press Enter for the default palindrome example): ")
        
        # Use default prompt if user didn't provide one
        if not user_prompt.strip():
            prompt = """Write a Python function to check if a string is a valid palindrome, considering only alphanumeric characters and ignoring case.

```python
def is_palindrome(s: str) -> bool:
"""
        else:
            # Format user's prompt for code generation
            prompt = f"""{user_prompt}

```python
"""
        
        # Generate code
        generated_code = generate_code(
            model_path=model_path,
            prompt=prompt,
            max_tokens=500 if user_prompt else 300,  # More tokens for custom prompts
            temperature=0.2  # Lower temperature for more focused code generation
        )
        
        # Display the result
        print("\nGenerated Code:")
        print("-" * 50)
        print("```python")
        if user_prompt:
            print(generated_code)
        else:
            print("def is_palindrome(s: str) -> bool:")
            print(generated_code)
        print("```")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        cleanup()

if __name__ == "__main__":
    main() 