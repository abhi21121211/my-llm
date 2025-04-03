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

# Force CPU usage
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
        
        # Initialize the model with appropriate parameters for CPU
        model = Llama(
            model_path=str(model_path),
            n_ctx=2048,        # Context window size
            n_batch=512,       # Batch size for prompt processing
            n_threads=4,       # Number of CPU threads to use
            n_gpu_layers=0     # No GPU layers
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
        # Python code generation prompt
        prompt = """Write a Python function to check if a string is a valid palindrome, considering only alphanumeric characters and ignoring case.

```python
def is_palindrome(s: str) -> bool:
"""
        
        # Generate code
        generated_code = generate_code(
            model_path=model_path,
            prompt=prompt,
            max_tokens=300,
            temperature=0.2  # Lower temperature for more focused code generation
        )
        
        # Display the result
        print("\nGenerated Code:")
        print("-" * 50)
        print("```python")
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