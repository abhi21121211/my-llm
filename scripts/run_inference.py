import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from pathlib import Path
import gc
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)  # Adjust this based on your CPU cores

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

def load_model(model_path):
    try:
        # Convert to Path object and resolve to absolute path
        model_path = Path(model_path).resolve()
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        # List directory contents for debugging
        list_directory_contents(model_path)
            
        # Check for required files
        required_files = ["config.json", "pytorch_model.bin"]
        missing_files = [f for f in required_files if not (model_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
            
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True
        )
        
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            device_map=None,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )
        
        # Ensure model is on CPU
        model = model.cpu()
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.7):
    try:
        logger.info(f"Generating text with prompt: '{prompt}'")
        
        # Tokenize the input text
        encoded_input = tokenizer.encode(prompt, return_tensors="pt")
        
        # Move to CPU to be safe
        encoded_input = encoded_input.to("cpu")
        
        # Generate output
        logger.info(f"Running generation with max_length={max_length}, temperature={temperature}")
        with torch.no_grad():
            output = model.generate(
                encoded_input,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode the output
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Generated {len(output[0])} tokens")
        
        return decoded_output
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise

def cleanup():
    """Clean up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    # Use tiny-gpt2 model path
    script_dir = Path(__file__).parent.parent
    model_path = script_dir / "models" / "tiny-gpt2"
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model(model_path)
        
        # Example prompt for GPT-2
        prompt = "Once upon a time, in a land far away,"
        
        # Generate text
        generated_text = generate_text(model, tokenizer, prompt, max_length=100)
        print("\nGenerated Response:")
        print("-" * 50)
        print(generated_text)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        cleanup()

if __name__ == "__main__":
    main()