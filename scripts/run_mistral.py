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
torch.set_num_threads(6)  # Use more threads for this model

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

def generate_chat_response(model_path, prompt, max_tokens=1024, temperature=0.7):
    """Generate a chat response using llama-cpp-python"""
    try:
        # Import here to avoid loading if just checking requirements
        from llama_cpp import Llama
        
        logger.info(f"Loading model from {model_path}")
        
        # Initialize the model with appropriate parameters for CPU
        model = Llama(
            model_path=str(model_path),
            n_ctx=4096,        # Larger context window size for chat
            n_batch=512,       # Batch size for prompt processing
            n_threads=6        # Number of CPU threads to use
        )
        
        # Format prompt for Mistral Instruct - simplified version
        mistral_prompt = f"<s>[INST] {prompt} [/INST]"
        logger.info(f"Generating response for prompt length: {len(prompt)} characters")
        
        # Generate text
        output = model.create_completion(
            mistral_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repeat_penalty=1.1,
            top_k=40,
            stop=["</s>", "[INST]"],  # Stop at the end of the response
            echo=False                # Don't include prompt in the output
        )
        
        # Get generated text and check it's valid
        if output and "choices" in output and len(output["choices"]) > 0:
            generated_text = output["choices"][0]["text"].strip()
            
            logger.info(f"Successfully generated {len(generated_text)} characters")
            return generated_text
        else:
            logger.error(f"Empty or invalid response from model: {output}")
            return "I'm sorry, I couldn't generate a response. Please try again."
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error. Please try again."

def cleanup():
    """Clean up memory"""
    gc.collect()
    
def main():
    # Get the path to the model
    script_dir = Path(__file__).parent.parent
    model_dir = script_dir / "models" / "mistral-7b-instruct-4bit"
    model_path = model_dir / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    # Check if model exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please run 'python scripts/download_mistral.py' first")
        sys.exit(1)
    
    # Verify model file
    file_size_gb = model_path.stat().st_size / (1024 * 1024 * 1024)
    logger.info(f"Model file size: {file_size_gb:.2f} GB")
    
    print("\n=== Mistral AI Chat ===")
    print("Type 'exit' to quit the chat")
    print("Type 'new' to start a new conversation")
    print("-" * 50)
    
    # Don't use conversation history - just respond to each prompt directly
    # This is simpler and more reliable
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting chat. Goodbye!")
                break
                
            # Check for new conversation command
            if user_input.lower() == 'new':
                print("Starting a new conversation")
                continue
            
            if not user_input.strip():
                print("Please enter a question or prompt.")
                continue
                
            # Generate response to the current input only
            # This avoids conversation history issues
            response = generate_chat_response(
                model_path=model_path,
                prompt=user_input,  # Just use the current input
                max_tokens=2048,
                temperature=0.7
            )
            
            # Print response
            print(f"\nMistral: {response}")
            
    except KeyboardInterrupt:
        print("\nChat terminated by user.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        cleanup()

if __name__ == "__main__":
    main() 