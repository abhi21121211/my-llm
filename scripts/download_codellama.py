import requests
import logging
from pathlib import Path
import os
import json
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_system_resources():
    """Check if system has enough resources for model download and loading"""
    # Check available disk space
    disk_usage = psutil.disk_usage(os.path.abspath(os.sep))
    free_disk_gb = disk_usage.free / (1024 * 1024 * 1024)
    
    # Check available RAM
    available_ram_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    
    logger.info(f"System resources: {free_disk_gb:.2f} GB free disk space, {available_ram_gb:.2f} GB available RAM")
    
    if free_disk_gb < 2:
        logger.warning(f"Low disk space: Only {free_disk_gb:.2f} GB available. At least 2 GB recommended.")
    
    if available_ram_gb < 8:
        logger.warning(f"Low RAM: Only {available_ram_gb:.2f} GB available. At least 8 GB recommended.")
    
    return True

def download_model():
    # Use CodeLlama 7B in 4-bit quantized form - good balance of capability and size
    model_id = "TheBloke/CodeLlama-7B-Python-GGUF"
    model_file = "codellama-7b-python.Q4_K_M.gguf"  # 4-bit quantized (about 3.8GB)
    output_dir = Path(__file__).parent.parent / "models" / "codellama-7b-python-4bit"
    
    try:
        # Check system resources
        check_system_resources()
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Setting up model {model_id} in {output_dir}")
        
        # We'll use HuggingFace transformers to download
        logger.info("Downloading and setting up tokenizer...")
        from huggingface_hub import hf_hub_download
        
        model_path = output_dir / model_file
        
        # Skip download if file already exists
        if not model_path.exists():
            logger.info(f"Downloading model file: {model_file}")
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=model_file,
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            logger.info(f"Model downloaded to: {downloaded_path}")
        else:
            logger.info(f"Model file already exists at: {model_path}")
            
        # Verify file size
        if model_path.exists():
            file_size_gb = model_path.stat().st_size / (1024 * 1024 * 1024)
            logger.info(f"Model file size: {file_size_gb:.2f} GB")
            
            if file_size_gb < 3.0:  # Expected size for the 4-bit model
                logger.warning(f"Model file seems smaller than expected ({file_size_gb:.2f} GB). Download may be incomplete.")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        logger.info("Model setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    download_model() 