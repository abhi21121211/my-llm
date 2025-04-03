import requests
import logging
from pathlib import Path
import os
import json
import torch
import psutil
from transformers import AutoTokenizer

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
    
    if free_disk_gb < 1:
        logger.warning(f"Low disk space: Only {free_disk_gb:.2f} GB available. At least 1 GB recommended.")
    
    if available_ram_gb < 4:
        logger.warning(f"Low RAM: Only {available_ram_gb:.2f} GB available. At least 4 GB recommended.")
    
    return True

def download_file(url, destination):
    """Download a file with progress indication"""
    logger.info(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total = int(response.headers.get('content-length', 0))
    file_size_mb = total / (1024 * 1024)
    
    logger.info(f"File size: {file_size_mb:.2f} MB")
    
    with open(destination, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                # Log progress every 5%
                if total > 0 and downloaded % (total // 20) < 8192:
                    percent = (downloaded / total) * 100
                    logger.info(f"Download progress: {percent:.1f}% ({downloaded/(1024*1024):.2f} MB / {file_size_mb:.2f} MB)")
    
    logger.info(f"Download complete: {destination}")
    return destination

def download_model():
    # Use the tiny GPT-2 model (only about 50MB)
    model_id = "sshleifer/tiny-gpt2"
    output_dir = Path(__file__).parent.parent / "models" / "tiny-gpt2"
    
    try:
        # Check system resources
        check_system_resources()
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Setting up model {model_id} in {output_dir}")
        
        # Set up model files to download
        base_url = f"https://huggingface.co/{model_id}/resolve/main"
        files_to_download = {
            "config.json": f"{base_url}/config.json",
            "pytorch_model.bin": f"{base_url}/pytorch_model.bin",
            "tokenizer_config.json": f"{base_url}/tokenizer_config.json",
            "vocab.json": f"{base_url}/vocab.json",
            "merges.txt": f"{base_url}/merges.txt"
        }
        
        # Download each file
        for filename, url in files_to_download.items():
            file_path = output_dir / filename
            if not file_path.exists():
                download_file(url, file_path)
            else:
                logger.info(f"File already exists: {file_path}")
        
        # Download and save tokenizer
        logger.info("Setting up tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(str(output_dir))
        logger.info("Tokenizer saved successfully!")
        
        # Verify all files were downloaded
        logger.info("Verifying downloaded files...")
        for filename in files_to_download.keys():
            file_path = output_dir / filename
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"Verified: {filename} exists ({file_size_mb:.2f} MB)")
            else:
                logger.error(f"Missing file: {filename}")
                raise FileNotFoundError(f"Required file not found: {filename}")
        
        logger.info("Model setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    download_model() 