#!/usr/bin/env python3
"""
Model Management Utility
-----------------------
This script helps manage the downloaded models in your project.
"""
import os
import sys
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_models_dir():
    """Get the models directory path"""
    script_dir = Path(__file__).parent.parent
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir

def list_models():
    """List all downloaded models and their sizes"""
    models_dir = get_models_dir()
    
    # Find all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("No models found. Use the download scripts to get models.")
        return
    
    print("\n=== Downloaded Models ===")
    print(f"{'Model Name':<30} {'Size':<15} {'Files':<10}")
    print("-" * 55)
    
    for model_dir in model_dirs:
        # Calculate total size
        total_size = 0
        files = 0
        for item in model_dir.glob('**/*'):
            if item.is_file():
                total_size += item.stat().st_size
                files += 1
        
        # Format size as GB or MB
        if total_size > 1024*1024*1024:
            size_str = f"{total_size/(1024*1024*1024):.2f} GB"
        else:
            size_str = f"{total_size/(1024*1024):.2f} MB"
            
        print(f"{model_dir.name:<30} {size_str:<15} {files:<10}")
    
    print("\nUse 'python scripts/manage_models.py clean <model_name>' to remove a model")

def clean_model(model_name):
    """Remove a specific model"""
    models_dir = get_models_dir()
    model_dir = models_dir / model_name
    
    if not model_dir.exists():
        print(f"Model '{model_name}' not found!")
        return
    
    # Calculate size before deletion
    total_size = sum(f.stat().st_size for f in model_dir.glob('**/*') if f.is_file())
    size_gb = total_size / (1024*1024*1024)
    
    # Confirm deletion
    confirm = input(f"Are you sure you want to delete {model_name} ({size_gb:.2f} GB)? (y/n): ")
    
    if confirm.lower() == 'y':
        try:
            shutil.rmtree(model_dir)
            print(f"Model '{model_name}' deleted successfully. {size_gb:.2f} GB freed.")
        except Exception as e:
            print(f"Error deleting model: {str(e)}")
    else:
        print("Deletion cancelled.")

def clean_all_models():
    """Remove all downloaded models"""
    models_dir = get_models_dir()
    
    # Calculate total size of all models
    total_size = 0
    for item in models_dir.glob('**/*'):
        if item.is_file():
            total_size += item.stat().st_size
    
    size_gb = total_size / (1024*1024*1024)
    
    if size_gb < 0.01:
        print("No models found to delete.")
        return
    
    # Confirm deletion
    confirm = input(f"Are you sure you want to delete ALL models ({size_gb:.2f} GB)? (y/n): ")
    
    if confirm.lower() == 'y':
        try:
            # Delete all subdirectories in models
            for item in models_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
            print(f"All models deleted successfully. {size_gb:.2f} GB freed.")
        except Exception as e:
            print(f"Error deleting models: {str(e)}")
    else:
        print("Deletion cancelled.")

def main():
    # Parse command line arguments
    if len(sys.argv) == 1 or sys.argv[1] == "list":
        list_models()
    elif sys.argv[1] == "clean" and len(sys.argv) == 3:
        clean_model(sys.argv[2])
    elif sys.argv[1] == "clean-all":
        clean_all_models()
    else:
        print("Usage:")
        print("  python scripts/manage_models.py list              # List all models")
        print("  python scripts/manage_models.py clean <model>     # Delete a specific model")
        print("  python scripts/manage_models.py clean-all         # Delete all models")

if __name__ == "__main__":
    main() 