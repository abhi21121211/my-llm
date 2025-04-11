#scripts/model_manager.py
import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path
import platform
import psutil

# Configuration for available models - CPU compatible versions with proper directory mappings
MODELS = {
    "tiny-gpt2": {
        "name": "Tiny-GPT2",
        "size": "50MB",
        "ram": "2GB",
        "download_script": "download_tiny_gpt2.py",
        "run_script": "run_tiny_gpt2.py",
        "description": "Very small text generation model",
        "model_dir": "tiny-gpt2"
    },
    "phi3-cpu": {
        "name": "Phi-3 Mini (CPU)",
        "size": "4.0GB",
        "ram": "8GB",
        "download_script": "download_phi3_cpu.py",
        "run_script": "run_phi3_cpu.py",
        "description": "Microsoft's compact model optimized for CPU",
        "model_dir": "phi3-mini-cpu"  # Actual directory name
    },
    "llama-gguf": {
        "name": "Llama 2 7B GGUF",
        "size": "4.0GB",
        "ram": "6GB",
        "download_script": "download_llama_gguf.py",
        "run_script": "run_llama_gguf.py",
        "description": "Meta's model in CPU-optimized GGUF format",
        "model_dir": "llama-gguf"  # Actual directory name
    },
    "llama3-gguf": {
        "name": "Llama 3 8B GGUF",
        "size": "4.5GB", 
        "ram": "8GB",
        "download_script": "download_llama3_gguf.py",
        "run_script": "run_llama3_gguf.py",
        "description": "Meta's model in CPU-optimized GGUF format",
        "model_dir": "llama3-8b-gguf"  # This should match the SAVE_DIR in download script
    },
    "mistral-gguf": {
        "name": "Mistral 7B GGUF",
        "size": "4.1GB",
        "ram": "6GB",
        "download_script": "download_mistral_gguf.py",
        "run_script": "run_mistral_gguf.py",
        "description": "High-quality model in GGUF format",
        "model_dir": "mistral-7b-instruct-4bit"  # Actual directory name
    },
}

def get_system_info():
    """Get system information for recommendations"""
    total_ram = round(psutil.virtual_memory().total / (1024 ** 3))  # GB
    free_ram = round(psutil.virtual_memory().available / (1024 ** 3))  # GB
    cpu_count = psutil.cpu_count(logical=True)
    
    return {
        "total_ram": total_ram,
        "free_ram": free_ram,
        "cpu_count": cpu_count,
        "os": platform.system()
    }

def recommend_models(system_info):
    """Recommend models based on system capabilities"""
    recommended = []
    
    # Simple recommendation logic based on available RAM
    for model_id, model in MODELS.items():
        required_ram = int(model["ram"].replace("GB", ""))
        
        # Add model if we have at least 2GB more RAM than required
        if system_info["free_ram"] >= required_ram + 2:
            recommended.append(model_id)
    
    return recommended

def list_models(installed_only=False, system_info=None):
    """List all available models with details"""
    print("\nAvailable Models (CPU-Compatible):")
    print("=================================")
    
    models_dir = Path("models")
    recommended = recommend_models(system_info) if system_info else []
    
    for model_id, model in MODELS.items():
        model_path = models_dir / model["model_dir"]
        is_installed = model_path.exists()
        
        if installed_only and not is_installed:
            continue
        
        status = "✓ [Installed]" if is_installed else "✗ [Not Installed]"
        recommendation = " ⭐ [Recommended]" if model_id in recommended else ""
        
        print(f"{model['name']} ({model_id})")
        print(f"  Description: {model['description']}")
        print(f"  Size: {model['size']}, RAM: {model['ram']}")
        print(f"  Status: {status}{recommendation}")
        print()

def download_model(model_id):
    """Download the specified model"""
    if model_id not in MODELS:
        print(f"Model '{model_id}' not found in available models.")
        return False
    
    model = MODELS[model_id]
    script_path = f"scripts/{model['download_script']}"
    
    if not os.path.exists(script_path):
        print(f"Download script not found: {script_path}")
        return False
    
    print(f"Downloading {model['name']}...")
    result = subprocess.run([sys.executable, script_path], check=False)
    
    return result.returncode == 0

def run_model(model_id, args=None):
    """Run the specified model"""
    if model_id not in MODELS:
        print(f"Model '{model_id}' not found in available models.")
        return False
    
    model = MODELS[model_id]
    script_path = f"scripts/{model['run_script']}"
    
    if not os.path.exists(script_path):
        print(f"Run script not found: {script_path}")
        return False
    
    # Check if model is installed using the correct directory name
    model_dir = Path("models") / model["model_dir"]
    if not model_dir.exists():
        print(f"Model '{model_id}' is not installed. Please download it first.")
        choice = input("Download now? (y/n): ").lower()
        if choice == 'y':
            success = download_model(model_id)
            if not success:
                return False
        else:
            return False
    
    # Prepare command
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    # Run the model
    print(f"Running {model['name']}...")
    subprocess.run(cmd, check=False)
    
    return True

def remove_model(model_id):
    """Remove the specified model from disk"""
    if model_id not in MODELS:
        print(f"Model '{model_id}' not found in available models.")
        return False
    
    model = MODELS[model_id]
    model_dir = Path("models") / model["model_dir"]
    
    if not model_dir.exists():
        print(f"Model '{model_id}' is not installed.")
        return False
    
    print(f"Removing {model['name']}...")
    shutil.rmtree(model_dir)
    print(f"Model {model['name']} removed successfully.")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="CPU-Only LLM Model Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--installed", action="store_true", help="Show only installed models")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model_id", help="ID of the model to download")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a model")
    run_parser.add_argument("model_id", help="ID of the model to run")
    run_parser.add_argument("args", nargs="*", help="Additional arguments to pass to the run script")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a model")
    remove_parser.add_argument("model_id", help="ID of the model to remove")
    
    # System info command
    subparsers.add_parser("sysinfo", help="Show system information and recommendations")
    
    args = parser.parse_args()
    
    if args.command == "list":
        system_info = get_system_info() if not args.installed else None
        list_models(args.installed, system_info)
    
    elif args.command == "download":
        download_model(args.model_id)
    
    elif args.command == "run":
        run_model(args.model_id, args.args)
    
    elif args.command == "remove":
        remove_model(args.model_id)
    
    elif args.command == "sysinfo":
        info = get_system_info()
        print("\nSystem Information:")
        print("==================")
        print(f"OS: {info['os']}")
        print(f"CPU Cores: {info['cpu_count']}")
        print(f"Total RAM: {info['total_ram']} GB")
        print(f"Available RAM: {info['free_ram']} GB")
        
        print("\nRecommended Models for Your System:")
        print("=================================")
        recommended = recommend_models(info)
        if recommended:
            for model_id in recommended:
                model = MODELS[model_id]
                print(f"- {model['name']} ({model_id}): {model['description']}")
        else:
            print("No models recommended for your current system.")
            print("Try closing other applications to free up more memory.")
    
    else:
        # Show help if no command specified
        parser.print_help()

if __name__ == "__main__":
    main()