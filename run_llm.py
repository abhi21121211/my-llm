#!/usr/bin/env python3
"""
My LLM - Main Menu
------------------
This script provides a menu to access all features of the project.
"""
import os
import sys
import subprocess
from pathlib import Path

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header"""
    clear_screen()
    print("=" * 60)
    print("                MY LLM - LOCAL LANGUAGE MODELS")
    print("=" * 60)
    print("Run open-source language models locally on your computer")
    print("-" * 60)

def print_menu():
    """Print the main menu options"""
    print("\nMAIN MENU:")
    print("1. Run Tiny-GPT2 (Text Generation)")
    print("2. Run CodeLlama (Code Generation)")
    print("3. Run Mistral (Chat)")
    print("4. Download Models")
    print("5. Manage Models (List/Delete)")
    print("6. Show System Info")
    print("q. Quit")
    
    return input("\nEnter your choice: ")

def run_script(script_path):
    """Run a Python script"""
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
    except FileNotFoundError:
        print(f"Script not found: {script_path}")
    
    input("\nPress Enter to return to the main menu...")

def download_menu():
    """Show the download menu"""
    while True:
        clear_screen()
        print("=" * 60)
        print("                DOWNLOAD MODELS")
        print("=" * 60)
        print("\nSELECT MODEL TO DOWNLOAD:")
        print("1. Tiny-GPT2 (~50MB)")
        print("2. CodeLlama 7B (~3.8GB)")
        print("3. Mistral 7B Instruct (~4.1GB)")
        print("b. Back to Main Menu")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            run_script("scripts/download_model.py")
        elif choice == '2':
            run_script("scripts/download_codellama.py")
        elif choice == '3':
            run_script("scripts/download_mistral.py")
        elif choice.lower() == 'b':
            break
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")

def show_system_info():
    """Display system information"""
    clear_screen()
    print("=" * 60)
    print("                SYSTEM INFORMATION")
    print("=" * 60)
    
    # Try to import necessary packages
    try:
        import platform
        import psutil
        import torch
        
        # System info
        print(f"\nOS: {platform.system()} {platform.version()}")
        print(f"Python: {platform.python_version()}")
        
        # CPU info
        print(f"\nCPU: {platform.processor()}")
        print(f"CPU Cores: {psutil.cpu_count(logical=False)} Physical, {psutil.cpu_count()} Logical")
        
        # Memory info
        mem = psutil.virtual_memory()
        print(f"\nTotal RAM: {mem.total / (1024**3):.2f} GB")
        print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
        
        # Disk info
        disk = psutil.disk_usage('/')
        print(f"\nDisk Space: {disk.total / (1024**3):.2f} GB Total, {disk.free / (1024**3):.2f} GB Free")
        
        # PyTorch info
        print(f"\nPyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        try:
            if torch.xpu.is_available():
                print("Intel XPU (GPU) Available: Yes")
        except:
            print("Intel XPU (GPU) Available: No")
            
    except ImportError as e:
        print(f"Could not import required packages: {e}")
    
    input("\nPress Enter to return to the main menu...")

def main():
    """Main function to run the menu"""
    while True:
        print_header()
        choice = print_menu()
        
        if choice == '1':
            run_script("scripts/run_inference.py")
        elif choice == '2':
            run_script("scripts/run_codellama.py")
        elif choice == '3':
            run_script("scripts/run_mistral.py")
        elif choice == '4':
            download_menu()
        elif choice == '5':
            run_script("scripts/manage_models.py")
        elif choice == '6':
            show_system_info()
        elif choice.lower() == 'q':
            print("\nExiting. Thank you for using My LLM!")
            break
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main() 