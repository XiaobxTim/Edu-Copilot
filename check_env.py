#!/usr/bin/env python3
"""
Environment setup script for Edu-Copilot
"""

import os
import subprocess
import sys

def check_and_install_requirements():
    """Check if requirements are installed, install if missing"""
    requirements = [
        "datasets",
        "huggingface_hub"
    ]
    
    print("Checking dependencies...")
    
    for package in requirements:
        try:
            if package == "datasets":
                import datasets
                print(f"✓ {package} is installed (version: {datasets.__version__})")
            elif package == "huggingface_hub":
                import huggingface_hub
                print(f"✓ {package} is installed (version: {huggingface_hub.__version__})")
        except ImportError:
            print(f"✗ {package} is not installed. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully!")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
                return False
    return True

def create_directories():
    """Create necessary project directories"""
    directories = [
        "data",
        "src/data_synthesis",
        "models",
        "logs"
    ]
    
    print("Creating project directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def main():
    """Main setup function"""
    print("=== Edu-Copilot Environment Setup ===")
    
    # Create directories
    create_directories()
    
    # Install requirements
    if check_and_install_requirements():
        print("\n✓ Environment setup completed successfully!")
    else:
        print("\n✗ Environment setup failed. Please install dependencies manually:")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()