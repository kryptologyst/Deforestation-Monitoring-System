#!/usr/bin/env python3
"""Setup script for deforestation monitoring system."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🌲 Deforestation Monitoring System Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    directories = [
        "data/raw", "data/processed", "data/external",
        "assets/models", "assets/plots", "assets/maps",
        "logs", "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Run quick test
    print("\n🧪 Running quick test...")
    if run_command("python demo/quick_demo.py", "Quick system test"):
        print("✅ System test passed")
    else:
        print("⚠️  System test failed, but setup completed")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the interactive demo: streamlit run demo/streamlit_app.py")
    print("2. Train models: python scripts/train_models.py")
    print("3. Run tests: pytest tests/")
    print("\nAuthor: kryptologyst - https://github.com/kryptologyst")


if __name__ == "__main__":
    main()
