#!/usr/bin/env python3
"""
OASIS Run Script
Runs the OASIS application using the virtual environment.
"""
import sys
import subprocess
from pathlib import Path

def get_venv_python():
    """Get the path to Python executable in the virtual environment."""
    if sys.platform == "win32":
        return Path(".venv") / "Scripts" / "python.exe"
    else:
        return Path(".venv") / "bin" / "python"

def main():
    """Run OASIS using the virtual environment."""
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print("❌ Virtual environment not found!")
        print("Please run setup.py first to create the virtual environment:")
        print("python setup.py")
        return 1
    
    print("🚀 Starting OASIS...")
    try:
        # Run main.py using the virtual environment Python
        subprocess.run([str(venv_python), "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running OASIS: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n✋ OASIS stopped by user")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 