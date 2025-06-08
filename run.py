#!/usr/bin/env python3
"""
OASIS Run Script
Runs the OASIS application using the virtual environment.
"""
import sys
import subprocess
import argparse
from pathlib import Path

def get_venv_python():
    """Get the path to Python executable in the virtual environment."""
    if sys.platform == "win32":
        return Path(".venv") / "Scripts" / "python.exe"
    else:
        return Path(".venv") / "bin" / "python"

def main():
    """Run OASIS using the virtual environment."""
    parser = argparse.ArgumentParser(description="Run OASIS with virtual environment")
    parser.add_argument("--backend", "-b", action="store_true", 
                       help="Run backend only (CLI mode)")
    parser.add_argument("--frontend", "-f", action="store_true", 
                       help="Run frontend only (GUI mode)")
    
    args = parser.parse_args()
    
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print("❌ Virtual environment not found!")
        print("Please run setup.py first to create the virtual environment:")
        print("cd backend && python setup.py")
        return 1

    print("🚀 Starting OASIS...")
    try:
        # Build command arguments
        cmd_args = [str(venv_python), "main.py"]
        if args.backend:
            cmd_args.append("--backend")
        elif args.frontend:
            cmd_args.append("--frontend")
        
        # Run main.py using the virtual environment Python
        subprocess.run(cmd_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running OASIS: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n✋ OASIS stopped by user")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 