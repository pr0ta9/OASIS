#!/usr/bin/env python3
"""
OASIS Run Script
Runs the OASIS multi-agent system using the virtual environment.
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
    parser = argparse.ArgumentParser(description="Run OASIS Multi-Agent System with virtual environment")
    parser.add_argument("--cli-test", "-t", action="store_true", 
                       help="Run backend in CLI test mode (for development/testing)")
    parser.add_argument("--gui", "-g", action="store_true", 
                       help="Start complete OASIS application (same as default)")
    
    # Legacy support for old flags
    parser.add_argument("--backend", "-b", action="store_true", 
                       help="Run backend only (deprecated, use --cli-test)")
    parser.add_argument("--frontend", "-f", action="store_true", 
                       help="Run frontend only (deprecated, default runs complete app)")
    
    args = parser.parse_args()
    
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print("❌ Virtual environment not found!")
        print("Please run setup.py first to create the virtual environment:")
        print("cd backend && python setup.py")
        return 1

    print("🚀 Starting OASIS Multi-Agent System...")
    try:
        # Build command arguments
        cmd_args = [str(venv_python), "main.py"]
        
        # Handle arguments
        if args.backend or args.cli_test:
            cmd_args.append("--cli-test")
            print("📋 Running in CLI test mode...")
        elif args.frontend:
            print("ℹ️  Note: --frontend is deprecated, running complete application")
        elif args.gui:
            cmd_args.append("--gui")
            print("🖥️  Running complete OASIS application...")
        else:
            print("🖥️  Running complete OASIS application (default)...")
        
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