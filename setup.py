#!/usr/bin/env python3
"""
OASIS Setup Script
Helps users install and configure OASIS quickly with proper virtual environment.
"""
import os
import sys
import subprocess
from pathlib import Path
import venv

def print_banner():
    """Print the OASIS banner."""
    print("""
  ██████╗  █████╗ ███████╗██╗███████╗
 ██╔═══██╗██╔══██╗██╔════╝██║██╔════╝
 ██║   ██║███████║███████╗██║███████╗
 ██║   ██║██╔══██║╚════██║██║╚════██║
 ╚██████╔╝██║  ██║███████║██║███████║
  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝
  
Opensource AI Small-model Integration System
""")

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Your version: {sys.version}")
        return False
    
    print(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def create_virtual_environment():
    """Create a virtual environment for the project."""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return True
    
    print("\n🔧 Creating virtual environment...")
    
    try:
        venv.create(venv_path, with_pip=True)
        print("✓ Virtual environment created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_venv_python():
    """Get the path to Python executable in the virtual environment."""
    if sys.platform == "win32":
        return Path(".venv") / "Scripts" / "python.exe"
    else:
        return Path(".venv") / "bin" / "python"

def install_requirements():
    """Install required packages in the virtual environment."""
    print("\n📦 Installing requirements in virtual environment...")
    
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print("❌ Virtual environment Python not found")
        return False
    
    try:
        # Upgrade pip first
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_environment():
    """Set up environment file."""
    env_file = Path(".env")
    example_file = Path("env_example.txt")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    if not example_file.exists():
        print("❌ env_example.txt not found")
        return False
    
    try:
        # Copy example to .env
        with open(example_file, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        
        print("✓ Created .env file from example")
        print("⚠️  Please edit .env file and add your GOOGLE_API_KEY")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def run_tests():
    """Run basic tests using virtual environment."""
    print("\n🧪 Running basic tests...")
    
    venv_python = get_venv_python()
    
    try:
        result = subprocess.run([str(venv_python), "test_basic.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Basic tests passed")
            return True
        else:
            print("❌ Basic tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def main():
    """Main setup function."""
    print_banner()
    
    print("Setting up OASIS...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create virtual environment
    if not create_virtual_environment():
        return 1
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Setup environment
    if not setup_environment():
        return 1
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Tests failed, but setup completed. You may need to check dependencies.")
    
    print("\n🎉 OASIS setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file and add your GOOGLE_API_KEY")
    print("2. Get a Gemini API key from: https://ai.google.dev/")
    print("3. Activate the virtual environment:")
    if sys.platform == "win32":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    print("4. Run the application: python main.py")
    print("\nFor help, see README.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 