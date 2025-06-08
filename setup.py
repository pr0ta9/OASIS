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
import platform
import shutil

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

def get_system_info():
    """Get system information."""
    system = platform.system().lower()
    architecture = platform.machine()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    print(f"🖥️  System: {platform.system()} {architecture}")
    print(f"🐍 Python: {python_version}")
    return system

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Your version: {sys.version}")
        return False
    
    print(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def find_python_executable():
    """Find the correct Python executable for the current system."""
    # Try different Python executable names
    python_names = ["python3", "python", "py"]
    
    for name in python_names:
        if shutil.which(name):
            try:
                # Verify it's Python 3.8+
                result = subprocess.run([name, "--version"], capture_output=True, text=True)
                if result.returncode == 0 and "Python 3." in result.stdout:
                    version_str = result.stdout.strip().split()[1]
                    major, minor = map(int, version_str.split('.')[:2])
                    if major == 3 and minor >= 8:
                        return name
            except:
                continue
    
    return sys.executable

def check_and_install_system_dependencies():
    """Check and install system dependencies like tkinter."""
    print("\n🔧 Checking system dependencies...")
    
    # Check if tkinter is available
    try:
        import tkinter
        print("✓ tkinter is available")
        return True
    except ImportError:
        print("❌ tkinter not found, attempting to install...")
        
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            return install_tkinter_macos()
        elif system == "linux":
            return install_tkinter_linux()
        elif system == "windows":
            return install_tkinter_windows()
        else:
            print(f"❌ Unsupported system: {system}")
            return False

def install_tkinter_macos():
    """Install tkinter on macOS using Homebrew."""
    print("🍺 Installing tkinter via Homebrew...")
    
    # Get current Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"🐍 Detected Python version: {python_version}")
    
    # Check if Homebrew is installed
    if not shutil.which("brew"):
        print("❌ Homebrew not found. Installing Homebrew first...")
        try:
            # Install Homebrew
            install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            subprocess.run(install_cmd, shell=True, check=True)
            print("✓ Homebrew installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Homebrew automatically")
            print("Please install Homebrew manually:")
            print('   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
            return False
    
    try:
        # First, try to install python-tk for the current Python version
        version_specific_commands = [
            ["brew", "install", f"python-tk@{python_version}"],
            ["brew", "install", "python-tk"],
            ["brew", "install", "tcl-tk"]
        ]
        
        for cmd in version_specific_commands:
            try:
                print(f"🔧 Trying: {' '.join(cmd)}")
                subprocess.check_call(cmd)
                print(f"✓ Installed {cmd[-1]} successfully")
                break
            except subprocess.CalledProcessError:
                print(f"❌ Failed: {' '.join(cmd)}")
                continue
        
        # Check if we need to reinstall Python with tkinter support
        print("🔧 Checking if Python was installed via Homebrew...")
        try:
            # Check if current Python is from Homebrew
            python_path = sys.executable
            if "/opt/homebrew/" in python_path or "/usr/local/" in python_path:
                print("✓ Python is from Homebrew, trying to reinstall with tkinter support...")
                subprocess.check_call(["brew", "reinstall", f"python@{python_version}"])
                print("✓ Python reinstalled with tkinter support")
            else:
                print("ℹ️  Python is not from Homebrew (system Python or other source)")
                print("🔧 Installing Homebrew Python with tkinter support...")
                subprocess.check_call(["brew", "install", f"python@{python_version}"])
                print("✓ Homebrew Python installed")
                print("ℹ️  You may need to use Homebrew Python instead:")
                print(f"   /opt/homebrew/bin/python{python_version}")
                print(f"   or /usr/local/bin/python{python_version}")
        except subprocess.CalledProcessError:
            print("ℹ️  Could not reinstall Python, continuing with tkinter installation...")
        
        # Verify installation by trying multiple approaches
        verification_attempts = [
            lambda: __import__('tkinter'),
            lambda: test_tkinter_import(),
            lambda: subprocess.check_call([sys.executable, "-c", "import tkinter; print('tkinter works!')"])
        ]
        
        for attempt in verification_attempts:
            try:
                attempt()
                print("✓ tkinter is now available")
                return True
            except:
                continue
        
        # If still not working, provide comprehensive troubleshooting
        print("❌ tkinter installation completed but still not importable")
        print("\n🔧 Troubleshooting steps:")
        print("1. Try restarting your terminal")
        print("2. Check your Python installation:")
        print(f"   which python3 -> {shutil.which('python3')}")
        print(f"   python3 --version -> Python {python_version}")
        print("3. Try using Homebrew Python:")
        print(f"   /opt/homebrew/bin/python{python_version}")
        print(f"   /usr/local/bin/python{python_version}")
        print("4. Or install Python via Homebrew with tkinter support:")
        print(f"   brew install python@{python_version}")
        print("5. Alternative: Use Anaconda/Miniconda which includes tkinter")
        
        return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install tkinter via Homebrew: {e}")
        print("ℹ️  Manual installation options:")
        print(f"1. brew install python-tk@{python_version}")
        print("2. brew install python-tk")
        print("3. brew install tcl-tk")
        print(f"4. brew install python@{python_version}")
        print("5. Use Anaconda/Miniconda which includes tkinter")
        return False

def test_tkinter_import():
    """Test tkinter import in a subprocess to avoid caching issues."""
    try:
        result = subprocess.run([sys.executable, "-c", "import tkinter; tkinter.Tk().destroy()"], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def install_tkinter_linux():
    """Install tkinter on Linux."""
    print("🐧 Installing tkinter on Linux...")
    
    # Detect Linux distribution
    distro_commands = {
        "ubuntu": ["sudo", "apt-get", "update", "&&", "sudo", "apt-get", "install", "-y", "python3-tk"],
        "debian": ["sudo", "apt-get", "update", "&&", "sudo", "apt-get", "install", "-y", "python3-tk"],
        "centos": ["sudo", "yum", "install", "-y", "tkinter"],
        "rhel": ["sudo", "yum", "install", "-y", "tkinter"],
        "fedora": ["sudo", "dnf", "install", "-y", "python3-tkinter"],
        "arch": ["sudo", "pacman", "-S", "--noconfirm", "tk"],
        "manjaro": ["sudo", "pacman", "-S", "--noconfirm", "tk"]
    }
    
    # Try to detect distribution
    distro = None
    try:
        with open("/etc/os-release", "r") as f:
            content = f.read().lower()
            for name in distro_commands:
                if name in content:
                    distro = name
                    break
    except FileNotFoundError:
        pass
    
    # Try the detected distribution first, then fallback to common commands
    commands_to_try = []
    if distro and distro in distro_commands:
        commands_to_try.append(distro_commands[distro])
    
    # Add fallback commands
    fallback_commands = [
        ["sudo", "apt-get", "update"],
        ["sudo", "apt-get", "install", "-y", "python3-tk"],
        ["sudo", "yum", "install", "-y", "tkinter"],
        ["sudo", "dnf", "install", "-y", "python3-tkinter"],
        ["sudo", "pacman", "-S", "--noconfirm", "tk"]
    ]
    commands_to_try.extend(fallback_commands)
    
    for cmd in commands_to_try:
        try:
            if "&&" in cmd:
                # Handle compound commands
                subprocess.run(" ".join(cmd), shell=True, check=True)
            else:
                subprocess.run(cmd, check=True)
            
            # Test if tkinter is now available
            try:
                import tkinter
                print("✓ tkinter installed successfully")
                return True
            except ImportError:
                continue
                
        except subprocess.CalledProcessError:
            continue
    
    print("❌ Failed to install tkinter automatically")
    print("Please install tkinter manually for your Linux distribution:")
    print("  Ubuntu/Debian: sudo apt-get install python3-tk")
    print("  CentOS/RHEL: sudo yum install tkinter")
    print("  Fedora: sudo dnf install python3-tkinter")
    print("  Arch/Manjaro: sudo pacman -S tk")
    return False

def install_tkinter_windows():
    """Handle tkinter installation on Windows."""
    print("🪟 Checking tkinter on Windows...")
    
    # On Windows, tkinter should be included with Python
    print("ℹ️  On Windows, tkinter is typically included with Python installations.")
    print("If tkinter is missing, you may need to:")
    print("1. Reinstall Python from python.org with 'tcl/tk and IDLE' option checked")
    print("2. Or install Python via Microsoft Store")
    print("3. Or use a Python distribution that includes tkinter (like Anaconda)")
    
    # Check if we can install it via pip (some distributions support this)
    try:
        python_exe = find_python_executable()
        subprocess.check_call([python_exe, "-m", "pip", "install", "tk"])
        print("✓ Installed tkinter via pip")
        
        # Verify installation
        try:
            import tkinter
            print("✓ tkinter is now available")
            return True
        except ImportError:
            pass
    except subprocess.CalledProcessError:
        pass
    
    print("❌ Could not install tkinter automatically on Windows")
    print("Please reinstall Python with tkinter support or use an alternative Python distribution")
    return False

def create_virtual_environment():
    """Create a virtual environment for the project."""
    venv_path = Path(".venv")
    
    # Check if virtual environment exists and is valid
    if venv_path.exists():
        print("📁 Virtual environment directory exists, checking validity...")
        
        # Check if the Python executable in venv is still valid
        venv_python = get_venv_python()
        
        if venv_python.exists():
            try:
                # Test if the venv Python is working
                result = subprocess.run([str(venv_python), "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("✓ Virtual environment is valid and working")
                    return True
                else:
                    print("❌ Virtual environment Python is not working")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                print("❌ Virtual environment Python is broken or missing")
        else:
            print("❌ Virtual environment Python executable not found")
        
        # If we get here, the venv is invalid - remove it
        print("🔧 Removing invalid virtual environment...")
        try:
            import shutil
            shutil.rmtree(venv_path)
            print("✓ Invalid virtual environment removed")
        except Exception as e:
            print(f"❌ Failed to remove invalid virtual environment: {e}")
            print("Please manually delete the .venv directory and run setup again")
            return False
    
    print("\n🔧 Creating virtual environment...")
    
    try:
        python_exe = find_python_executable()
        print(f"🐍 Using Python: {python_exe}")
        
        # Use the found Python executable to create venv
        subprocess.check_call([python_exe, "-m", "venv", ".venv"])
        print("✓ Virtual environment created successfully")
        
        # Verify the new venv works
        venv_python = get_venv_python()
        if venv_python.exists():
            try:
                result = subprocess.run([str(venv_python), "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"✓ New virtual environment verified: {result.stdout.strip()}")
                    return True
                else:
                    print("❌ New virtual environment verification failed")
                    return False
            except Exception as e:
                print(f"❌ Failed to verify new virtual environment: {e}")
                return False
        else:
            print("❌ New virtual environment Python not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        # Fallback to built-in venv module
        try:
            print("🔧 Trying fallback venv creation method...")
            venv.create(venv_path, with_pip=True)
            print("✓ Virtual environment created successfully (fallback)")
            
            # Verify fallback venv
            venv_python = get_venv_python()
            if venv_python.exists():
                result = subprocess.run([str(venv_python), "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"✓ Fallback virtual environment verified: {result.stdout.strip()}")
                    return True
            
            return True
        except Exception as e2:
            print(f"❌ Failed to create virtual environment (fallback): {e2}")
            return False

def get_venv_python():
    """Get the path to Python executable in the virtual environment."""
    system = platform.system().lower()
    
    if system == "windows":
        # Try different possible paths on Windows
        possible_paths = [
            Path(".venv") / "Scripts" / "python.exe",
            Path(".venv") / "Scripts" / "python3.exe",
            Path(".venv") / "bin" / "python.exe",
            Path(".venv") / "bin" / "python3.exe"
        ]
    else:
        # Unix-like systems (macOS, Linux)
        possible_paths = [
            Path(".venv") / "bin" / "python",
            Path(".venv") / "bin" / "python3",
            Path(".venv") / "Scripts" / "python",
            Path(".venv") / "Scripts" / "python3"
        ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Fallback to first option
    if system == "windows":
        return Path(".venv") / "Scripts" / "python.exe"
    else:
        return Path(".venv") / "bin" / "python"

def install_requirements():
    """Install required packages in the virtual environment."""
    print("\n📦 Installing requirements in virtual environment...")
    
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print("❌ Virtual environment Python not found")
        print(f"Expected at: {venv_python}")
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
        with open(example_file, 'r', encoding='utf-8') as src, open(env_file, 'w', encoding='utf-8') as dst:
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
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Basic tests passed")
            return True
        else:
            print("❌ Basic tests failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def print_activation_instructions():
    """Print platform-specific virtual environment activation instructions."""
    system = platform.system().lower()
    
    print("3. Activate the virtual environment:")
    
    if system == "windows":
        # Windows has multiple shells
        print("   Command Prompt: .venv\\Scripts\\activate.bat")
        print("   PowerShell: .venv\\Scripts\\Activate.ps1")
        print("   Git Bash: source .venv/Scripts/activate")
    else:
        # Unix-like systems (macOS, Linux)
        print("   source .venv/bin/activate")

def main():
    """Main setup function."""
    print_banner()
    
    print("Setting up OASIS...")
    print("=" * 50)
    
    # Get system info
    system = get_system_info()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check and install system dependencies
    if not check_and_install_system_dependencies():
        print("⚠️  System dependency check failed, continuing with setup...")
        print("You may need to install tkinter manually for the GUI to work.")
    
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
    print_activation_instructions()
    print("4. Run the application: python main.py")
    print("\nFor help, see README.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 