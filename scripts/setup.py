"""Setup script for MotionMatch MVP"""
import os
import subprocess
import sys

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 11):
        print("Error: Python 3.11 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = ["storage", "temp", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_docker():
    """Check if Docker is available"""
    try:
        subprocess.check_output(["docker", "--version"], stderr=subprocess.STDOUT)
        print("✓ Docker is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Docker not found - you'll need to install Milvus and Redis manually")
        return False

def start_services():
    """Start Docker services"""
    if not check_docker():
        return False
    
    print("Starting Docker services...")
    try:
        subprocess.check_call(["docker-compose", "up", "-d"])
        print("✓ Docker services started")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error starting services: {e}")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} (count: {gpu_count})")
            return True
        else:
            print("⚠ No GPU detected - will use CPU (slower performance)")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet - GPU check will be done after installation")
        return False

def main():
    """Main setup function"""
    print("=== MotionMatch MVP Setup ===\n")
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install requirements
    install_requirements()
    
    # Check GPU after PyTorch installation
    gpu_available = check_gpu()
    
    # Start Docker services
    services_started = start_services()
    
    print("\n=== Setup Complete ===")
    print("Next steps:")
    print("1. Wait for Docker services to start (30-60 seconds)")
    print("2. Set environment variables:")
    if gpu_available:
        print("   export CUDA_AVAILABLE=true")
    else:
        print("   export CUDA_AVAILABLE=false")
    print("3. Start the API server:")
    print("   python api.py")
    print("4. Test the system:")
    print("   python test_client.py")
    print("\nAPI will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()