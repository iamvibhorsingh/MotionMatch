#!/usr/bin/env python3
"""Validation script for MotionMatch MVP setup"""
import sys
import importlib
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name}")
        return False

def check_packages():
    """Check all required packages"""
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("torchvision", "torchvision"),
        ("opencv-python", "cv2"),
        ("decord", "decord"),
        ("einops", "einops"),
        ("numpy", "numpy"),
        ("pymilvus", "pymilvus"),
        ("fastapi", "fastapi"),
        ("celery", "celery"),
        ("redis", "redis"),
        ("psycopg2-binary", "psycopg2"),
        ("sqlalchemy", "sqlalchemy"),
        ("boto3", "boto3"),
        ("pandas", "pandas"),
        ("pillow", "PIL"),
        ("tqdm", "tqdm"),
        ("pydantic", "pydantic"),
        ("python-multipart", "multipart"),
        ("uvicorn", "uvicorn"),
        ("fastdtw", "fastdtw"),
        ("scipy", "scipy")
    ]
    
    print("Checking Python packages:")
    all_good = True
    for package, import_name in packages:
        if not check_package(package, import_name):
            all_good = False
    
    return all_good

def check_docker():
    """Check Docker availability"""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Docker: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Docker not found")
        return False

def check_docker_compose():
    """Check Docker Compose availability"""
    try:
        result = subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Docker Compose: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Docker Compose not found")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU: {gpu_name} (count: {gpu_count})")
            return True
        else:
            print("⚠ No GPU detected (will use CPU)")
            return False
    except ImportError:
        print("⚠ Cannot check GPU (PyTorch not installed)")
        return False

def check_files():
    """Check if all required files exist"""
    required_files = [
        "requirements.txt",
        "config.py",
        "models.py",
        "encoder_service.py",
        "vector_db.py",
        "search_service.py",
        "indexing_service.py",
        "api.py",
        "docker-compose.yml",
        "Dockerfile",
        "start.py",
        "example.py",
        "test_client.py",
        "static/index.html"
    ]
    
    print("Checking required files:")
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            all_good = False
    
    return all_good

def check_directories():
    """Check if required directories exist or can be created"""
    required_dirs = [
        "storage",
        "temp",
        "static",
        "sample_videos"
    ]
    
    print("Checking/creating directories:")
    for dir_path in required_dirs:
        try:
            Path(dir_path).mkdir(exist_ok=True)
            print(f"✓ {dir_path}")
        except Exception as e:
            print(f"✗ {dir_path}: {e}")
            return False
    
    return True

def check_services():
    """Check if Docker services can be started"""
    print("Checking Docker services (this may take a moment)...")
    
    try:
        # Try to start services
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print("✓ Docker services started successfully")
            
            # Check if services are running
            result = subprocess.run(
                ["docker-compose", "ps"],
                capture_output=True,
                text=True
            )
            
            if "Up" in result.stdout:
                print("✓ Services are running")
                return True
            else:
                print("⚠ Services started but may not be ready")
                return False
        else:
            print(f"✗ Failed to start services: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠ Service startup timed out (may still be starting)")
        return False
    except Exception as e:
        print(f"✗ Error checking services: {e}")
        return False

def main():
    """Main validation function"""
    print("🔍 MotionMatch MVP Setup Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Python Packages", check_packages),
        ("Docker", check_docker),
        ("Docker Compose", check_docker_compose),
        ("GPU Support", check_gpu),
        ("Required Files", check_files),
        ("Directories", check_directories),
    ]
    
    results = {}
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        results[check_name] = check_func()
    
    # Optional service check
    print(f"\nDocker Services (optional):")
    results["Docker Services"] = check_services()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Validation Summary:")
    
    critical_checks = [
        "Python Version",
        "Python Packages", 
        "Required Files",
        "Directories"
    ]
    
    optional_checks = [
        "Docker",
        "Docker Compose",
        "GPU Support",
        "Docker Services"
    ]
    
    critical_passed = all(results.get(check, False) for check in critical_checks)
    optional_passed = sum(results.get(check, False) for check in optional_checks)
    
    if critical_passed:
        print("✅ All critical checks passed!")
        print("   You can run the system with: python start.py")
        
        if optional_passed >= 2:
            print("✅ Most optional features available")
        else:
            print("⚠ Some optional features may not work")
            print("   Install Docker for full functionality")
    else:
        print("❌ Some critical checks failed")
        print("   Please fix the issues above before running the system")
        
        failed_critical = [check for check in critical_checks if not results.get(check, False)]
        print(f"   Failed: {', '.join(failed_critical)}")
    
    print("\n🚀 Next steps:")
    if critical_passed:
        print("   1. python start.py          # Start the system")
        print("   2. python example.py        # Run the demo")
        print("   3. Open http://localhost:8000  # Use web interface")
    else:
        print("   1. Fix the failed checks above")
        print("   2. Run this validation again")
        print("   3. python setup.py          # Automated setup")

if __name__ == "__main__":
    main()