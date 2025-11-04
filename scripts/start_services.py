#!/usr/bin/env python3
"""Startup script for MotionMatch MVP"""
import os
import sys
import time
import subprocess
import logging
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required services are running"""
    try:
        # Check if we can import required packages
        import torch
        import transformers
        import pymilvus
        import redis
        logger.info("✓ All Python dependencies available")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        logger.error("Run: pip install -r requirements.txt")
        return False

def check_services():
    """Check if external services are available"""
    services_ok = True
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        logger.info("✓ Redis is running")
    except Exception as e:
        logger.error(f"✗ Redis not available: {e}")
        logger.error("Start with: docker-compose up -d redis")
        services_ok = False
    
    # Check Milvus
    try:
        from pymilvus import connections
        connections.connect("default", host="localhost", port="19530")
        logger.info("✓ Milvus is running")
        connections.disconnect("default")
    except Exception as e:
        logger.error(f"✗ Milvus not available: {e}")
        logger.error("Start with: docker-compose up -d")
        services_ok = False
    
    return services_ok

def setup_environment():
    """Setup environment variables"""
    # Set CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            os.environ["CUDA_AVAILABLE"] = "true"
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            os.environ["CUDA_AVAILABLE"] = "false"
            logger.info("⚠ CUDA not available, using CPU")
    except:
        os.environ["CUDA_AVAILABLE"] = "false"
        logger.info("⚠ PyTorch not available, using CPU")
    
    # Set other environment variables
    os.environ.setdefault("MILVUS_HOST", "localhost")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

def start_api_server():
    """Start the FastAPI server"""
    logger.info("Starting MotionMatch API server...")
    
    try:
        import uvicorn
        from motionmatch.api.main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("🚀 Starting MotionMatch MVP")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Check services
    if not check_services():
        print("\n💡 To start required services:")
        print("   docker-compose up -d")
        print("   # Wait 30-60 seconds for services to start")
        print("   python start.py")
        sys.exit(1)
    
    print("\n✅ All systems ready!")
    print("🌐 Web interface: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("🔧 Health check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start server
    start_api_server()

if __name__ == "__main__":
    main()