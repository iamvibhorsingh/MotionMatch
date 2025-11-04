#!/usr/bin/env python3
"""Test start script for MotionMatch MVP - starts system and runs test setup"""
import os
import sys
import time
import subprocess
import logging
import signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestStart:
    """Test start manager for MotionMatch MVP"""
    
    def __init__(self):
        self.processes = []
        self.shutdown_event = threading.Event()
        self.api_process = None
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal, cleaning up...")
        self.shutdown_event.set()
        self.cleanup_processes()
        sys.exit(0)
    
    def cleanup_processes(self):
        """Clean up all started processes"""
        for process in self.processes:
            try:
                if process.poll() is None:  # Process is still running
                    logger.info(f"Terminating process {process.pid}")
                    process.terminate()
                    process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing process {process.pid}")
                process.kill()
            except Exception as e:
                logger.error(f"Error cleaning up process: {e}")
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        try:
            # Check Python packages
            import torch
            import transformers
            import fastapi
            import pymilvus
            logger.info("✓ Python dependencies available")
            
            # Check Docker
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("✓ Docker available")
            
            # Check Docker Compose
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("✓ Docker Compose available")
            
            return True
            
        except ImportError as e:
            logger.error(f"✗ Missing Python dependency: {e}")
            logger.error("Run: pip install -r requirements.txt")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Docker/Docker Compose not available: {e}")
            return False
        except FileNotFoundError as e:
            logger.error(f"✗ Command not found: {e}")
            return False
    
    def setup_environment(self):
        """Setup environment variables"""
        # Set CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                os.environ["CUDA_AVAILABLE"] = "true"
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✓ CUDA available: {gpu_name}")
            else:
                os.environ["CUDA_AVAILABLE"] = "false"
                logger.info("⚠ CUDA not available, using CPU")
        except:
            os.environ["CUDA_AVAILABLE"] = "false"
            logger.info("⚠ PyTorch not available, using CPU")
        
        # Set other environment variables
        os.environ.setdefault("MILVUS_HOST", "localhost")
        os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
        
        logger.info("Environment variables configured")
    
    def start_docker_services(self) -> bool:
        """Start Docker services"""
        logger.info("Starting Docker services...")
        
        try:
            # Check if services are already running
            result = subprocess.run(
                ["docker-compose", "ps", "-q"],
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                logger.info("Docker services already running")
                return True
            
            # Start services
            process = subprocess.Popen(
                ["docker-compose", "up", "-d"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=120)
            
            if process.returncode == 0:
                logger.info("✓ Docker services started successfully")
                return True
            else:
                logger.error(f"✗ Failed to start Docker services: {stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("✗ Docker service startup timed out")
            return False
        except Exception as e:
            logger.error(f"✗ Error starting Docker services: {e}")
            return False
    
    def wait_for_services(self, max_wait: int = 60) -> bool:
        """Wait for Docker services to be ready"""
        logger.info("Waiting for Docker services to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                # Check Redis
                import redis
                r = redis.Redis(host='localhost', port=6379, db=0)
                r.ping()
                
                # Check Milvus
                from pymilvus import connections
                connections.connect("default", host="localhost", port="19530")
                connections.disconnect("default")
                
                logger.info("✓ All services are ready")
                return True
                
            except Exception:
                pass
            
            if self.shutdown_event.is_set():
                return False
                
            logger.info("Services not ready yet, waiting...")
            time.sleep(5)
        
        logger.error(f"Services did not become ready within {max_wait} seconds")
        return False
    
    def start_api_server(self) -> subprocess.Popen:
        """Start the API server"""
        logger.info("Starting MotionMatch API server...")
        
        try:
            process = subprocess.Popen(
                [sys.executable, "api.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes.append(process)
            self.api_process = process
            
            # Start a thread to monitor API output
            def monitor_api():
                for line in process.stdout:
                    if not self.shutdown_event.is_set():
                        logger.info(f"API: {line.strip()}")
            
            threading.Thread(target=monitor_api, daemon=True).start()
            
            logger.info(f"✓ API server started (PID: {process.pid})")
            return process
            
        except Exception as e:
            logger.error(f"✗ Failed to start API server: {e}")
            return None
    
    def wait_for_api(self, max_wait: int = 60) -> bool:
        """Wait for API to become available"""
        logger.info("Waiting for API to become available...")
        
        import requests
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    health = response.json()
                    if health.get("status") == "healthy":
                        logger.info("✓ API is healthy and ready")
                        return True
                
            except requests.exceptions.RequestException:
                pass
            
            if self.shutdown_event.is_set():
                return False
            
            # Check if API process is still running
            if self.api_process and self.api_process.poll() is not None:
                logger.error("✗ API process has terminated")
                return False
            
            time.sleep(2)
        
        logger.error(f"API did not become available within {max_wait} seconds")
        return False
    
    def run_test_setup(self) -> bool:
        """Run the test setup script"""
        logger.info("Running test setup...")
        
        try:
            process = subprocess.Popen(
                [sys.executable, "testsetup.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            for line in process.stdout:
                if not self.shutdown_event.is_set():
                    print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                logger.info("✓ Test setup completed successfully")
                return True
            else:
                logger.error(f"✗ Test setup failed with exit code {process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Error running test setup: {e}")
            return False
    
    def run_interactive_mode(self):
        """Run in interactive mode after setup"""
        print("\n" + "="*60)
        print("🎉 MotionMatch MVP is running!")
        print("="*60)
        print("🌐 Web Interface: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔧 Health Check: http://localhost:8000/health")
        print("\nThe system is ready for testing and development.")
        print("Press Ctrl+C to stop all services.")
        print("="*60)
        
        try:
            # Keep the main thread alive
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            self.shutdown_event.set()
    
    def run_full_test_start(self):
        """Run the complete test start process"""
        print("🚀 MotionMatch MVP Test Start")
        print("="*50)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Step 1: Check dependencies
            if not self.check_dependencies():
                return False
            
            # Step 2: Setup environment
            self.setup_environment()
            
            # Step 3: Start Docker services
            if not self.start_docker_services():
                return False
            
            # Step 4: Wait for services to be ready
            if not self.wait_for_services():
                return False
            
            # Step 5: Start API server
            api_process = self.start_api_server()
            if not api_process:
                return False
            
            # Step 6: Wait for API to be ready
            if not self.wait_for_api():
                return False
            
            # Step 7: Run test setup
            if not self.run_test_setup():
                logger.warning("Test setup failed, but system is still running")
            
            # Step 8: Enter interactive mode
            self.run_interactive_mode()
            
            return True
            
        except Exception as e:
            logger.error(f"Test start failed: {e}")
            return False
        finally:
            self.cleanup_processes()

def main():
    """Main function"""
    test_start = TestStart()
    
    try:
        success = test_start.run_full_test_start()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()