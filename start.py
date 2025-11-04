#!/usr/bin/env python3
"""Quick start script for MotionMatch"""
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set environment
os.environ.setdefault("MILVUS_HOST", "localhost")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

# GPU Optimizations
os.environ.setdefault("BATCH_SIZE", "8")  # Larger batch for RTX 4070 Ti
os.environ.setdefault("USE_MIXED_PRECISION", "true")
os.environ.setdefault("TORCH_COMPILE", "false")  # Set to "true" for PyTorch 2.0+

# PyTorch optimizations
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        os.environ["CUDA_AVAILABLE"] = "true"
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"✓ Batch size: {os.environ.get('BATCH_SIZE')}")
        print(f"✓ Mixed precision: {os.environ.get('USE_MIXED_PRECISION')}")
    else:
        os.environ["CUDA_AVAILABLE"] = "false"
        print("⚠ CUDA not available, using CPU")
except:
    os.environ["CUDA_AVAILABLE"] = "false"

# Run API
import uvicorn
from motionmatch.api.main import app

print("🚀 Starting MotionMatch API")
print("🌐 Web interface: http://localhost:8000")
print("📚 API docs: http://localhost:8000/docs")
print("\nPress Ctrl+C to stop\n")

uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
