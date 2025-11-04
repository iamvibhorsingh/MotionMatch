"""Configuration settings for MotionMatch MVP"""
import os
from typing import Optional

class Config:
    # V-JEPA 2 Model
    MODEL_NAME = os.getenv("MODEL_NAME", "facebook/vjepa2-vitl-fpc64-256")
    
    # Auto-detect CUDA if not explicitly set
    @staticmethod
    def _detect_device():
        cuda_env = os.getenv("CUDA_AVAILABLE", "").lower()
        if cuda_env == "true":
            return "cuda"
        elif cuda_env == "false":
            return "cpu"
        else:
            # Auto-detect
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except:
                return "cpu"
    
    DEVICE = _detect_device.__func__()
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))  # Increased for RTX 4070 Ti
    NUM_FRAMES = int(os.getenv("NUM_FRAMES", "64"))
    FRAME_SIZE = int(os.getenv("FRAME_SIZE", "256"))
    
    # GPU Optimization
    USE_MIXED_PRECISION = os.getenv("USE_MIXED_PRECISION", "true").lower() == "true"
    TORCH_COMPILE = os.getenv("TORCH_COMPILE", "false").lower() == "true"  # PyTorch 2.0+ optimization
    
    # Vector Database (Milvus)
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    COLLECTION_NAME = "motion_vectors"
    VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))  # Make configurable
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # PostgreSQL
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://motionmatch:password@127.0.0.1:5432/motionmatch")
    
    # Storage
    STORAGE_PATH = os.getenv("STORAGE_PATH", "./storage")
    TEMP_PATH = os.getenv("TEMP_PATH", "./temp")
    
    # API
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Performance
    MAX_QUERY_LATENCY = 5.0  # seconds
    SEARCH_TOP_K = 50  # Retrieve 50 candidates for re-ranking
    RESULT_TOP_K = 20  # Return top 20 to user
    
    # Query Cache Settings
    ENABLE_QUERY_CACHE = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"
    QUERY_CACHE_SIZE_MB = int(os.getenv("QUERY_CACHE_SIZE_MB", "500"))  # Max disk cache size
    
    # Celery
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    
    # Feature Flags
    ENABLE_SHOT_SEGMENTATION = os.getenv("ENABLE_SHOT_SEGMENTATION", "false").lower() == "true"
    ENABLE_ROI_DETECTION = os.getenv("ENABLE_ROI_DETECTION", "false").lower() == "true"
    ENABLE_CELERY = os.getenv("ENABLE_CELERY", "true").lower() == "true"
    
    # Shot Segmentation
    SHOT_DETECTION_THRESHOLD = float(os.getenv("SHOT_DETECTION_THRESHOLD", "30.0"))
    MIN_SCENE_LENGTH = float(os.getenv("MIN_SCENE_LENGTH", "1.0"))
    
    # ROI Detection
    ROI_MODEL_NAME = os.getenv("ROI_MODEL_NAME", "yolov8n.pt")
    ROI_CONFIDENCE_THRESHOLD = float(os.getenv("ROI_CONFIDENCE_THRESHOLD", "0.5"))

config = Config()