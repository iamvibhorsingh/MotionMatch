"""Data models for MotionMatch MVP"""
from dataclasses import dataclass
# from datetime import datetime  # Not needed anymore
from typing import List, Optional, Dict, Any
import numpy as np
from pydantic import BaseModel

@dataclass
class VideoFeatures:
    """Video feature representation"""
    video_id: str
    global_features: np.ndarray  # Shape: [1024]
    temporal_features: np.ndarray  # Shape: [T, 1024]
    metadata: Dict[str, Any]
    created_at: float

class SearchFilters(BaseModel):
    """Search filters"""
    duration_min: Optional[float] = None
    duration_max: Optional[float] = None
    tags: Optional[List[str]] = None

class SearchOptions(BaseModel):
    """Search options"""
    enable_reranking: bool = False
    include_temporal_analysis: bool = False

class SearchRequest(BaseModel):
    """Search request model matching spec"""
    query_video_url: str
    top_k: int = 20
    filters: Optional[SearchFilters] = None
    options: Optional[SearchOptions] = None

class SearchResult(BaseModel):
    """Search result model"""
    video_id: str
    similarity_score: float
    distance: float
    video_path: str
    thumbnail_path: Optional[str] = None
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    """Search response model"""
    query_id: str
    processing_time_ms: float
    results: List[SearchResult]
    total_results: int

class VideoMetadata(BaseModel):
    """Video metadata model"""
    title: Optional[str] = None
    tags: Optional[List[str]] = None
    duration: Optional[float] = None

class VideoSubmission(BaseModel):
    """Single video submission"""
    video_id: str
    video_url: str
    metadata: Optional[VideoMetadata] = None

class IndexingOptions(BaseModel):
    """Indexing options"""
    segment_shots: bool = False
    detect_roi: bool = False
    priority: str = "normal"

class IndexRequest(BaseModel):
    """Index request model matching spec"""
    videos: List[VideoSubmission]
    options: Optional[IndexingOptions] = None

class IndexStatus(BaseModel):
    """Index status model"""
    job_id: str
    status: str  # 'queued', 'processing', 'completed', 'failed'
    total_videos: int
    completed: int
    failed: int
    progress_percentage: float
    eta_seconds: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    gpu_memory_mb: Optional[float] = None