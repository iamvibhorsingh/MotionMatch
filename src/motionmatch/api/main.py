"""FastAPI application for MotionMatch MVP"""
import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from motionmatch.db.models import (
    SearchRequest, SearchResponse, IndexRequest, IndexStatus,
    HealthResponse, VideoSubmission, IndexingOptions, SearchOptions
)
from motionmatch.db.postgres import log_search_query, get_video_metadata
from motionmatch.services.search import search_service
from motionmatch.services.indexing import indexing_service
from motionmatch.services.encoder import encoder_service
from motionmatch.services.anomaly_detection import anomaly_service
from motionmatch.db.vector_db import vector_db
from motionmatch.core.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MotionMatch API",
    description="Physics-based video search engine using V-JEPA 2",
    version="1.0.0"
)

# Create v1 router for API versioning
from fastapi import APIRouter
v1_router = APIRouter(prefix="/v1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure storage directories exist
os.makedirs(config.STORAGE_PATH, exist_ok=True)
os.makedirs(config.TEMP_PATH, exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root-level endpoints (for compatibility with test scripts)
@app.get("/health", response_model=HealthResponse)
async def root_health_check():
    """Root-level health check endpoint"""
    return await health_check()

@app.get("/stats")
async def root_get_stats():
    """Root-level stats endpoint"""
    return await get_stats()

@app.post("/index/single")
async def root_index_single_video(video_path: str, skip_if_exists: bool = True):
    """Root-level single video indexing endpoint"""
    return await index_single_video(video_path, skip_if_exists)

@app.post("/search")
async def root_search_videos(request: SearchRequest):
    """Root-level search endpoint"""
    return await search_videos(request)

@app.post("/search/upload")
async def root_search_upload(file: UploadFile = File(...), top_k: int = Form(20)):
    """Root-level search upload endpoint"""
    return await search_with_upload(file, top_k)

@v1_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check encoder service
        encoder_health = encoder_service.health_check()
        
        # Check vector database
        db_health = vector_db.health_check()
        
        # Overall health
        healthy = (
            encoder_health.get("status") == "healthy" and
            db_health.get("status") == "healthy"
        )
        
        return HealthResponse(
            status="healthy" if healthy else "unhealthy",
            model_loaded=encoder_health.get("model_loaded", False),
            device=encoder_health.get("device", "unknown"),
            gpu_memory_mb=encoder_health.get("gpu_memory_mb")
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="unknown"
        )

@v1_router.post("/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest):
    """Search for similar videos"""
    try:
        # Validate query video path
        if not os.path.exists(request.query_video_url):
            raise HTTPException(
                status_code=400,
                detail=f"Query video not found: {request.query_video_url}"
            )
        
        # Process search
        response = search_service.search(request)
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.post("/search/upload")
async def search_with_upload(
    file: UploadFile = File(...),
    top_k: int = Form(20),
    enable_reranking: str = Form("false")
):
    """Search with uploaded query video"""
    try:
        # Save uploaded file
        temp_path = os.path.join(config.TEMP_PATH, f"query_{file.filename}")
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Convert to boolean - handle both string and Form object
        if isinstance(enable_reranking, str):
            reranking_enabled = enable_reranking.lower() in ('true', '1', 'yes')
        else:
            # It's a Form object or other type, convert to string first
            reranking_enabled = str(enable_reranking).lower() in ('true', '1', 'yes')
        
        request = SearchRequest(
            query_video_url=temp_path,
            top_k=top_k,
            options=SearchOptions(enable_reranking=reranking_enabled)
        )
        
        # Process search
        response = search_service.search(request)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return response
        
    except Exception as e:
        logger.error(f"Upload search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.post("/index/submit")
async def submit_indexing_job(request: IndexRequest):
    """Submit videos for indexing"""
    try:
        # Validate video paths
        invalid_paths = [path for path in request.video_paths if not os.path.exists(path)]
        if invalid_paths:
            raise HTTPException(
                status_code=400,
                detail=f"Video files not found: {invalid_paths}"
            )
        
        # Submit indexing job
        job_id = indexing_service.submit_indexing_job(request)
        
        return {"job_id": job_id, "status": "submitted"}
        
    except Exception as e:
        logger.error(f"Indexing submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.post("/index/single")
async def index_single_video(video_path: str, skip_if_exists: bool = True):
    """Index a single video
    
    Args:
        video_path: Path to video file
        skip_if_exists: Skip indexing if video already exists (default: True)
    """
    try:
        if not os.path.exists(video_path):
            raise HTTPException(
                status_code=400,
                detail=f"Video file not found: {video_path}"
            )
        
        success = indexing_service.index_single_video(video_path, skip_if_exists=skip_if_exists)
        
        if success:
            return {"status": "success", "message": f"Video indexed successfully: {video_path}"}
        else:
            raise HTTPException(status_code=500, detail="Indexing failed")
        
    except Exception as e:
        logger.error(f"Single video indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.get("/index/status/{job_id}", response_model=IndexStatus)
async def get_indexing_status(job_id: str):
    """Get indexing job status"""
    try:
        status = indexing_service.get_job_status(job_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        video_count = vector_db.get_video_count()
        
        return {
            "total_videos": video_count,
            "model_name": config.MODEL_NAME,
            "device": config.DEVICE,
            "vector_dim": config.VECTOR_DIM
        }
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.get("/videos/list")
async def list_videos(limit: int = 10):
    """List indexed video IDs (for debugging)"""
    try:
        all_ids = vector_db.get_all_video_ids()
        return {
            "total": len(all_ids),
            "sample": all_ids[:limit]
        }
    except Exception as e:
        logger.error(f"List videos failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Anomaly Detection Endpoints
@v1_router.post("/anomaly/baseline")
async def establish_anomaly_baseline(video_paths: List[str]):
    """Establish baseline from normal behavior videos"""
    try:
        baseline = anomaly_service.establish_baseline(video_paths)
        return {
            "status": "success",
            "baseline": {
                "num_videos": baseline["num_videos"],
                "mean_motion_magnitude": float(baseline["mean_motion_magnitude"]),
                "std_motion_magnitude": float(baseline["std_motion_magnitude"])
            }
        }
    except Exception as e:
        logger.error(f"Baseline establishment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.post("/anomaly/detect")
async def detect_anomaly(video_path: str, threshold: float = 2.0):
    """Detect if video contains anomalous behavior"""
    try:
        if not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail=f"Video not found: {video_path}")
        
        result = anomaly_service.detect_anomaly(video_path, threshold)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.post("/anomaly/detect/upload")
async def detect_anomaly_upload(file: UploadFile = File(...), threshold: float = Form(2.0)):
    """Detect anomalies in uploaded video"""
    try:
        # Save uploaded file
        temp_path = os.path.join(config.TEMP_PATH, f"anomaly_{file.filename}")
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Detect anomaly
        result = anomaly_service.detect_anomaly(temp_path, threshold)
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Anomaly detection upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.post("/anomaly/moments")
async def detect_temporal_anomalies(video_path: str, window_size: int = 16):
    """Detect anomalous moments within a video"""
    try:
        if not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail=f"Video not found: {video_path}")
        
        anomalies = anomaly_service.detect_temporal_anomalies(video_path, window_size)
        return {
            "video_path": video_path,
            "num_anomalies": len(anomalies),
            "anomalies": anomalies
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Temporal anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video from the index"""
    try:
        success = vector_db.delete_video(video_id)
        
        if success:
            return {"status": "success", "message": f"Video deleted: {video_id}"}
        else:
            raise HTTPException(status_code=500, detail="Deletion failed")
        
    except Exception as e:
        logger.error(f"Video deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.delete("/videos")
async def clear_all_videos():
    """Clear all videos from the index"""
    try:
        success = vector_db.clear_all()
        
        if success:
            return {"status": "success", "message": "All videos cleared from index"}
        else:
            raise HTTPException(status_code=500, detail="Clear operation failed")
        
    except Exception as e:
        logger.error(f"Clear all failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include v1 router
app.include_router(v1_router)

@app.get("/")
async def root():
    """Serve the main web interface"""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(
        "motionmatch.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )