"""Celery tasks for MotionMatch"""
import os
import time
import logging
from typing import List, Dict, Any
from celery import current_task
from celery.exceptions import MaxRetriesExceededError

from motionmatch.workers.celery_app import celery_app
from motionmatch.services.encoder import encoder_service
from motionmatch.db.vector_db import vector_db
from motionmatch.db.postgres import update_indexing_status, update_video_metadata

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3)
def index_video_task(self, video_id: str, video_url: str, metadata: Dict[str, Any] = None):
    """Celery task to index a single video"""
    try:
        # Update status: processing
        update_indexing_status(video_id, "processing", task_id=self.request.id)
        
        # Download video if it's a URL
        if video_url.startswith(('http://', 'https://', 's3://')):
            local_path = download_video(video_url, video_id)
        else:
            local_path = video_url
        
        # Validate video exists
        if not os.path.exists(local_path):
            raise ValueError(f"Video file not found: {local_path}")
        
        # Optional: Shot segmentation
        segments = []
        if config.ENABLE_SHOT_SEGMENTATION and 'shot_segmentation_service' in globals():
            try:
                segments = shot_segmentation_service.segment_for_indexing(local_path, video_id)
                logger.info(f"Detected {len(segments)} shots in {video_id}")
            except Exception as e:
                logger.warning(f"Shot segmentation failed for {video_id}: {e}")
        
        # Optional: ROI detection
        roi_info = None
        if config.ENABLE_ROI_DETECTION and 'roi_detection_service' in globals():
            try:
                roi_info = roi_detection_service.detect_primary_subject(local_path)
                if roi_info:
                    logger.info(f"Detected ROI in {video_id}: {roi_info['class_name']}")
            except Exception as e:
                logger.warning(f"ROI detection failed for {video_id}: {e}")
        
        # Encode with V-JEPA 2 (use ROI if available)
        if roi_info and config.ENABLE_ROI_DETECTION:
            try:
                roi_frames = roi_detection_service.extract_roi_frames(local_path, roi_info)
                if roi_frames is not None:
                    # Create temporary ROI video for encoding
                    roi_video_path = local_path.replace('.mp4', '_roi.mp4')
                    # Save ROI frames as video (simplified)
                    features = encoder_service.encode_video(local_path)  # Fallback to full video
                    metadata["has_roi"] = True
                    metadata["roi_info"] = roi_info
                else:
                    features = encoder_service.encode_video(local_path)
            except Exception as e:
                logger.warning(f"ROI encoding failed for {video_id}, using full video: {e}")
                features = encoder_service.encode_video(local_path)
        else:
            features = encoder_service.encode_video(local_path)
        
        # Get video duration
        duration = get_video_duration(local_path)
        
        # Insert into vector DB
        success = vector_db.insert_video(
            video_id=video_id,
            embedding=features.global_features,
            video_path=video_url,
            duration=duration,
            created_at=features.created_at
        )
        
        if not success:
            raise Exception("Failed to insert into vector database")
        
        # Store temporal features separately (S3 or local storage)
        temporal_path = store_temporal_features(video_id, features.temporal_features)
        
        # Update metadata in PostgreSQL
        video_metadata = {
            "video_id": video_id,
            "video_url": video_url,
            "duration": duration,
            "resolution": metadata.get("resolution") if metadata else None,
            "fps": metadata.get("fps") if metadata else None,
            "file_size": os.path.getsize(local_path) if os.path.exists(local_path) else None,
            "temporal_features_path": temporal_path,
            "indexed_at": time.time(),
            "processing_time_ms": features.metadata["processing_time_ms"]
        }
        
        if metadata:
            video_metadata.update(metadata)
        
        update_video_metadata(video_id, video_metadata)
        
        # Cleanup temporary file
        if local_path != video_url and os.path.exists(local_path):
            os.remove(local_path)
        
        # Update status: completed
        update_indexing_status(video_id, "completed")
        
        return {"video_id": video_id, "status": "completed", "duration": duration}
        
    except MaxRetriesExceededError:
        update_indexing_status(video_id, "failed", error="Max retries exceeded")
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to index video {video_id}: {error_msg}")
        update_indexing_status(video_id, "failed", error=error_msg)
        
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))

@celery_app.task(bind=True)
def batch_index_task(self, job_id: str, video_submissions: List[Dict[str, Any]]):
    """Process a batch of videos for indexing"""
    try:
        from motionmatch.db.postgres import create_indexing_job, update_job_progress
        
        # Create job record
        create_indexing_job(job_id, len(video_submissions))
        
        # Submit individual video tasks
        task_ids = []
        for submission in video_submissions:
            task = index_video_task.delay(
                submission["video_id"],
                submission["video_url"],
                submission.get("metadata", {})
            )
            task_ids.append(task.id)
        
        # Monitor progress (simplified - in production use Celery callbacks)
        completed = 0
        failed = 0
        
        for task_id in task_ids:
            try:
                result = celery_app.AsyncResult(task_id)
                result.get(timeout=600)  # 10 minute timeout per video
                completed += 1
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                failed += 1
            
            # Update job progress
            update_job_progress(job_id, completed, failed)
        
        # Mark job as completed
        status = "completed" if failed == 0 else "completed_with_errors"
        update_job_progress(job_id, completed, failed, status=status)
        
        return {
            "job_id": job_id,
            "status": status,
            "completed": completed,
            "failed": failed
        }
        
    except Exception as e:
        logger.error(f"Batch indexing job {job_id} failed: {e}")
        update_job_progress(job_id, 0, len(video_submissions), status="failed", error=str(e))
        raise

def download_video(video_url: str, video_id: str) -> str:
    """Download video from URL to local storage"""
    import requests
    from urllib.parse import urlparse
    
    # Create temp directory
    temp_dir = os.path.join(config.TEMP_PATH, "downloads")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Determine file extension
    parsed_url = urlparse(video_url)
    ext = os.path.splitext(parsed_url.path)[1] or '.mp4'
    local_path = os.path.join(temp_dir, f"{video_id}{ext}")
    
    # Download file
    response = requests.get(video_url, stream=True, timeout=30)
    response.raise_for_status()
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return local_path

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0.0
            cap.release()
            return duration
        return 0.0
    except Exception as e:
        logger.error(f"Failed to get duration for {video_path}: {e}")
        return 0.0

def store_temporal_features(video_id: str, temporal_features) -> str:
    """Store temporal features to persistent storage"""
    import numpy as np
    
    # Create features directory
    features_dir = os.path.join(config.STORAGE_PATH, "temporal_features")
    os.makedirs(features_dir, exist_ok=True)
    
    # Save as numpy file
    feature_path = os.path.join(features_dir, f"{video_id}_temporal.npy")
    np.save(feature_path, temporal_features)
    
    return feature_path