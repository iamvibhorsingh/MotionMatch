"""Indexing service for MotionMatch MVP"""
import os
import time
import uuid
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from motionmatch.services.encoder import encoder_service
from motionmatch.db.vector_db import vector_db
from motionmatch.db.models import IndexRequest, IndexStatus
from motionmatch.core.config import config
from motionmatch.workers.tasks import batch_index_task
from motionmatch.db.postgres import create_indexing_job, get_indexing_job
from motionmatch.services.preprocessing.shot_segmentation import shot_segmentation_service
from motionmatch.services.preprocessing.roi_detection import roi_detection_service

logger = logging.getLogger(__name__)

class IndexingService:
    """Service for indexing video libraries"""
    
    def __init__(self):
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def submit_indexing_job(self, request: IndexRequest) -> str:
        """Submit videos for indexing using Celery"""
        job_id = str(uuid.uuid4())
        
        # Process video submissions
        video_submissions = []
        for video_submission in request.videos:
            submission_data = {
                "video_id": video_submission.video_id,
                "video_url": video_submission.video_url,
                "video_metadata": video_submission.metadata.dict() if video_submission.metadata else {}
            }
            
            # Add indexing options
            if request.options:
                submission_data["options"] = request.options.dict()
            
            video_submissions.append(submission_data)
        
        # Submit to Celery
        batch_index_task.delay(job_id, video_submissions)
        
        logger.info(f"Submitted Celery indexing job {job_id} with {len(video_submissions)} videos")
        return job_id
    
    def get_job_status(self, job_id: str) -> IndexStatus:
        """Get indexing job status from database"""
        job_data = get_indexing_job(job_id)
        if job_data:
            return IndexStatus(**job_data)
        return None
    
    def _process_videos(self, job_id: str, video_paths: List[str]):
        """Process videos in batch"""
        job = self.jobs[job_id]
        job.status = "processing"
        
        start_time = time.time()
        
        for i, video_path in enumerate(video_paths):
            try:
                # Check if video file exists
                if not os.path.exists(video_path):
                    logger.error(f"Video file not found: {video_path}")
                    job.failed += 1
                    continue
                
                # Encode video
                logger.info(f"Indexing video {i+1}/{len(video_paths)}: {video_path}")
                features = encoder_service.encode_video(video_path)
                
                # Get video duration (simplified)
                duration = self._get_video_duration(video_path)
                
                # Insert into vector database
                success = vector_db.insert_video(
                    video_id=features.video_id,
                    embedding=features.global_features,
                    video_path=video_path,
                    duration=duration,
                    created_at=features.created_at
                )
                
                if success:
                    job.completed += 1
                    logger.info(f"Successfully indexed {features.video_id}")
                else:
                    job.failed += 1
                    logger.error(f"Failed to insert {features.video_id} into database")
                
            except Exception as e:
                logger.error(f"Failed to index video {video_path}: {e}")
                job.failed += 1
            
            # Update progress
            job.progress_percentage = ((job.completed + job.failed) / job.total_videos) * 100
            
            # Estimate ETA
            if job.completed > 0:
                elapsed_time = time.time() - start_time
                avg_time_per_video = elapsed_time / (job.completed + job.failed)
                remaining_videos = job.total_videos - job.completed - job.failed
                job.eta_seconds = remaining_videos * avg_time_per_video
        
        # Mark job as completed
        if job.failed == 0:
            job.status = "completed"
        else:
            job.status = "completed_with_errors"
        
        job.eta_seconds = None
        
        logger.info(f"Indexing job {job_id} finished: {job.completed} successful, {job.failed} failed")
    
    def _get_video_duration(self, video_path: str) -> float:
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
    
    def index_single_video(self, video_path: str, skip_if_exists: bool = True) -> bool:
        """Index a single video (synchronous)
        
        Args:
            video_path: Path to video file
            skip_if_exists: If True, skip indexing if video already exists
            
        Returns:
            True if indexed successfully or already exists, False otherwise
        """
        try:
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return False
            
            # Generate video ID - match the format used by encoder (absolute path without extension)
            video_id = os.path.abspath(video_path).rsplit('.', 1)[0]
            
            # Check if already indexed
            if skip_if_exists:
                exists = vector_db.video_exists(video_id)
                logger.info(f"Checking if {video_id} exists: {exists}")
                if exists:
                    logger.info(f"Video already indexed, skipping")
                    return True
            
            # Encode video
            features = encoder_service.encode_video(video_path)
            
            # Get video duration
            duration = self._get_video_duration(video_path)
            
            # Insert into vector database
            success = vector_db.insert_video(
                video_id=features.video_id,
                embedding=features.global_features,
                video_path=video_path,
                duration=duration,
                created_at=features.created_at
            )
            
            if success:
                logger.info(f"Successfully indexed {features.video_id}")
            else:
                logger.error(f"Failed to insert {features.video_id} into database")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to index video {video_path}: {e}")
            return False

# Global indexing service instance
indexing_service = IndexingService()