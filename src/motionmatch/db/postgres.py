"""PostgreSQL database operations for MotionMatch"""
import logging
import time
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from motionmatch.core.config import config

logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(config.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Video(Base):
    __tablename__ = "videos"
    
    video_id = Column(String(255), primary_key=True)
    video_url = Column(String(500), nullable=False)
    title = Column(String(500))
    duration = Column(Float)
    resolution = Column(String(20))
    fps = Column(Integer)
    file_size = Column(Integer)
    created_at = Column(DateTime, default=func.now())
    indexed_at = Column(DateTime)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text)
    temporal_features_path = Column(String(500))
    thumbnail_url = Column(String(500))
    processing_time_ms = Column(Float)
    video_metadata = Column(JSON)  # Flexible metadata storage

class IndexingJob(Base):
    __tablename__ = "indexing_jobs"
    
    job_id = Column(String(255), primary_key=True)
    total_videos = Column(Integer, nullable=False)
    completed = Column(Integer, default=0)
    failed = Column(Integer, default=0)
    status = Column(String(50), default="queued")  # queued, processing, completed, failed
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    job_metadata = Column(JSON)

class SearchQuery(Base):
    __tablename__ = "search_queries"
    
    query_id = Column(String(255), primary_key=True)
    user_id = Column(String(255))
    query_video_url = Column(String(500))
    filters = Column(JSON)
    num_results = Column(Integer)
    processing_time_ms = Column(Float)
    created_at = Column(DateTime, default=func.now())

class SearchClick(Base):
    __tablename__ = "search_clicks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(String(255), nullable=False)
    result_video_id = Column(String(255), nullable=False)
    rank = Column(Integer)
    similarity_score = Column(Float)
    clicked_at = Column(DateTime, default=func.now())

# Create tables
def init_database():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

# Database operations
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise

def update_indexing_status(video_id: str, status: str, task_id: str = None, error: str = None):
    """Update video indexing status"""
    db = get_db()
    try:
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            # Create new video record
            video = Video(
                video_id=video_id,
                video_url="",  # Will be updated later
                status=status
            )
            db.add(video)
        else:
            video.status = status
        
        if status == "processing":
            video.indexed_at = func.now()
        elif status == "completed":
            video.indexed_at = func.now()
        elif status == "failed":
            video.error_message = error
        
        db.commit()
        logger.info(f"Updated video {video_id} status to {status}")
        
    except Exception as e:
        logger.error(f"Failed to update video status: {e}")
        db.rollback()
    finally:
        db.close()

def update_video_metadata(video_id: str, metadata: Dict[str, Any]):
    """Update video metadata"""
    db = get_db()
    try:
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            video = Video(video_id=video_id)
            db.add(video)
        
        # Update fields
        for key, value in metadata.items():
            # Handle the renamed metadata column
            if key == "metadata":
                setattr(video, "video_metadata", value)
            elif hasattr(video, key):
                setattr(video, key, value)
        
        db.commit()
        logger.info(f"Updated metadata for video {video_id}")
        
    except Exception as e:
        logger.error(f"Failed to update video metadata: {e}")
        db.rollback()
    finally:
        db.close()

def create_indexing_job(job_id: str, total_videos: int, metadata: Dict[str, Any] = None):
    """Create new indexing job"""
    db = get_db()
    try:
        job = IndexingJob(
            job_id=job_id,
            total_videos=total_videos,
            status="queued",
            job_metadata=metadata or {}
        )
        db.add(job)
        db.commit()
        logger.info(f"Created indexing job {job_id} with {total_videos} videos")
        
    except Exception as e:
        logger.error(f"Failed to create indexing job: {e}")
        db.rollback()
    finally:
        db.close()

def update_job_progress(job_id: str, completed: int, failed: int, status: str = None, error: str = None):
    """Update indexing job progress"""
    db = get_db()
    try:
        job = db.query(IndexingJob).filter(IndexingJob.job_id == job_id).first()
        if job:
            job.completed = completed
            job.failed = failed
            
            if status:
                job.status = status
                if status == "processing" and not job.started_at:
                    job.started_at = func.now()
                elif status in ["completed", "completed_with_errors", "failed"]:
                    job.completed_at = func.now()
            
            if error:
                job.error_message = error
            
            db.commit()
            logger.info(f"Updated job {job_id} progress: {completed} completed, {failed} failed")
        
    except Exception as e:
        logger.error(f"Failed to update job progress: {e}")
        db.rollback()
    finally:
        db.close()

def get_indexing_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get indexing job status"""
    db = get_db()
    try:
        job = db.query(IndexingJob).filter(IndexingJob.job_id == job_id).first()
        if job:
            progress_pct = (job.completed + job.failed) / job.total_videos * 100 if job.total_videos > 0 else 0
            
            # Calculate ETA
            eta_seconds = None
            if job.started_at and job.completed > 0:
                elapsed = (func.now() - job.started_at).total_seconds()
                avg_time = elapsed / (job.completed + job.failed)
                remaining = job.total_videos - job.completed - job.failed
                eta_seconds = remaining * avg_time
            
            return {
                "job_id": job.job_id,
                "status": job.status,
                "total_videos": job.total_videos,
                "completed": job.completed,
                "failed": job.failed,
                "progress_percentage": progress_pct,
                "eta_seconds": eta_seconds,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error_message": job.error_message
            }
        return None
        
    except Exception as e:
        logger.error(f"Failed to get indexing job: {e}")
        return None
    finally:
        db.close()

def log_search_query(query_id: str, query_video_url: str, filters: Dict[str, Any], 
                    num_results: int, processing_time_ms: float, user_id: str = None):
    """Log search query for analytics"""
    db = get_db()
    try:
        query = SearchQuery(
            query_id=query_id,
            user_id=user_id,
            query_video_url=query_video_url,
            filters=filters,
            num_results=num_results,
            processing_time_ms=processing_time_ms
        )
        db.add(query)
        db.commit()
        
    except Exception as e:
        logger.error(f"Failed to log search query: {e}")
        db.rollback()
    finally:
        db.close()

def log_search_click(query_id: str, result_video_id: str, rank: int, similarity_score: float):
    """Log search result click for analytics"""
    db = get_db()
    try:
        click = SearchClick(
            query_id=query_id,
            result_video_id=result_video_id,
            rank=rank,
            similarity_score=similarity_score
        )
        db.add(click)
        db.commit()
        
    except Exception as e:
        logger.error(f"Failed to log search click: {e}")
        db.rollback()
    finally:
        db.close()

def get_video_metadata(video_id: str) -> Optional[Dict[str, Any]]:
    """Get video metadata"""
    db = get_db()
    try:
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if video:
            return {
                "video_id": video.video_id,
                "video_url": video.video_url,
                "title": video.title,
                "duration": video.duration,
                "resolution": video.resolution,
                "fps": video.fps,
                "file_size": video.file_size,
                "created_at": video.created_at.isoformat() if video.created_at else None,
                "indexed_at": video.indexed_at.isoformat() if video.indexed_at else None,
                "status": video.status,
                "temporal_features_path": video.temporal_features_path,
                "thumbnail_url": video.thumbnail_url,
                "processing_time_ms": video.processing_time_ms,
                "metadata": video.video_metadata
            }
        return None
        
    except Exception as e:
        logger.error(f"Failed to get video metadata: {e}")
        return None
    finally:
        db.close()

# Initialize database on import
try:
    init_database()
except Exception as e:
    logger.warning(f"Database initialization failed: {e}")
    # Don't crash the application