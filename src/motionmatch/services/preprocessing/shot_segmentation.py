"""Shot segmentation for MotionMatch using PySceneDetect"""
import logging
from typing import List, Tuple
import os
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector, ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg

logger = logging.getLogger(__name__)

class ShotSegmentationService:
    """Service for detecting and splitting video shots"""
    
    def __init__(self, threshold: float = 30.0, min_scene_len: float = 1.0):
        """
        Initialize shot segmentation service
        
        Args:
            threshold: Content detection threshold (higher = less sensitive)
            min_scene_len: Minimum scene length in seconds
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len
    
    def detect_shots(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Detect shot boundaries in video
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        try:
            # Create video manager
            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            
            # Add content detector
            scene_manager.add_detector(
                ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len)
            )
            
            # Start video manager
            video_manager.set_duration()
            video_manager.start()
            
            # Detect scenes
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()
            
            # Convert to time tuples
            shots = []
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                shots.append((start_time, end_time))
            
            logger.info(f"Detected {len(shots)} shots in {video_path}")
            return shots
            
        except Exception as e:
            logger.error(f"Shot detection failed for {video_path}: {e}")
            # Return single shot covering entire video
            duration = self._get_video_duration(video_path)
            return [(0.0, duration)]
    
    def split_video_by_shots(self, video_path: str, output_dir: str, 
                           shots: List[Tuple[float, float]] = None) -> List[str]:
        """
        Split video into shot segments
        
        Args:
            video_path: Path to input video
            output_dir: Directory for output segments
            shots: Pre-detected shots, or None to detect automatically
            
        Returns:
            List of output file paths
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Detect shots if not provided
            if shots is None:
                shots = self.detect_shots(video_path)
            
            # Generate output paths
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_paths = []
            
            for i, (start_time, end_time) in enumerate(shots):
                output_path = os.path.join(output_dir, f"{base_name}_shot_{i:03d}.mp4")
                output_paths.append(output_path)
            
            # Split video using ffmpeg
            video_manager = VideoManager([video_path])
            scene_list = []
            
            # Convert time tuples back to FrameTimecode objects
            for start_time, end_time in shots:
                start_frame = video_manager.get_base_timecode() + start_time
                end_frame = video_manager.get_base_timecode() + end_time
                scene_list.append((start_frame, end_frame))
            
            # Perform the split
            split_video_ffmpeg(
                input_video_paths=[video_path],
                scene_list=scene_list,
                output_file_template=os.path.join(output_dir, f"{base_name}_shot_$SCENE_NUMBER.mp4"),
                video_name=base_name,
                arg_override='-c:v libx264 -crf 23 -c:a aac'  # Good quality/size balance
            )
            
            logger.info(f"Split video into {len(output_paths)} shots")
            return output_paths
            
        except Exception as e:
            logger.error(f"Video splitting failed for {video_path}: {e}")
            return []
    
    def segment_for_indexing(self, video_path: str, video_id: str) -> List[dict]:
        """
        Segment video for indexing with metadata
        
        Args:
            video_path: Path to video file
            video_id: Base video ID
            
        Returns:
            List of segment metadata dicts
        """
        try:
            shots = self.detect_shots(video_path)
            
            segments = []
            for i, (start_time, end_time) in enumerate(shots):
                duration = end_time - start_time
                
                # Skip very short segments
                if duration < self.min_scene_len:
                    continue
                
                segment_id = f"{video_id}_shot_{i:03d}"
                
                segment_metadata = {
                    "video_id": segment_id,
                    "parent_video_id": video_id,
                    "shot_index": i,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "is_shot_segment": True
                }
                
                segments.append(segment_metadata)
            
            logger.info(f"Created {len(segments)} segments for {video_id}")
            return segments
            
        except Exception as e:
            logger.error(f"Segmentation failed for {video_id}: {e}")
            return []
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0.0
                cap.release()
                return duration
            return 0.0
        except Exception:
            return 0.0

# Global instance with error handling
try:
    shot_segmentation_service = ShotSegmentationService()
except Exception as e:
    logger.warning(f"Shot segmentation service unavailable: {e}")
    shot_segmentation_service = None