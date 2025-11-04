"""
Anomaly Detection Service using V-JEPA 2
Detects unusual behavior in video feeds by analyzing temporal patterns
"""
import logging
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

from motionmatch.services.encoder import encoder_service
from motionmatch.db.vector_db import vector_db

logger = logging.getLogger(__name__)


class AnomalyDetectionService:
    """Detect anomalies in video feeds using V-JEPA 2 temporal features"""
    
    def __init__(self):
        self.baseline_features = {}
    
    def establish_baseline(self, normal_videos: List[str]) -> Dict:
        """
        Establish baseline of normal behavior from training videos
        
        Args:
            normal_videos: List of video paths showing normal behavior
            
        Returns:
            Baseline statistics
        """
        logger.info(f"Establishing baseline from {len(normal_videos)} normal videos")
        
        all_temporal_features = []
        all_temporal_variances = []
        all_motion_magnitudes = []
        
        for video_path in normal_videos:
            try:
                features = encoder_service.encode_video(video_path)
                temporal = features.temporal_features  # [T, D]
                
                # Calculate temporal statistics
                temporal_variance = np.var(temporal, axis=0)  # Variance per dimension
                all_temporal_variances.append(temporal_variance)
                
                # Calculate motion magnitude (frame-to-frame changes)
                temporal_diffs = np.diff(temporal, axis=0)
                motion_magnitude = np.linalg.norm(temporal_diffs, axis=1).mean()
                all_motion_magnitudes.append(motion_magnitude)
                
                all_temporal_features.append(temporal)
                
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
        
        if not all_temporal_features:
            raise ValueError("No valid videos processed for baseline")
        
        # Calculate baseline statistics
        self.baseline_features = {
            "mean_temporal_variance": np.mean(all_temporal_variances, axis=0),
            "std_temporal_variance": np.std(all_temporal_variances, axis=0),
            "mean_motion_magnitude": np.mean(all_motion_magnitudes),
            "std_motion_magnitude": np.std(all_motion_magnitudes),
            "num_videos": len(all_temporal_features)
        }
        
        logger.info(f"Baseline established:")
        logger.info(f"  Mean motion magnitude: {self.baseline_features['mean_motion_magnitude']:.6f}")
        logger.info(f"  Std motion magnitude: {self.baseline_features['std_motion_magnitude']:.6f}")
        
        return self.baseline_features
    
    def detect_anomaly(self, video_path: str, threshold: float = 2.0) -> Dict:
        """
        Detect if a video contains anomalous behavior
        
        Args:
            video_path: Path to video to analyze
            threshold: Number of standard deviations for anomaly (default: 2.0)
            
        Returns:
            Anomaly detection results
        """
        if not self.baseline_features:
            raise ValueError("Baseline not established. Call establish_baseline() first.")
        
        # Encode video
        features = encoder_service.encode_video(video_path)
        temporal = features.temporal_features  # [T, D]
        
        # Calculate temporal variance
        temporal_variance = np.var(temporal, axis=0)
        
        # Calculate motion magnitude
        temporal_diffs = np.diff(temporal, axis=0)
        motion_magnitude = np.linalg.norm(temporal_diffs, axis=1).mean()
        
        # Calculate anomaly scores
        # Score 1: Motion magnitude deviation
        motion_z_score = (
            (motion_magnitude - self.baseline_features['mean_motion_magnitude']) /
            (self.baseline_features['std_motion_magnitude'] + 1e-8)
        )
        
        # Score 2: Temporal variance deviation
        variance_diff = np.abs(
            temporal_variance - self.baseline_features['mean_temporal_variance']
        )
        variance_z_score = np.mean(
            variance_diff / (self.baseline_features['std_temporal_variance'] + 1e-8)
        )
        
        # Combined anomaly score
        anomaly_score = (abs(motion_z_score) + variance_z_score) / 2
        
        # Determine if anomalous
        is_anomaly = anomaly_score > threshold
        
        result = {
            "video_path": video_path,
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score),
            "motion_z_score": float(motion_z_score),
            "variance_z_score": float(variance_z_score),
            "motion_magnitude": float(motion_magnitude),
            "threshold": threshold,
            "confidence": min(100, float(anomaly_score / threshold * 100))
        }
        
        logger.info(f"Anomaly detection for {Path(video_path).name}:")
        logger.info(f"  Anomaly score: {anomaly_score:.2f} (threshold: {threshold})")
        logger.info(f"  Is anomaly: {is_anomaly}")
        
        return result
    
    def detect_temporal_anomalies(self, video_path: str, window_size: int = 16) -> List[Dict]:
        """
        Detect anomalies at specific moments within a video
        
        Args:
            video_path: Path to video
            window_size: Number of frames per window
            
        Returns:
            List of anomalous moments with timestamps
        """
        if not self.baseline_features:
            raise ValueError("Baseline not established. Call establish_baseline() first.")
        
        features = encoder_service.encode_video(video_path)
        temporal = features.temporal_features  # [T, D]
        
        anomalies = []
        num_windows = len(temporal) - window_size + 1
        
        for i in range(num_windows):
            window = temporal[i:i+window_size]
            
            # Calculate motion in this window
            window_diffs = np.diff(window, axis=0)
            window_motion = np.linalg.norm(window_diffs, axis=1).mean()
            
            # Calculate z-score
            motion_z_score = (
                (window_motion - self.baseline_features['mean_motion_magnitude']) /
                (self.baseline_features['std_motion_magnitude'] + 1e-8)
            )
            
            # If significantly different from baseline
            if abs(motion_z_score) > 2.0:
                anomalies.append({
                    "frame_start": i,
                    "frame_end": i + window_size,
                    "timestamp_start": i / len(temporal),  # Normalized timestamp
                    "timestamp_end": (i + window_size) / len(temporal),
                    "motion_z_score": float(motion_z_score),
                    "motion_magnitude": float(window_motion)
                })
        
        logger.info(f"Found {len(anomalies)} anomalous moments in {Path(video_path).name}")
        
        return anomalies
    
    def compare_to_normal(self, video_path: str, top_k: int = 5) -> List[Dict]:
        """
        Compare video to indexed normal videos to find most/least similar
        
        Args:
            video_path: Path to video to analyze
            top_k: Number of similar videos to return
            
        Returns:
            List of similar normal videos with similarity scores
        """
        # Encode query video
        features = encoder_service.encode_video(video_path)
        
        # Search for similar videos
        results = vector_db.search_similar(
            query_embedding=features.global_features,
            top_k=top_k
        )
        
        # Add interpretation
        for result in results:
            if result.similarity_score > 0.95:
                result.metadata['interpretation'] = "Very similar to normal behavior"
            elif result.similarity_score > 0.90:
                result.metadata['interpretatioan'] = "Somewhat similar to normal behavior"
            else:
                result.metadata['interpretation'] = "Different from normal behavior"
        
        return results


# Global anomaly detection service
anomaly_service = AnomalyDetectionService()
