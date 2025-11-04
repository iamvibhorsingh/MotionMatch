"""ROI (Region of Interest) detection using YOLOv8 and ByteTrack"""
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv

logger = logging.getLogger(__name__)

class ROIDetectionService:
    """Service for detecting and tracking regions of interest in videos"""
    
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize ROI detection service
        
        Args:
            model_name: YOLOv8 model name (yolov8n.pt, yolov8s.pt, etc.)
            confidence_threshold: Detection confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.tracker = None
        self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """Load YOLOv8 model and ByteTrack tracker"""
        try:
            # Load YOLO model (will auto-download if not present)
            logger.info(f"Loading YOLOv8 model: {model_name} (will download if needed)")
            self.model = YOLO(model_name)
            logger.info(f"✓ Loaded YOLOv8 model: {model_name}")
            
            # Initialize ByteTrack tracker
            self.tracker = sv.ByteTrack()
            logger.info("✓ Initialized ByteTrack tracker")
            
        except Exception as e:
            logger.error(f"Failed to load ROI detection model: {e}")
            logger.info("ROI detection will be disabled")
            self.model = None
            self.tracker = None
    
    def detect_primary_subject(self, video_path: str, target_classes: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Detect primary subject in video and return bounding box info
        
        Args:
            video_path: Path to video file
            target_classes: List of target class names (e.g., ['person', 'car'])
            
        Returns:
            Dictionary with ROI information or None if no subject detected
        """
        if self.model is None:
            logger.warning("ROI detection model not available, skipping detection")
            return None
            
        if target_classes is None:
            target_classes = ['person']  # Default to person detection
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample frames for detection (every 30 frames or 1 second)
            sample_interval = max(1, int(fps))
            frame_indices = range(0, total_frames, sample_interval)
            
            all_detections = []
            frame_count = 0
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Run detection
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]
                            
                            if class_name in target_classes:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                
                                detection = {
                                    'frame_idx': frame_idx,
                                    'class_name': class_name,
                                    'confidence': float(confidence),
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'area': (x2 - x1) * (y2 - y1)
                                }
                                all_detections.append(detection)
                
                frame_count += 1
                if frame_count >= 10:  # Limit sampling for efficiency
                    break
            
            cap.release()
            
            if not all_detections:
                logger.info(f"No subjects detected in {video_path}")
                return None
            
            # Find most consistent detection (highest average confidence + area)
            roi_info = self._analyze_detections(all_detections, video_path)
            
            logger.info(f"Detected primary subject in {video_path}: {roi_info['class_name']}")
            return roi_info
            
        except Exception as e:
            logger.error(f"ROI detection failed for {video_path}: {e}")
            return None
    
    def _analyze_detections(self, detections: List[Dict], video_path: str) -> Dict[str, Any]:
        """Analyze detections to find primary subject ROI"""
        if not detections:
            return None
        
        # Group by class
        class_groups = {}
        for det in detections:
            class_name = det['class_name']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(det)
        
        # Find best class (most detections with highest confidence)
        best_class = None
        best_score = 0
        
        for class_name, class_detections in class_groups.items():
            avg_confidence = np.mean([d['confidence'] for d in class_detections])
            avg_area = np.mean([d['area'] for d in class_detections])
            detection_count = len(class_detections)
            
            # Score based on confidence, area, and consistency
            score = avg_confidence * np.log(avg_area + 1) * np.log(detection_count + 1)
            
            if score > best_score:
                best_score = score
                best_class = class_name
        
        if best_class is None:
            return None
        
        # Calculate average bounding box for best class
        best_detections = class_groups[best_class]
        avg_bbox = np.mean([d['bbox'] for d in best_detections], axis=0)
        avg_confidence = np.mean([d['confidence'] for d in best_detections])
        
        return {
            'class_name': best_class,
            'confidence': float(avg_confidence),
            'bbox': avg_bbox.tolist(),  # [x1, y1, x2, y2]
            'detection_count': len(best_detections),
            'video_path': video_path
        }
    
    def extract_roi_frames(self, video_path: str, roi_info: Dict[str, Any], 
                          num_frames: int = 64) -> Optional[np.ndarray]:
        """
        Extract frames cropped to ROI
        
        Args:
            video_path: Path to video file
            roi_info: ROI information from detect_primary_subject
            num_frames: Number of frames to extract
            
        Returns:
            Cropped video array [T, H, W, C] or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            # Get ROI bounding box
            x1, y1, x2, y2 = roi_info['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Crop to ROI
                    roi_frame = frame[y1:y2, x1:x2]
                    
                    # Resize to standard size (256x256)
                    roi_frame = cv2.resize(roi_frame, (256, 256))
                    
                    # Convert BGR to RGB
                    roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                    
                    frames.append(roi_frame)
            
            cap.release()
            
            if len(frames) != num_frames:
                raise ValueError(f"Expected {num_frames} frames, got {len(frames)}")
            
            # Convert to numpy array and normalize
            video_array = np.stack(frames, axis=0)
            video_array = video_array.astype(np.float32) / 255.0
            
            logger.info(f"Extracted ROI frames from {video_path}")
            return video_array
            
        except Exception as e:
            logger.error(f"ROI frame extraction failed for {video_path}: {e}")
            return None
    
    def track_subjects(self, video_path: str, target_classes: List[str] = None) -> List[Dict[str, Any]]:
        """
        Track subjects throughout video using ByteTrack
        
        Args:
            video_path: Path to video file
            target_classes: List of target class names
            
        Returns:
            List of track information dictionaries
        """
        if target_classes is None:
            target_classes = ['person']
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            tracks = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Convert to supervision format
                detections = sv.Detections.empty()
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        # Filter for target classes
                        valid_indices = []
                        for i, box in enumerate(boxes):
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]
                            if class_name in target_classes:
                                valid_indices.append(i)
                        
                        if valid_indices:
                            # Extract valid detections
                            xyxy = boxes.xyxy[valid_indices].cpu().numpy()
                            confidence = boxes.conf[valid_indices].cpu().numpy()
                            class_ids = boxes.cls[valid_indices].cpu().numpy().astype(int)
                            
                            detections = sv.Detections(
                                xyxy=xyxy,
                                confidence=confidence,
                                class_id=class_ids
                            )
                
                # Update tracker
                detections = self.tracker.update_with_detections(detections)
                
                # Store track information
                if len(detections) > 0:
                    for i in range(len(detections)):
                        track_info = {
                            'frame_idx': frame_idx,
                            'track_id': detections.tracker_id[i] if detections.tracker_id is not None else -1,
                            'bbox': detections.xyxy[i].tolist(),
                            'confidence': detections.confidence[i],
                            'class_id': detections.class_id[i]
                        }
                        tracks.append(track_info)
                
                frame_idx += 1
                
                # Limit processing for efficiency (sample every 5th frame)
                if frame_idx % 5 != 0:
                    continue
            
            cap.release()
            
            logger.info(f"Tracked {len(set(t['track_id'] for t in tracks))} subjects in {video_path}")
            return tracks
            
        except Exception as e:
            logger.error(f"Subject tracking failed for {video_path}: {e}")
            return []

# Global instance with error handling
try:
    roi_detection_service = ROIDetectionService()
except Exception as e:
    logger.warning(f"ROI detection service unavailable: {e}")
    roi_detection_service = None