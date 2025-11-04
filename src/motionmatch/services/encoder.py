"""V-JEPA 2 Encoder Service for MotionMatch MVP"""
import os
import time
import logging
from typing import Optional
import torch
import numpy as np
import cv2
from transformers import AutoModel, AutoVideoProcessor
from motionmatch.core.config import config
from motionmatch.db.models import VideoFeatures

logger = logging.getLogger(__name__)

class VJEPA2EncoderService:
    """V-JEPA 2 video encoder service"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = config.DEVICE
        self._load_model()
    
    def _load_model(self):
        """Load V-JEPA 2 model and processor"""
        try:
            logger.info(f"Loading model {config.MODEL_NAME} on {self.device}")
            
            # Load model with appropriate dtype
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.model = AutoModel.from_pretrained(
                config.MODEL_NAME,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            self.processor = AutoVideoProcessor.from_pretrained(config.MODEL_NAME)
            
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            
            self.model.eval()
            
            # Optional: Compile model for faster inference (PyTorch 2.0+)
            if config.TORCH_COMPILE and self.device == "cuda":
                try:
                    logger.info("Compiling model with torch.compile for faster inference...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Model compiled successfully")
                except Exception as e:
                    logger.warning(f"Failed to compile model: {e}")
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_video(self, video_path: str, num_frames: int = None) -> np.ndarray:
        """Load and preprocess video - optimized version"""
        if num_frames is None:
            num_frames = config.NUM_FRAMES
            
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            # Pre-allocate array for better performance
            frames = np.zeros((num_frames, config.FRAME_SIZE, config.FRAME_SIZE, 3), dtype=np.uint8)
            
            for i, idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB and resize in one operation
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (config.FRAME_SIZE, config.FRAME_SIZE), 
                                     interpolation=cv2.INTER_LINEAR)  # Faster interpolation
                    frames[i] = frame
            
            cap.release()
            
            # Convert to float32 and normalize to [0, 1] - vectorized operation
            video_array = frames.astype(np.float32) / 255.0
            
            return video_array
            
        except Exception as e:
            logger.error(f"Failed to load video {video_path}: {e}")
            raise
    
    def encode_video(self, video_path: str, use_amp: bool = True) -> VideoFeatures:
        """Encode video to V-JEPA 2 features
        
        Args:
            video_path: Path to video file
            use_amp: Use automatic mixed precision for faster GPU inference
        """
        start_time = time.time()
        
        try:
            # Load video
            video_array = self.load_video(video_path)
            
            # Validate video array
            if video_array.shape[0] != config.NUM_FRAMES:
                raise ValueError(f"Expected {config.NUM_FRAMES} frames, got {video_array.shape[0]}")
            if video_array.shape[1:3] != (config.FRAME_SIZE, config.FRAME_SIZE):
                raise ValueError(f"Expected frame size {config.FRAME_SIZE}x{config.FRAME_SIZE}, got {video_array.shape[1:3]}")
            if video_array.shape[3] != 3:
                raise ValueError(f"Expected 3 color channels, got {video_array.shape[3]}")
            
            # Preprocess for V-JEPA 2
            # V-JEPA 2 expects videos in format [T, C, H, W] (without batch dimension for processor)
            # Current video_array is [T, H, W, C], need to transpose to [T, C, H, W]
            video_array = np.transpose(video_array, (0, 3, 1, 2))  # [T, C, H, W]
            
            logger.debug(f"Video array shape for processing: {video_array.shape}")
            
            # Convert to torch tensor
            video_tensor = torch.from_numpy(video_array).float()
            
            # Process with AutoVideoProcessor
            inputs = self.processor(video_tensor, return_tensors="pt")
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # V-JEPA 2 inference - get full model output for temporal features
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs, output_hidden_states=True)
                else:
                    outputs = self.model(**inputs, output_hidden_states=True)
            
            # Extract features from model output
            # V-JEPA 2 returns last_hidden_state with shape [B, num_patches, D]
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state  # [B, num_patches, D]
                logger.debug(f"Hidden states shape: {hidden_states.shape}")
                
                # Global features: mean pool over all patches
                global_features = hidden_states.mean(dim=1).squeeze(0)  # [D]
                
                # Temporal features: reshape patches to temporal dimension
                # V-JEPA 2 processes video as spatiotemporal patches
                # For 64 frames, we can group patches by time
                num_patches = hidden_states.shape[1]
                
                # Try to extract temporal structure from patches
                # Assuming patches are ordered temporally (common in video transformers)
                if num_patches >= config.NUM_FRAMES:
                    # Sample patches uniformly across time
                    patch_indices = torch.linspace(0, num_patches - 1, config.NUM_FRAMES, dtype=torch.long)
                    temporal_features = hidden_states[0, patch_indices, :]  # [T, D]
                else:
                    # If fewer patches than frames, interpolate
                    temporal_features = hidden_states.squeeze(0)  # [num_patches, D]
                    # Pad or interpolate to NUM_FRAMES
                    if temporal_features.shape[0] < config.NUM_FRAMES:
                        # Repeat last feature to fill
                        padding = config.NUM_FRAMES - temporal_features.shape[0]
                        temporal_features = torch.cat([
                            temporal_features,
                            temporal_features[-1:].repeat(padding, 1)
                        ], dim=0)
                
                logger.info(f"Extracted temporal features shape: {temporal_features.shape}")
                
            elif hasattr(outputs, 'pooler_output'):
                # Fallback: use pooler output if available
                global_features = outputs.pooler_output.squeeze(0)
                temporal_features = global_features.unsqueeze(0).repeat(config.NUM_FRAMES, 1)
                logger.warning("Using pooler_output, temporal information may be limited")
                
            else:
                # Last resort: try get_vision_features
                video_embeddings = self.model.get_vision_features(**inputs)
                global_features = video_embeddings.squeeze(0) if len(video_embeddings.shape) == 2 else video_embeddings.mean(dim=1).squeeze(0)
                temporal_features = global_features.unsqueeze(0).repeat(config.NUM_FRAMES, 1)
                logger.warning("Using get_vision_features fallback, temporal information may be limited")
            
            # Convert to numpy
            global_features = global_features.cpu().numpy()
            temporal_features = temporal_features.cpu().numpy()
            
            # Validate feature shapes
            if global_features.ndim != 1:
                raise ValueError(f"Expected 1D global features, got shape: {global_features.shape}")
            if temporal_features.ndim != 2:
                raise ValueError(f"Expected 2D temporal features, got shape: {temporal_features.shape}")
            
            # Validate dimensions match expected
            actual_dim = global_features.shape[0]
            if actual_dim != config.VECTOR_DIM:
                logger.error(f"Feature dimension mismatch: expected {config.VECTOR_DIM}, got {actual_dim}")
                raise ValueError(f"Feature dimension {actual_dim} doesn't match configured {config.VECTOR_DIM}. Update VECTOR_DIM in config.")
            
            logger.debug(f"Global features shape: {global_features.shape}")
            logger.debug(f"Temporal features shape: {temporal_features.shape}")
            
            # L2 normalize features (with epsilon to avoid division by zero)
            eps = 1e-8
            
            # Normalize temporal features first
            temporal_norms = np.linalg.norm(temporal_features, axis=1, keepdims=True)
            temporal_norms = np.maximum(temporal_norms, eps)
            temporal_features = temporal_features / temporal_norms
            
            # For global features, use the original mean pooling from hidden states
            # This preserves V-JEPA 2's learned representations
            global_features = temporal_features.mean(axis=0)
            
            # Normalize global features
            global_norm = np.linalg.norm(global_features)
            if global_norm > eps:
                global_features = global_features / global_norm
            else:
                logger.warning("Global features have near-zero norm, skipping normalization")
            
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = {
                "video_path": video_path,
                "num_frames": config.NUM_FRAMES,
                "frame_size": config.FRAME_SIZE,
                "processing_time_ms": processing_time * 1000,
                "global_shape": list(global_features.shape),
                "temporal_shape": list(temporal_features.shape)
            }
            
            # Extract video ID - use absolute path without extension for uniqueness
            video_id = os.path.abspath(video_path).rsplit('.', 1)[0]
            
            return VideoFeatures(
                video_id=video_id,
                global_features=global_features,
                temporal_features=temporal_features,
                metadata=metadata,
                created_at=time.time()
            )
            
        except Exception as e:
            logger.error(f"Failed to encode video {video_path}: {e}")
            raise
    
    def encode_batch(self, video_paths: list, batch_size: int = None) -> list:
        """Encode multiple videos in batches for better GPU utilization
        
        Args:
            video_paths: List of video file paths
            batch_size: Batch size for processing (default: config.BATCH_SIZE)
        
        Returns:
            List of VideoFeatures objects
        """
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        results = []
        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i:i + batch_size]
            
            # Process batch sequentially for now (true batching requires more work)
            for path in batch_paths:
                try:
                    features = self.encode_video(path)
                    results.append(features)
                except Exception as e:
                    logger.error(f"Failed to encode {path}: {e}")
                    results.append(None)
        
        return results
    
    def health_check(self) -> dict:
        """Health check for the encoder service"""
        try:
            gpu_memory = None
            gpu_utilization = None
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                # Get GPU utilization if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = util.gpu
                    pynvml.nvmlShutdown()
                except:
                    pass
            
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "device": self.device,
                "gpu_memory_mb": gpu_memory,
                "gpu_utilization_percent": gpu_utilization
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": False,
                "device": self.device
            }

# Global encoder instance
encoder_service = VJEPA2EncoderService()