#!/usr/bin/env python3
"""Test script to validate V-JEPA 2 integration and video processing"""
import os
import sys
import logging
import numpy as np
import torch
from pathlib import Path
import cv2

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_test_video(output_path: str, duration: int = 5, fps: int = 30):
    """Create a simple test video for validation"""
    width, height = 256, 256
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create a simple moving pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving circle
        t = frame_num / total_frames
        x = int(50 + (width - 100) * t)
        y = height // 2
        
        cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Created test video: {output_path}")

def test_video_loading():
    """Test video loading functionality"""
    logger.info("Testing video loading...")
    
    # Create test video
    test_video_path = "test_video.mp4"
    create_test_video(test_video_path)
    
    try:
        from encoder_service import encoder_service
        
        # Test video loading
        video_array = encoder_service.load_video(test_video_path, num_frames=64)
        
        logger.info(f"Video array shape: {video_array.shape}")
        logger.info(f"Video array dtype: {video_array.dtype}")
        logger.info(f"Video array range: [{video_array.min():.3f}, {video_array.max():.3f}]")
        
        # Validate shape
        expected_shape = (64, 256, 256, 3)
        if video_array.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {video_array.shape}")
        
        # Validate data type and range
        if video_array.dtype != np.float32:
            raise ValueError(f"Expected float32, got {video_array.dtype}")
        
        if not (0.0 <= video_array.min() and video_array.max() <= 1.0):
            raise ValueError(f"Expected values in [0, 1], got [{video_array.min()}, {video_array.max()}]")
        
        logger.info("✓ Video loading test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Video loading test failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_video_path):
            os.remove(test_video_path)

def test_model_loading():
    """Test V-JEPA 2 model loading"""
    logger.info("Testing V-JEPA 2 model loading...")
    
    try:
        from encoder_service import encoder_service
        
        # Check if model is loaded
        if encoder_service.model is None:
            raise ValueError("Model not loaded")
        
        if encoder_service.processor is None:
            raise ValueError("Processor not loaded")
        
        logger.info(f"Model device: {encoder_service.device}")
        logger.info(f"Model type: {type(encoder_service.model)}")
        logger.info(f"Processor type: {type(encoder_service.processor)}")
        
        # Test health check
        health = encoder_service.health_check()
        logger.info(f"Health check: {health}")
        
        if health["status"] != "healthy":
            raise ValueError(f"Model not healthy: {health}")
        
        logger.info("✓ Model loading test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading test failed: {e}")
        return False

def test_video_preprocessing():
    """Test video preprocessing for V-JEPA 2"""
    logger.info("Testing video preprocessing...")
    
    # Create test video
    test_video_path = "test_video.mp4"
    create_test_video(test_video_path)
    
    try:
        from encoder_service import encoder_service
        
        # Load video
        video_array = encoder_service.load_video(test_video_path)
        
        # Test preprocessing steps
        logger.info(f"Original video shape: {video_array.shape}")
        
        # Transpose to [T, C, H, W]
        video_transposed = np.transpose(video_array, (0, 3, 1, 2))
        logger.info(f"Transposed shape: {video_transposed.shape}")
        
        # Add batch dimension
        video_batched = np.expand_dims(video_transposed, axis=0)
        logger.info(f"Batched shape: {video_batched.shape}")
        
        # Test processor
        try:
            inputs = encoder_service.processor(
                videos=video_batched,
                return_tensors="pt"
            )
            logger.info(f"Processor output keys: {list(inputs.keys())}")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: {value.shape}, {value.dtype}")
                else:
                    logger.info(f"  {key}: {type(value)}")
            
        except Exception as e:
            logger.warning(f"Primary preprocessing failed: {e}")
            
            # Try alternative format
            video_alt = np.transpose(video_batched, (0, 2, 1, 3, 4))  # [B, C, T, H, W]
            logger.info(f"Alternative shape: {video_alt.shape}")
            
            inputs = encoder_service.processor(
                videos=video_alt,
                return_tensors="pt"
            )
            logger.info("✓ Alternative preprocessing succeeded")
        
        logger.info("✓ Video preprocessing test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Video preprocessing test failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_video_path):
            os.remove(test_video_path)

def test_model_inference():
    """Test V-JEPA 2 model inference"""
    logger.info("Testing V-JEPA 2 model inference...")
    
    # Create test video
    test_video_path = "test_video.mp4"
    create_test_video(test_video_path)
    
    try:
        from encoder_service import encoder_service
        
        # Load and preprocess video
        video_array = encoder_service.load_video(test_video_path)
        video_transposed = np.transpose(video_array, (0, 3, 1, 2))
        video_batched = np.expand_dims(video_transposed, axis=0)
        
        # Process with processor
        inputs = encoder_service.processor(
            videos=video_batched,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(encoder_service.device) for k, v in inputs.items()}
        
        # Test inference
        with torch.no_grad():
            if encoder_service.device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = encoder_service.model(**inputs)
            else:
                outputs = encoder_service.model(**inputs)
        
        logger.info(f"Model output type: {type(outputs)}")
        
        # Analyze outputs
        if hasattr(outputs, 'last_hidden_state'):
            hidden_state = outputs.last_hidden_state
            logger.info(f"Hidden state shape: {hidden_state.shape}")
        elif hasattr(outputs, 'hidden_states'):
            hidden_state = outputs.hidden_states[-1]
            logger.info(f"Hidden state shape (from hidden_states): {hidden_state.shape}")
        elif isinstance(outputs, torch.Tensor):
            hidden_state = outputs
            logger.info(f"Direct tensor output shape: {hidden_state.shape}")
        else:
            logger.info(f"Output attributes: {dir(outputs)}")
            if hasattr(outputs, 'logits'):
                hidden_state = outputs.logits
                logger.info(f"Logits shape: {hidden_state.shape}")
            else:
                raise ValueError(f"Cannot extract features from output: {type(outputs)}")
        
        # Test feature extraction
        if len(hidden_state.shape) == 5:  # [B, T, H, W, D]
            global_features = hidden_state.mean(dim=(1, 2, 3)).squeeze(0)
            temporal_features = hidden_state.mean(dim=(2, 3)).squeeze(0)
        elif len(hidden_state.shape) == 4:  # [B, T, N_patches, D]
            global_features = hidden_state.mean(dim=(1, 2)).squeeze(0)
            temporal_features = hidden_state.mean(dim=2).squeeze(0)
        elif len(hidden_state.shape) == 3:  # [B, T, D]
            global_features = hidden_state.mean(dim=1).squeeze(0)
            temporal_features = hidden_state.squeeze(0)
        else:
            raise ValueError(f"Unsupported hidden state shape: {hidden_state.shape}")
        
        logger.info(f"Global features shape: {global_features.shape}")
        logger.info(f"Temporal features shape: {temporal_features.shape}")
        
        # Convert to numpy and validate
        global_np = global_features.cpu().numpy()
        temporal_np = temporal_features.cpu().numpy()
        
        logger.info(f"Global features range: [{global_np.min():.3f}, {global_np.max():.3f}]")
        logger.info(f"Temporal features range: [{temporal_np.min():.3f}, {temporal_np.max():.3f}]")
        
        logger.info("✓ Model inference test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(test_video_path):
            os.remove(test_video_path)

def test_end_to_end_encoding():
    """Test complete end-to-end video encoding"""
    logger.info("Testing end-to-end video encoding...")
    
    # Create test video
    test_video_path = "test_video.mp4"
    create_test_video(test_video_path)
    
    try:
        from encoder_service import encoder_service
        
        # Test complete encoding
        features = encoder_service.encode_video(test_video_path)
        
        logger.info(f"Video ID: {features.video_id}")
        logger.info(f"Global features shape: {features.global_features.shape}")
        logger.info(f"Temporal features shape: {features.temporal_features.shape}")
        logger.info(f"Processing time: {features.metadata['processing_time_ms']:.1f}ms")
        
        # Validate features
        if features.global_features.shape[0] != 1024:  # Expected V-JEPA 2 dimension
            logger.warning(f"Unexpected global feature dimension: {features.global_features.shape[0]}")
        
        if features.temporal_features.shape != (64, 1024):  # Expected shape
            logger.warning(f"Unexpected temporal feature shape: {features.temporal_features.shape}")
        
        # Check normalization
        global_norm = np.linalg.norm(features.global_features)
        logger.info(f"Global features L2 norm: {global_norm:.3f}")
        
        temporal_norms = np.linalg.norm(features.temporal_features, axis=1)
        logger.info(f"Temporal features L2 norms range: [{temporal_norms.min():.3f}, {temporal_norms.max():.3f}]")
        
        logger.info("✓ End-to-end encoding test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ End-to-end encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(test_video_path):
            os.remove(test_video_path)

def main():
    """Run all V-JEPA 2 integration tests"""
    print("🧪 V-JEPA 2 Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Video Loading", test_video_loading),
        ("Model Loading", test_model_loading),
        ("Video Preprocessing", test_video_preprocessing),
        ("Model Inference", test_model_inference),
        ("End-to-End Encoding", test_end_to_end_encoding)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! V-JEPA 2 integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)