"""Quick test to check if V-JEPA 2 features are working correctly"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from motionmatch.services.encoder import encoder_service
import numpy as np

# Test with a video
video_path = "testvideo/jump1.mp4"  # Change this to any video you have

print(f"Testing feature extraction on: {video_path}")
print("=" * 60)

try:
    features = encoder_service.encode_video(video_path)
    
    print(f"\n✓ Encoding successful!")
    print(f"  Global features shape: {features.global_features.shape}")
    print(f"  Temporal features shape: {features.temporal_features.shape}")
    
    # Check if temporal features are actually different across time
    temporal = features.temporal_features
    
    # Calculate variance across time dimension
    variance_across_time = np.var(temporal, axis=0).mean()
    print(f"\n  Variance across time: {variance_across_time:.6f}")
    
    # Check if all temporal features are identical
    unique_rows = len(set(tuple(row) for row in temporal))
    print(f"  Unique temporal features: {unique_rows}/{len(temporal)}")
    
    # Check similarity between first and last frame
    first_frame = temporal[0]
    last_frame = temporal[-1]
    similarity = np.dot(first_frame, last_frame) / (np.linalg.norm(first_frame) * np.linalg.norm(last_frame))
    print(f"  Similarity (first vs last frame): {similarity:.4f}")
    
    print("\n" + "=" * 60)
    print("Interpretation:")
    print("=" * 60)
    
    if unique_rows == 1:
        print("❌ PROBLEM: All temporal features are identical!")
        print("   This means we're not capturing temporal/motion information.")
        print("   The model is only giving us a single video-level embedding.")
    elif unique_rows == len(temporal):
        print("✅ GOOD: All temporal features are unique!")
        print("   The model is providing frame-level information.")
    else:
        print(f"⚠️  PARTIAL: {unique_rows} unique features out of {len(temporal)}")
        print("   Some temporal information, but may be limited.")
    
    if variance_across_time < 0.001:
        print("\n❌ Low variance: Features don't change much over time")
        print("   Motion information may be weak")
    elif variance_across_time > 0.01:
        print(f"\n✅ Good variance: Features change significantly over time")
        print("   Motion information is being captured")
    else:
        print(f"\n⚠️  Moderate variance: Some temporal variation")
    
    if similarity > 0.99:
        print(f"\n❌ Very high similarity between frames: {similarity:.4f}")
        print("   Temporal features may be too similar")
    elif similarity < 0.9:
        print(f"\n✅ Good variation between frames: {similarity:.4f}")
        print("   Temporal dynamics are captured")
    else:
        print(f"\n⚠️  Moderate similarity: {similarity:.4f}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
