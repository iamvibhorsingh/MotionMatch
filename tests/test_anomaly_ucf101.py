"""
Quick anomaly detection test using your existing UCF101 videos
"""
import requests
from pathlib import Path

API_BASE = "http://localhost:8000/v1"

def test_with_ucf101():
    print("=" * 70)
    print("Quick Anomaly Detection Test with UCF101")
    print("=" * 70)
    
    video_dir = Path("testvideo")
    
    # Define "normal" behavior - use one action class
    normal_pattern = "v_Walking*"  # Walking is "normal"
    normal_videos = list(video_dir.glob(normal_pattern))[:10]
    
    if not normal_videos:
        print("❌ No walking videos found. Using any videos as baseline...")
        normal_videos = list(video_dir.glob("*.mp4"))[:10]
    
    print(f"\n[1] Establishing baseline from {len(normal_videos)} 'normal' videos")
    print(f"    Pattern: {normal_pattern}")
    
    # Establish baseline
    try:
        response = requests.post(
            f"{API_BASE}/anomaly/baseline",
            json=[str(v) for v in normal_videos],
            timeout=300
        )
        
        if response.status_code == 200:
            baseline = response.json()
            print(f"    ✓ Baseline established")
            print(f"      Mean motion: {baseline['baseline']['mean_motion_magnitude']:.6f}")
        else:
            print(f"    ✗ Failed: {response.text}")
            return
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return
    
    # Test different action classes
    print(f"\n[2] Testing different action classes")
    print("-" * 70)
    
    test_patterns = [
        ("Walking (Normal)", "v_Walking*", 11, 12),  # Should be normal
        ("Biking (Anomaly)", "v_Biking*", 0, 1),     # Should be anomaly
        ("ApplyEyeMakeup (Anomaly)", "v_ApplyEyeMakeup*", 0, 1),  # Should be anomaly
        ("Running (Similar?)", "v_Running*", 0, 1),  # Might be similar to walking
        ("Jumping (Anomaly)", "v_Jumping*", 0, 1),   # Should be anomaly
    ]
    
    results = []
    
    for label, pattern, start, end in test_patterns:
        videos = list(video_dir.glob(pattern))[start:end]
        
        if not videos:
            print(f"\n  {label}: No videos found")
            continue
        
        video = str(videos[0])
        print(f"\n  Testing: {label}")
        print(f"    Video: {Path(video).name}")
        
        try:
            response = requests.post(
                f"{API_BASE}/anomaly/detect",
                params={"video_path": video, "threshold": 2.0},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                status = "🚨 ANOMALY" if result['is_anomaly'] else "✓ Normal"
                print(f"    Result: {status}")
                print(f"    Score: {result['anomaly_score']:.2f} (threshold: 2.0)")
                print(f"    Confidence: {result['confidence']:.1f}%")
                
                results.append({
                    "label": label,
                    "is_anomaly": result['is_anomaly'],
                    "score": result['anomaly_score']
                })
            else:
                print(f"    ✗ Failed: {response.text}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    if results:
        print("\nAction Class          | Status    | Score")
        print("-" * 50)
        for r in results:
            status = "ANOMALY" if r['is_anomaly'] else "Normal "
            print(f"{r['label']:20} | {status:9} | {r['score']:.2f}")
        
        # Interpretation
        print("\n💡 Interpretation:")
        anomalies = [r for r in results if r['is_anomaly']]
        normals = [r for r in results if not r['is_anomaly']]
        
        print(f"  • Detected {len(anomalies)} anomalies out of {len(results)} tests")
        print(f"  • {len(normals)} videos classified as normal")
        
        if len(anomalies) >= len(results) - 1:  # All except walking should be anomalies
            print("\n  ✅ System is working! Different actions detected as anomalies")
        else:
            print("\n  ⚠️  System may need tuning - adjust threshold or add more baseline videos")
    
    print("\n📊 Next Steps:")
    print("  1. Download UCF-Crime dataset for real anomaly detection")
    print("  2. Use more baseline videos (20-50) for better accuracy")
    print("  3. Adjust threshold based on your use case")
    print("  4. Test with real surveillance footage")


if __name__ == "__main__":
    test_with_ucf101()
