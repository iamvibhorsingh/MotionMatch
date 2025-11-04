"""
Performance testing script - measures search latency improvements

Tests:
1. Cold start (first search, no cache)
2. Warm start (repeat search, with cache)
3. Different video search (cache miss)
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests" / "integration"))

from test_api_client import MotionMatchClient


def test_search_performance():
    """Test search performance with caching"""
    print("=" * 70)
    print("Search Performance Test")
    print("=" * 70)
    
    client = MotionMatchClient()
    
    # Get test videos
    test_dir = Path("testvideo")
    videos = list(test_dir.glob("*.mp4"))[:3]  # Test with 3 videos
    
    if len(videos) < 2:
        print("Need at least 2 test videos")
        return
    
    print(f"\n📹 Testing with {len(videos)} videos")
    
    results = {
        "cold_start": [],
        "warm_start": [],
        "cache_miss": []
    }
    
    # Test 1: Cold start (first search)
    print("\n🧊 Test 1: Cold Start (No Cache)")
    print("-" * 70)
    
    for video in videos:
        print(f"  Searching: {video.name}...", end=" ")
        start = time.time()
        try:
            response = client.search_video(str(video), top_k=10)
            elapsed = (time.time() - start) * 1000
            results["cold_start"].append(elapsed)
            print(f"{elapsed:.0f}ms")
        except Exception as e:
            print(f"Error: {e}")
    
    # Test 2: Warm start (repeat same searches)
    print("\n🔥 Test 2: Warm Start (With Cache)")
    print("-" * 70)
    
    for video in videos:
        print(f"  Searching: {video.name}...", end=" ")
        start = time.time()
        try:
            response = client.search_video(str(video), top_k=10)
            elapsed = (time.time() - start) * 1000
            results["warm_start"].append(elapsed)
            print(f"{elapsed:.0f}ms")
        except Exception as e:
            print(f"Error: {e}")
    
    # Test 3: Cache miss (different videos)
    print("\n❄️  Test 3: Cache Miss (Different Videos)")
    print("-" * 70)
    
    other_videos = list(test_dir.glob("*.mp4"))[3:6]  # Different videos
    if other_videos:
        for video in other_videos:
            print(f"  Searching: {video.name}...", end=" ")
            start = time.time()
            try:
                response = client.search_video(str(video), top_k=10)
                elapsed = (time.time() - start) * 1000
                results["cache_miss"].append(elapsed)
                print(f"{elapsed:.0f}ms")
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("  (Not enough videos for cache miss test)")
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)
    
    if results["cold_start"]:
        cold_avg = sum(results["cold_start"]) / len(results["cold_start"])
        print(f"\n🧊 Cold Start (First Search):")
        print(f"  Average: {cold_avg:.0f}ms")
        print(f"  Min: {min(results['cold_start']):.0f}ms")
        print(f"  Max: {max(results['cold_start']):.0f}ms")
    
    if results["warm_start"]:
        warm_avg = sum(results["warm_start"]) / len(results["warm_start"])
        print(f"\n🔥 Warm Start (Cached):")
        print(f"  Average: {warm_avg:.0f}ms")
        print(f"  Min: {min(results['warm_start']):.0f}ms")
        print(f"  Max: {max(results['warm_start']):.0f}ms")
        
        if results["cold_start"]:
            speedup = cold_avg / warm_avg
            print(f"\n📈 Cache Speedup: {speedup:.1f}x faster")
            print(f"  Time saved: {cold_avg - warm_avg:.0f}ms per query")
    
    if results["cache_miss"]:
        miss_avg = sum(results["cache_miss"]) / len(results["cache_miss"])
        print(f"\n❄️  Cache Miss:")
        print(f"  Average: {miss_avg:.0f}ms")
    
    # Interpretation
    print("\n💡 Interpretation:")
    print("-" * 70)
    
    if results["warm_start"]:
        if warm_avg < 2000:
            print("  ✅ EXCELLENT - Cached searches are fast (<2s)")
            print("     System is ready for production use")
        elif warm_avg < 5000:
            print("  ⚠️  GOOD - Cached searches are acceptable (<5s)")
            print("     Consider further optimizations")
        else:
            print("  ❌ SLOW - Cached searches still too slow (>5s)")
            print("     Need more optimization work")
    
    if results["cold_start"] and results["warm_start"]:
        if speedup >= 5:
            print(f"  ✅ Cache is very effective ({speedup:.1f}x speedup)")
        elif speedup >= 2:
            print(f"  ⚠️  Cache provides moderate benefit ({speedup:.1f}x speedup)")
        else:
            print(f"  ❌ Cache not providing enough benefit ({speedup:.1f}x speedup)")
    
    print("\n🎯 Recommendations:")
    print("-" * 70)
    if results["cold_start"] and cold_avg > 10000:
        print("  • Cold start is slow - consider pre-computing common queries")
        print("  • Run: python scripts/manage_cache.py --precompute")
    
    if results["warm_start"] and warm_avg > 2000:
        print("  • Warm start still slow - check Milvus search parameters")
        print("  • Consider reducing NUM_FRAMES in config")
    
    if results["warm_start"] and warm_avg < 2000:
        print("  • Performance is good! Focus on user testing and features")


if __name__ == "__main__":
    test_search_performance()
