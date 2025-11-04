"""
Baseline Comparison - Is motion-based search better than alternatives?

Compares MotionMatch (V-JEPA motion) against:
1. Random selection (baseline)
2. Visual similarity (CLIP - appearance-based)
"""
import sys
from pathlib import Path
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests" / "integration"))

from test_api_client import MotionMatchClient


def get_video_category(video_path):
    """Extract category from video filename"""
    name = Path(video_path).stem.lower()
    for category in ["jump", "run", "climb", "surf", "walk", "dance"]:
        if name.startswith(category):
            return category
    return "other"


def random_baseline(query_video, all_videos, top_k=10):
    """Random selection baseline"""
    # Exclude query video itself
    candidates = [v for v in all_videos if v != query_video]
    return random.sample(candidates, min(top_k, len(candidates)))


def calculate_precision(results, query_category):
    """Calculate what % of results match query category"""
    if not results:
        return 0
    
    same_category = sum(1 for v in results if get_video_category(v) == query_category)
    return (same_category / len(results)) * 100


def compare_methods():
    """Compare different search methods"""
    print("=" * 60)
    print("Baseline Comparison Test")
    print("=" * 60)
    
    client = MotionMatchClient()
    
    # Get all videos
    video_dir = Path("testvideo")
    all_videos = list(video_dir.glob("*.mp4"))
    
    if len(all_videos) < 10:
        print("\n⚠️  Need at least 10 videos for meaningful comparison")
        return
    
    print(f"\n📹 Testing with {len(all_videos)} videos")
    
    # Test queries
    test_queries = [v for v in all_videos if get_video_category(v) != "other"][:5]
    
    if not test_queries:
        print("\n⚠️  No categorized videos found for testing")
        return
    
    print(f"🔍 Running {len(test_queries)} test queries...\n")
    
    results = {
        "random": [],
        "motion": []
    }
    
    for query_video in test_queries:
        query_category = get_video_category(query_video)
        print(f"Query: {query_video.name} (category: {query_category})")
        
        # Method 1: Random baseline
        random_results = random_baseline(str(query_video), [str(v) for v in all_videos])
        random_precision = calculate_precision(random_results, query_category)
        results["random"].append(random_precision)
        print(f"  Random: {random_precision:.1f}% precision")
        
        # Method 2: MotionMatch (V-JEPA)
        try:
            search_response = client.search_video(str(query_video), top_k=10)
            motion_results = [r["video_path"] for r in search_response.get("results", [])]
            motion_precision = calculate_precision(motion_results, query_category)
            results["motion"].append(motion_precision)
            print(f"  Motion: {motion_precision:.1f}% precision")
        except Exception as e:
            print(f"  Motion: Error - {e}")
            results["motion"].append(0)
        
        print()
    
    # Calculate averages
    avg_random = sum(results["random"]) / len(results["random"]) if results["random"] else 0
    avg_motion = sum(results["motion"]) / len(results["motion"]) if results["motion"] else 0
    
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Average Precision:")
    print(f"  Random Baseline: {avg_random:.1f}%")
    print(f"  MotionMatch:     {avg_motion:.1f}%")
    
    if avg_motion > 0:
        improvement = ((avg_motion - avg_random) / avg_random * 100) if avg_random > 0 else 0
        improvement_factor = (avg_motion / avg_random) if avg_random > 0 else 0
        
        print(f"\n📈 Improvement:")
        print(f"  Absolute: +{avg_motion - avg_random:.1f} percentage points")
        print(f"  Relative: {improvement:.1f}% better")
        print(f"  Factor: {improvement_factor:.1f}x")
        
        print("\n💡 Interpretation:")
        if improvement_factor >= 2.0:
            print("  ✅ EXCELLENT - Motion-based search is significantly better")
            print("     Provides clear value over random selection")
        elif improvement_factor >= 1.5:
            print("  ⚠️  MODERATE - Motion-based search is somewhat better")
            print("     Shows promise but needs improvement")
        elif improvement_factor >= 1.1:
            print("  ⚠️  SLIGHT - Motion-based search is marginally better")
            print("     May not provide enough value")
        else:
            print("  ❌ POOR - Motion-based search is not better than random")
            print("     System needs significant improvement")
        
        print(f"\n  Target: ≥2.0x improvement for real value")
        print(f"  Your score: {improvement_factor:.1f}x")
    
    return results


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    compare_methods()
