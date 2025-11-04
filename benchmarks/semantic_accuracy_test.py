"""
Semantic Accuracy Test - Does MotionMatch understand motion categories?

This test measures if the system can distinguish between different types of motion
(jumping vs running vs climbing, etc.)
"""
import sys
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests" / "integration"))

from test_api_client import MotionMatchClient


def categorize_videos(video_dir="testvideo"):
    """Automatically categorize videos based on filename"""
    categories = defaultdict(list)
    
    video_dir = Path(video_dir)
    for video in video_dir.glob("*.mp4"):
        # Extract category from filename (e.g., "jump1.mp4" -> "jump")
        name = video.stem.lower()
        
        # Find category prefix
        for category in ["jump", "run", "climb", "surf", "walk", "dance"]:
            if name.startswith(category):
                categories[category].append(video)
                break
        else:
            categories["other"].append(video)
    
    return categories


def test_semantic_accuracy():
    """Test if searches return videos from the same category"""
    print("=" * 60)
    print("Semantic Accuracy Test")
    print("=" * 60)
    
    client = MotionMatchClient()
    
    # Categorize videos
    categories = categorize_videos()
    
    print(f"\n📁 Found {len(categories)} categories:")
    for cat, videos in categories.items():
        print(f"  {cat}: {len(videos)} videos")
    
    if len(categories) < 2:
        print("\n⚠️  Need at least 2 categories to test semantic accuracy")
        print("   Add videos with different motion types (jump, run, walk, etc.)")
        return
    
    # Test each category
    results = []
    
    print("\n🔍 Testing semantic accuracy...")
    for category, videos in categories.items():
        if category == "other" or len(videos) < 2:
            continue
        
        # Use first video as query
        query_video = videos[0]
        print(f"\n  Testing category: {category}")
        print(f"  Query: {query_video.name}")
        
        try:
            search_results = client.search_video(str(query_video), top_k=10)
            
            # Count how many results are from same category
            same_category = 0
            different_category = 0
            
            for result in search_results.get("results", []):
                result_path = Path(result["video_path"])
                result_name = result_path.stem.lower()
                
                # Check if result is same category
                if result_name.startswith(category):
                    same_category += 1
                else:
                    different_category += 1
            
            total_results = same_category + different_category
            accuracy = (same_category / total_results * 100) if total_results > 0 else 0
            
            print(f"  Results: {same_category}/{total_results} same category ({accuracy:.1f}%)")
            
            results.append({
                "category": category,
                "accuracy": accuracy,
                "same": same_category,
                "different": different_category
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Calculate overall accuracy
    if results:
        avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
        
        print("\n" + "=" * 60)
        print("SEMANTIC ACCURACY RESULTS")
        print("=" * 60)
        
        for r in results:
            status = "✓" if r["accuracy"] >= 70 else "✗"
            print(f"{status} {r['category']}: {r['accuracy']:.1f}% ({r['same']}/{r['same']+r['different']})")
        
        print(f"\n📊 Overall Accuracy: {avg_accuracy:.1f}%")
        
        # Interpretation
        print("\n💡 Interpretation:")
        if avg_accuracy >= 70:
            print("  ✅ EXCELLENT - System understands motion semantics well")
            print("     This provides real value over visual similarity")
        elif avg_accuracy >= 50:
            print("  ⚠️  MODERATE - System has some semantic understanding")
            print("     May need model tuning or better features")
        else:
            print("  ❌ POOR - System struggles with motion semantics")
            print("     Not much better than random/visual similarity")
        
        print(f"\n  Target: ≥70% for real value")
        print(f"  Your score: {avg_accuracy:.1f}%")
        
        return avg_accuracy
    else:
        print("\n⚠️  No results to analyze")
        return 0


if __name__ == "__main__":
    test_semantic_accuracy()
