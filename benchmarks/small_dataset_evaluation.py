"""
Better evaluation metrics for small datasets

Instead of precision (which is misleading), we measure:
1. Recall - Did we find ALL relevant videos?
2. Rank Quality - Are relevant videos ranked higher?
3. Top-1 Accuracy - Is the best match actually similar?
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

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


def categorize_videos(video_dir="testvideo"):
    """Group videos by category"""
    categories = defaultdict(list)
    
    video_dir = Path(video_dir)
    for video in video_dir.glob("*.mp4"):
        category = get_video_category(video)
        categories[category].append(video)
    
    return categories


def calculate_metrics(query_video, results, all_videos_in_category):
    """Calculate multiple metrics for small datasets"""
    query_category = get_video_category(query_video)
    
    # Get result categories and positions
    result_categories = []
    relevant_positions = []
    
    for i, result in enumerate(results):
        result_path = Path(result["video_path"])
        result_category = get_video_category(result_path)
        result_categories.append(result_category)
        
        if result_category == query_category and result_path.name != Path(query_video).name:
            relevant_positions.append(i + 1)  # 1-indexed
    
    # Calculate metrics
    num_relevant = len(all_videos_in_category) - 1  # Exclude query itself
    num_found = len(relevant_positions)
    
    # 1. Recall - Did we find all relevant videos?
    recall = (num_found / num_relevant * 100) if num_relevant > 0 else 0
    
    # 2. Top-1 Accuracy - Is the first result from same category?
    top1_correct = result_categories[0] == query_category if result_categories else False
    
    # 3. Mean Reciprocal Rank (MRR) - How high are relevant results ranked?
    mrr = (1 / relevant_positions[0]) if relevant_positions else 0
    
    # 4. Average Precision - Considers both precision and rank
    precisions = []
    num_relevant_seen = 0
    for i, cat in enumerate(result_categories):
        if cat == query_category:
            num_relevant_seen += 1
            precision_at_i = num_relevant_seen / (i + 1)
            precisions.append(precision_at_i)
    
    avg_precision = (sum(precisions) / num_relevant) if num_relevant > 0 and precisions else 0
    
    return {
        "recall": recall,
        "top1_correct": top1_correct,
        "mrr": mrr,
        "avg_precision": avg_precision,
        "num_found": num_found,
        "num_relevant": num_relevant,
        "relevant_positions": relevant_positions
    }


def evaluate_small_dataset():
    """Comprehensive evaluation for small datasets"""
    print("=" * 70)
    print("Small Dataset Evaluation - Better Metrics")
    print("=" * 70)
    
    client = MotionMatchClient()
    categories = categorize_videos()
    
    print(f"\n📁 Dataset Overview:")
    total_videos = sum(len(videos) for videos in categories.values())
    print(f"  Total videos: {total_videos}")
    for cat, videos in sorted(categories.items()):
        print(f"  {cat}: {len(videos)} videos")
    
    # Test each category
    all_results = []
    
    print("\n" + "=" * 70)
    print("Testing Each Category")
    print("=" * 70)
    
    for category, videos in categories.items():
        if category == "other" or len(videos) < 2:
            continue
        
        print(f"\n📹 Category: {category.upper()} ({len(videos)} videos)")
        print("-" * 70)
        
        category_results = []
        
        for query_video in videos:
            print(f"\n  Query: {query_video.name}")
            
            try:
                search_response = client.search_video(str(query_video), top_k=10)
                results = search_response.get("results", [])
                
                metrics = calculate_metrics(str(query_video), results, videos)
                category_results.append(metrics)
                
                # Show results
                print(f"    ✓ Recall: {metrics['recall']:.0f}% ({metrics['num_found']}/{metrics['num_relevant']} found)")
                print(f"    ✓ Top-1: {'✓ Correct' if metrics['top1_correct'] else '✗ Wrong'}")
                print(f"    ✓ MRR: {metrics['mrr']:.3f}")
                
                if metrics['relevant_positions']:
                    print(f"    ✓ Relevant videos at positions: {metrics['relevant_positions']}")
                else:
                    print(f"    ✗ No relevant videos found in top-10")
                
                # Show top-3 results
                print(f"    Top-3 results:")
                for i, result in enumerate(results[:3], 1):
                    result_path = Path(result["video_path"])
                    result_cat = get_video_category(result_path)
                    match = "✓" if result_cat == category else "✗"
                    score = result.get("similarity_score", 0) * 100
                    print(f"      {i}. {match} {result_path.name} ({result_cat}) - {score:.1f}%")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
        
        # Category summary
        if category_results:
            avg_recall = np.mean([r["recall"] for r in category_results])
            avg_mrr = np.mean([r["mrr"] for r in category_results])
            top1_accuracy = np.mean([r["top1_correct"] for r in category_results]) * 100
            
            print(f"\n  📊 {category.upper()} Summary:")
            print(f"    Avg Recall: {avg_recall:.1f}%")
            print(f"    Top-1 Accuracy: {top1_accuracy:.1f}%")
            print(f"    Avg MRR: {avg_mrr:.3f}")
            
            all_results.extend(category_results)
    
    # Overall summary
    if all_results:
        print("\n" + "=" * 70)
        print("OVERALL RESULTS")
        print("=" * 70)
        
        avg_recall = np.mean([r["recall"] for r in all_results])
        avg_mrr = np.mean([r["mrr"] for r in all_results])
        top1_accuracy = np.mean([r["top1_correct"] for r in all_results]) * 100
        avg_ap = np.mean([r["avg_precision"] for r in all_results])
        
        print(f"\n📊 Key Metrics:")
        print(f"  Recall@10:        {avg_recall:.1f}%  (Did we find all similar videos?)")
        print(f"  Top-1 Accuracy:   {top1_accuracy:.1f}%  (Is best match correct?)")
        print(f"  Mean Reciprocal Rank: {avg_mrr:.3f}  (How high are good results?)")
        print(f"  Mean Avg Precision:   {avg_ap:.3f}  (Overall ranking quality)")
        
        print("\n💡 Interpretation:")
        print("-" * 70)
        
        # Recall interpretation
        if avg_recall >= 80:
            print("  ✅ RECALL: Excellent - Finding almost all similar videos")
        elif avg_recall >= 60:
            print("  ⚠️  RECALL: Good - Finding most similar videos")
        elif avg_recall >= 40:
            print("  ⚠️  RECALL: Moderate - Missing some similar videos")
        else:
            print("  ❌ RECALL: Poor - Missing many similar videos")
        
        # Top-1 interpretation
        if top1_accuracy >= 80:
            print("  ✅ TOP-1: Excellent - Best match is usually correct")
        elif top1_accuracy >= 60:
            print("  ⚠️  TOP-1: Good - Best match is often correct")
        elif top1_accuracy >= 40:
            print("  ⚠️  TOP-1: Moderate - Best match is sometimes correct")
        else:
            print("  ❌ TOP-1: Poor - Best match is rarely correct")
        
        # MRR interpretation
        if avg_mrr >= 0.7:
            print("  ✅ RANKING: Excellent - Similar videos ranked very high")
        elif avg_mrr >= 0.5:
            print("  ⚠️  RANKING: Good - Similar videos ranked fairly high")
        elif avg_mrr >= 0.3:
            print("  ⚠️  RANKING: Moderate - Similar videos ranked mid-range")
        else:
            print("  ❌ RANKING: Poor - Similar videos ranked low")
        
        print("\n🎯 Value Assessment:")
        print("-" * 70)
        
        # Overall value assessment
        if avg_recall >= 80 and top1_accuracy >= 80:
            print("  ✅ HIGH VALUE - System works very well!")
            print("     → Focus on: Speed optimization, UI/UX, user testing")
            print("     → Ready for: Real-world pilot with target users")
        elif avg_recall >= 60 and top1_accuracy >= 60:
            print("  ⚠️  MODERATE VALUE - System shows promise")
            print("     → Focus on: Model tuning, more training data")
            print("     → Need: More test videos to validate")
        elif avg_recall >= 40 or top1_accuracy >= 40:
            print("  ⚠️  LIMITED VALUE - System needs improvement")
            print("     → Focus on: Feature engineering, different model")
            print("     → Consider: Alternative approaches")
        else:
            print("  ❌ LOW VALUE - System not working well")
            print("     → Focus on: Root cause analysis, model debugging")
            print("     → Consider: Major changes or pivot")
        
        print("\n📈 Next Steps:")
        print("-" * 70)
        if avg_recall >= 60:
            print("  1. Add more test videos (target: 50+ videos, 5+ per category)")
            print("  2. Fix search latency (current: ~14s, target: <2s)")
            print("  3. Test with real users (3-5 people)")
            print("  4. Measure time savings vs manual search")
        else:
            print("  1. Debug why similar videos aren't being found")
            print("  2. Check if V-JEPA embeddings are meaningful")
            print("  3. Try different similarity metrics")
            print("  4. Add more diverse test videos")
        
        return {
            "recall": avg_recall,
            "top1_accuracy": top1_accuracy,
            "mrr": avg_mrr,
            "map": avg_ap
        }


if __name__ == "__main__":
    evaluate_small_dataset()
