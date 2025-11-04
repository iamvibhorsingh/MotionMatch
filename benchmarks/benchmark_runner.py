"""
MotionMatch Benchmark Runner
Evaluates system performance across multiple dimensions
"""
import time
import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import sys

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests" / "integration"))

# Ensure CUDA is detected
try:
    import torch
    if torch.cuda.is_available():
        os.environ["CUDA_AVAILABLE"] = "true"
        print(f"🎮 GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        os.environ["CUDA_AVAILABLE"] = "false"
        print("⚠️  No GPU detected, using CPU")
except:
    os.environ["CUDA_AVAILABLE"] = "false"

from test_api_client import MotionMatchClient


class MotionMatchBenchmark:
    def __init__(self, client: MotionMatchClient, test_videos_dir: str = "testvideo"):
        self.client = client
        self.test_videos_dir = Path(test_videos_dir)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "tests": []
        }
    
    def run_all_benchmarks(self) -> Dict:
        """Run complete benchmark suite"""
        print("=" * 60)
        print("MotionMatch Benchmark Suite")
        print("=" * 60)
        
        # 1. Indexing Performance
        print("\n[1/5] Testing Indexing Performance...")
        self.benchmark_indexing()
        
        # 2. Search Latency
        print("\n[2/5] Testing Search Latency...")
        self.benchmark_search_latency()
        
        # 3. Search Quality
        print("\n[3/5] Testing Search Quality...")
        self.benchmark_search_quality()
        
        # 4. Throughput
        print("\n[4/5] Testing Throughput...")
        self.benchmark_throughput()
        
        # 5. System Resources
        print("\n[5/5] Testing System Resources...")
        self.benchmark_resources()
        
        # Generate report
        self.generate_report()
        
        return self.results
    
    def benchmark_indexing(self):
        """Measure indexing performance"""
        # Index ALL videos for proper search quality testing
        videos = list(self.test_videos_dir.glob("*.mp4"))
        
        print(f"  Indexing {len(videos)} videos...")
        
        times = []
        skipped = 0
        for video in videos:
            start = time.time()
            try:
                result = self.client.index_single_video(str(video))
                elapsed = time.time() - start
                
                # Check if video was skipped (very fast = already indexed)
                if elapsed < 1.0:
                    print(f"  ⊘ Skipped {video.name} (already indexed)")
                    skipped += 1
                else:
                    times.append(elapsed)
                    print(f"  ✓ Indexed {video.name}: {elapsed:.2f}s")
            except Exception as e:
                print(f"  ✗ Failed {video.name}: {e}")
        
        if times:
            self.results["metrics"]["indexing"] = {
                "avg_time_seconds": np.mean(times),
                "min_time_seconds": np.min(times),
                "max_time_seconds": np.max(times),
                "videos_per_minute": 60 / np.mean(times) if times else 0,
                "total_videos": len(times)
            }
            print(f"\n  Average: {np.mean(times):.2f}s per video")
            print(f"  Throughput: {60/np.mean(times):.1f} videos/minute")
    
    def benchmark_search_latency(self):
        """Measure search response times"""
        videos = list(self.test_videos_dir.glob("*.mp4"))[:5]  # Sample 5 for speed
        
        print(f"  Testing with {len(videos)} queries...")
        
        latencies = []
        for i, video in enumerate(videos, 1):
            start = time.time()
            try:
                results = self.client.search_video(str(video), top_k=10)
                elapsed = (time.time() - start) * 1000  # Convert to ms
                latencies.append(elapsed)
                print(f"  [{i}/{len(videos)}] {video.name}: {elapsed:.0f}ms")
            except Exception as e:
                print(f"  ✗ Failed {video.name}: {e}")
        
        if latencies:
            self.results["metrics"]["search_latency"] = {
                "p50_ms": np.percentile(latencies, 50),
                "p90_ms": np.percentile(latencies, 90),
                "p99_ms": np.percentile(latencies, 99),
                "avg_ms": np.mean(latencies),
                "min_ms": np.min(latencies),
                "max_ms": np.max(latencies),
                "num_queries": len(latencies)
            }
            print(f"\n  Average: {np.mean(latencies):.0f}ms")
            print(f"  P50: {np.percentile(latencies, 50):.0f}ms")
            print(f"  P99: {np.percentile(latencies, 99):.0f}ms")
            
            # Warn if latency is very high
            if np.mean(latencies) > 10000:
                print(f"  ⚠️  High latency detected - possible cold start or performance issue")
    
    def benchmark_search_quality(self):
        """Measure search result quality using ground truth"""
        # Define ground truth: videos that should match
        ground_truth = {
            "jump1.mp4": ["jump2.mp4", "jump3.mp4", "jump5.mp4"],
            "run1.mp4": ["run2.mp4", "run3.mp4"],
            "climb1.mp4": ["climb2.mp4"],
            "surf1.mp4": ["surf2.mp4"]
        }
        
        precision_at_k = {5: [], 10: []}
        recall_at_k = {5: [], 10: []}
        
        for query_video, expected_matches in ground_truth.items():
            query_path = self.test_videos_dir / query_video
            if not query_path.exists():
                continue
            
            try:
                results = self.client.search_video(str(query_path), top_k=10)
                
                # Extract result video names
                result_names = []
                for r in results.get("results", []):
                    video_path = Path(r.get("video_path", ""))
                    result_names.append(video_path.name)
                
                # Calculate precision@k and recall@k
                for k in [5, 10]:
                    top_k_results = result_names[:k]
                    relevant_found = sum(1 for name in top_k_results if name in expected_matches)
                    
                    # Precision: what fraction of retrieved items are relevant
                    precision = relevant_found / k if k > 0 else 0
                    precision_at_k[k].append(precision)
                    
                    # Recall: what fraction of relevant items were retrieved
                    recall = relevant_found / len(expected_matches) if expected_matches else 0
                    recall_at_k[k].append(recall)
                
                relevant_found_10 = sum(1 for name in result_names[:10] if name in expected_matches)
                print(f"  ✓ {query_video}: Found {relevant_found_10}/{len(expected_matches)} matches in top-10 (Recall: {relevant_found_10/len(expected_matches):.0%})")
                
            except Exception as e:
                print(f"  ✗ Failed {query_video}: {e}")
        
        if precision_at_k[10]:
            self.results["metrics"]["search_quality"] = {
                "precision_at_5": np.mean(precision_at_k[5]),
                "precision_at_10": np.mean(precision_at_k[10]),
                "recall_at_5": np.mean(recall_at_k[5]),
                "recall_at_10": np.mean(recall_at_k[10]),
                "num_queries": len(precision_at_k[10])
            }
            print(f"\n  Precision@5: {np.mean(precision_at_k[5]):.1%}")
            print(f"  Precision@10: {np.mean(precision_at_k[10]):.1%}")
            print(f"  Recall@5: {np.mean(recall_at_k[5]):.1%}")
            print(f"  Recall@10: {np.mean(recall_at_k[10]):.1%}")
    
    def benchmark_throughput(self):
        """Measure concurrent query throughput"""
        import concurrent.futures
        
        videos = list(self.test_videos_dir.glob("*.mp4"))[:10]
        
        # Sequential baseline
        start = time.time()
        for video in videos[:5]:
            try:
                self.client.search_video(str(video), top_k=5)
            except:
                pass
        sequential_time = time.time() - start
        
        # Concurrent (simulated)
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(self.client.search_video, str(v), 5) for v in videos[:5]]
            concurrent.futures.wait(futures)
        concurrent_time = time.time() - start
        
        self.results["metrics"]["throughput"] = {
            "sequential_qps": 5 / sequential_time if sequential_time > 0 else 0,
            "concurrent_qps": 5 / concurrent_time if concurrent_time > 0 else 0,
            "speedup": sequential_time / concurrent_time if concurrent_time > 0 else 0
        }
        
        print(f"  Sequential: {5/sequential_time:.1f} QPS")
        print(f"  Concurrent: {5/concurrent_time:.1f} QPS")
    
    def benchmark_resources(self):
        """Check system resource usage"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_available": True,
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
                    "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
                }
                print(f"  GPU: {gpu_info['gpu_name']}")
                print(f"  VRAM Used: {gpu_info['gpu_memory_allocated_gb']:.1f}GB / {gpu_info['gpu_memory_total_gb']:.1f}GB")
            else:
                gpu_info = {"gpu_available": False}
                print("  GPU: Not available (CPU mode)")
        except:
            gpu_info = {"gpu_available": False, "error": "torch not available"}
        
        self.results["metrics"]["resources"] = gpu_info
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        report_path = Path("benchmark_report.json")
        
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        metrics = self.results["metrics"]
        
        # Indexing
        if "indexing" in metrics:
            idx = metrics["indexing"]
            print(f"\n📊 Indexing Performance:")
            print(f"  • Average Time: {idx['avg_time_seconds']:.2f}s per video")
            print(f"  • Throughput: {idx['videos_per_minute']:.1f} videos/minute")
            
            # Compare to targets
            target_throughput = 10  # From docs: minimum 10 videos/min
            status = "✓" if idx['videos_per_minute'] >= target_throughput else "✗"
            print(f"  {status} Target: ≥{target_throughput} videos/min")
        
        # Search Latency
        if "search_latency" in metrics:
            lat = metrics["search_latency"]
            print(f"\n⚡ Search Latency:")
            print(f"  • P50: {lat['p50_ms']:.0f}ms")
            print(f"  • P99: {lat['p99_ms']:.0f}ms")
            
            # Compare to targets (from docs: <2s p99)
            target_p99 = 2000
            status = "✓" if lat['p99_ms'] < target_p99 else "✗"
            print(f"  {status} Target: <{target_p99}ms (p99)")
        
        # Search Quality
        if "search_quality" in metrics:
            qual = metrics["search_quality"]
            print(f"\n🎯 Search Quality:")
            print(f"  • Precision@10: {qual['precision_at_10']:.1%} (relevant items in top-10)")
            print(f"  • Recall@10: {qual['recall_at_10']:.1%} (% of relevant items found)")
            
            # Compare to targets (from docs: 70% recall@10 is more meaningful)
            target_recall = 0.70
            status = "✓" if qual['recall_at_10'] >= target_recall else "✗"
            print(f"  {status} Target: ≥{target_recall:.0%} recall")
        
        # Throughput
        if "throughput" in metrics:
            thr = metrics["throughput"]
            print(f"\n🚀 Throughput:")
            print(f"  • Concurrent QPS: {thr['concurrent_qps']:.1f}")
            
            # Compare to targets (from docs: 100 QPS)
            target_qps = 100
            status = "✓" if thr['concurrent_qps'] >= target_qps else "⚠"
            print(f"  {status} Target: ≥{target_qps} QPS")
        
        print(f"\n📄 Full report saved to: {report_path}")
        print("=" * 60)


def main():
    """Run benchmark suite"""
    client = MotionMatchClient()
    
    # Check if system is ready
    try:
        health = client.health_check()
        if health.get("status") != "healthy":
            print("⚠ System not healthy. Please start services first.")
            return
    except:
        print("✗ Cannot connect to API. Please run: python start.py")
        return
    
    # Run benchmarks
    benchmark = MotionMatchBenchmark(client)
    results = benchmark.run_all_benchmarks()
    
    return results


if __name__ == "__main__":
    main()
