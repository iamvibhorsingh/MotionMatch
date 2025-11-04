"""
Compare benchmark results across multiple runs
Track performance improvements/regressions over time
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import sys


def load_benchmark_results(file_path: str) -> Dict:
    """Load benchmark results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_metrics(baseline: Dict, current: Dict) -> Dict:
    """Compare two benchmark runs"""
    comparison = {
        "baseline_timestamp": baseline.get("timestamp"),
        "current_timestamp": current.get("timestamp"),
        "improvements": [],
        "regressions": [],
        "unchanged": []
    }
    
    baseline_metrics = baseline.get("metrics", {})
    current_metrics = current.get("metrics", {})
    
    # Compare indexing
    if "indexing" in baseline_metrics and "indexing" in current_metrics:
        b_throughput = baseline_metrics["indexing"]["videos_per_minute"]
        c_throughput = current_metrics["indexing"]["videos_per_minute"]
        diff_pct = ((c_throughput - b_throughput) / b_throughput) * 100
        
        result = {
            "metric": "Indexing Throughput",
            "baseline": f"{b_throughput:.1f} videos/min",
            "current": f"{c_throughput:.1f} videos/min",
            "change": f"{diff_pct:+.1f}%"
        }
        
        if diff_pct > 5:
            comparison["improvements"].append(result)
        elif diff_pct < -5:
            comparison["regressions"].append(result)
        else:
            comparison["unchanged"].append(result)
    
    # Compare search latency
    if "search_latency" in baseline_metrics and "search_latency" in current_metrics:
        b_p99 = baseline_metrics["search_latency"]["p99_ms"]
        c_p99 = current_metrics["search_latency"]["p99_ms"]
        diff_pct = ((c_p99 - b_p99) / b_p99) * 100
        
        result = {
            "metric": "Search Latency (P99)",
            "baseline": f"{b_p99:.0f}ms",
            "current": f"{c_p99:.0f}ms",
            "change": f"{diff_pct:+.1f}%"
        }
        
        if diff_pct < -5:  # Lower is better for latency
            comparison["improvements"].append(result)
        elif diff_pct > 5:
            comparison["regressions"].append(result)
        else:
            comparison["unchanged"].append(result)
    
    # Compare search quality
    if "search_quality" in baseline_metrics and "search_quality" in current_metrics:
        b_precision = baseline_metrics["search_quality"]["precision_at_10"]
        c_precision = current_metrics["search_quality"]["precision_at_10"]
        diff_pct = ((c_precision - b_precision) / b_precision) * 100
        
        result = {
            "metric": "Precision@10",
            "baseline": f"{b_precision:.1%}",
            "current": f"{c_precision:.1%}",
            "change": f"{diff_pct:+.1f}%"
        }
        
        if diff_pct > 5:
            comparison["improvements"].append(result)
        elif diff_pct < -5:
            comparison["regressions"].append(result)
        else:
            comparison["unchanged"].append(result)
    
    return comparison


def print_comparison(comparison: Dict):
    """Print comparison results"""
    print("=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"\nBaseline: {comparison['baseline_timestamp']}")
    print(f"Current:  {comparison['current_timestamp']}")
    
    if comparison["improvements"]:
        print("\n✅ IMPROVEMENTS:")
        for item in comparison["improvements"]:
            print(f"  • {item['metric']}")
            print(f"    {item['baseline']} → {item['current']} ({item['change']})")
    
    if comparison["regressions"]:
        print("\n❌ REGRESSIONS:")
        for item in comparison["regressions"]:
            print(f"  • {item['metric']}")
            print(f"    {item['baseline']} → {item['current']} ({item['change']})")
    
    if comparison["unchanged"]:
        print("\n➡️  UNCHANGED:")
        for item in comparison["unchanged"]:
            print(f"  • {item['metric']}: {item['current']}")
    
    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_runs.py <baseline.json> <current.json>")
        print("\nExample:")
        print("  python compare_runs.py benchmark_baseline.json benchmark_report.json")
        return
    
    baseline_file = sys.argv[1]
    current_file = sys.argv[2]
    
    try:
        baseline = load_benchmark_results(baseline_file)
        current = load_benchmark_results(current_file)
        
        comparison = compare_metrics(baseline, current)
        print_comparison(comparison)
        
        # Save comparison
        output_file = "benchmark_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure both benchmark files exist.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
