"""
Advanced evaluation metrics for motion-based video search
"""
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path


class SearchEvaluator:
    """Evaluate search quality using various metrics"""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Precision@K"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for item in retrieved_k if item in relevant)
        return relevant_retrieved / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Recall@K"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for item in retrieved_k if item in relevant)
        return relevant_retrieved / len(relevant) if relevant else 0.0
    
    @staticmethod
    def average_precision(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Average Precision (AP)"""
        if not relevant:
            return 0.0
        
        score = 0.0
        num_relevant = 0
        
        for i, item in enumerate(retrieved):
            if item in relevant:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                score += precision_at_i
        
        return score / len(relevant)
    
    @staticmethod
    def mean_average_precision(results: List[Tuple[List[str], List[str]]]) -> float:
        """Calculate Mean Average Precision (MAP)"""
        aps = [SearchEvaluator.average_precision(ret, rel) for ret, rel in results]
        return np.mean(aps) if aps else 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K"""
        retrieved_k = retrieved[:k]
        
        # DCG
        dcg = 0.0
        for i, item in enumerate(retrieved_k):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0
        
        # IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(results: List[Tuple[List[str], List[str]]]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        reciprocal_ranks = []
        
        for retrieved, relevant in results:
            for i, item in enumerate(retrieved):
                if item in relevant:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


class PerformanceEvaluator:
    """Evaluate system performance metrics"""
    
    @staticmethod
    def calculate_percentiles(values: List[float], percentiles: List[int] = [50, 90, 95, 99]) -> Dict[str, float]:
        """Calculate percentile statistics"""
        if not values:
            return {f"p{p}": 0.0 for p in percentiles}
        
        return {f"p{p}": np.percentile(values, p) for p in percentiles}
    
    @staticmethod
    def throughput_metrics(num_requests: int, total_time: float) -> Dict[str, float]:
        """Calculate throughput metrics"""
        return {
            "requests_per_second": num_requests / total_time if total_time > 0 else 0,
            "avg_time_per_request": total_time / num_requests if num_requests > 0 else 0,
            "total_requests": num_requests,
            "total_time_seconds": total_time
        }
    
    @staticmethod
    def compare_to_baseline(current: float, baseline: float, higher_is_better: bool = True) -> Dict:
        """Compare current metric to baseline"""
        if baseline == 0:
            return {"improvement": 0, "percentage": 0, "status": "no_baseline"}
        
        improvement = current - baseline
        percentage = (improvement / baseline) * 100
        
        if higher_is_better:
            status = "better" if improvement > 0 else "worse"
        else:
            status = "better" if improvement < 0 else "worse"
        
        return {
            "current": current,
            "baseline": baseline,
            "improvement": improvement,
            "percentage": percentage,
            "status": status
        }


def generate_test_ground_truth() -> Dict[str, List[str]]:
    """
    Generate ground truth for test videos
    Videos with similar motion should match
    """
    return {
        # Jumping motions
        "jump1.mp4": ["jump2.mp4", "jump3.mp4", "jump5.mp4"],
        "jump2.mp4": ["jump1.mp4", "jump3.mp4", "jump5.mp4"],
        "jump3.mp4": ["jump1.mp4", "jump2.mp4", "jump5.mp4"],
        "jump5.mp4": ["jump1.mp4", "jump2.mp4", "jump3.mp4"],
        
        # Running motions
        "run1.mp4": ["run2.mp4", "run3.mp4"],
        "run2.mp4": ["run1.mp4", "run3.mp4"],
        "run3.mp4": ["run1.mp4", "run2.mp4"],
        
        # Climbing motions
        "climb1.mp4": ["climb2.mp4"],
        "climb2.mp4": ["climb1.mp4"],
        
        # Surfing motions
        "surf1.mp4": ["surf2.mp4"],
        "surf2.mp4": ["surf1.mp4"],
    }
