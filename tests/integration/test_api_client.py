"""Test client for MotionMatch MVP"""
import requests
import json
import time
import os
from typing import List

class MotionMatchClient:
    """Client for testing MotionMatch API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_stats(self):
        """Get system statistics"""
        response = requests.get(f"{self.base_url}/stats")
        return response.json()
    
    def index_single_video(self, video_path: str, skip_if_exists: bool = True):
        """Index a single video"""
        response = requests.post(
            f"{self.base_url}/index/single",
            params={"video_path": video_path, "skip_if_exists": skip_if_exists}
        )
        return response.json()
    
    def index_videos(self, video_paths: List[str], job_id: str = None):
        """Index multiple videos"""
        data = {
            "video_paths": video_paths,
            "job_id": job_id
        }
        response = requests.post(
            f"{self.base_url}/index",
            json=data
        )
        return response.json()
    
    def get_indexing_status(self, job_id: str):
        """Get indexing job status"""
        response = requests.get(f"{self.base_url}/index/status/{job_id}")
        return response.json()
    
    def search_video(self, query_video_path: str, top_k: int = 20, enable_reranking: bool = False):
        """Search for similar videos"""
        data = {
            "query_video_url": query_video_path,  # API expects query_video_url
            "top_k": top_k,
            "options": {
                "enable_reranking": enable_reranking
            }
        }
        response = requests.post(
            f"{self.base_url}/search",
            json=data
        )
        return response.json()
    
    def search_with_upload(self, video_file_path: str, top_k: int = 20, enable_reranking: bool = False):
        """Search with uploaded video file"""
        with open(video_file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'top_k': top_k,
                'enable_reranking': enable_reranking
            }
            response = requests.post(
                f"{self.base_url}/search/upload",
                files=files,
                data=data
            )
        return response.json()
    
    def delete_video(self, video_id: str):
        """Delete a video from index"""
        response = requests.delete(f"{self.base_url}/videos/{video_id}")
        return response.json()
    
    def clear_all_videos(self):
        """Clear all videos from index"""
        response = requests.delete(f"{self.base_url}/v1/videos")
        return response.json()

def demo_workflow():
    """Demonstrate the complete workflow"""
    client = MotionMatchClient()
    
    print("=== MotionMatch MVP Demo ===\n")
    
    # 1. Health check
    print("1. Checking system health...")
    health = client.health_check()
    print(f"Health: {health}\n")
    
    # 2. Get initial stats
    print("2. Getting system stats...")
    stats = client.get_stats()
    print(f"Stats: {stats}\n")
    
    # 3. Index sample videos (you need to provide actual video paths)
    sample_videos = [
        # Add paths to your sample videos here
        # "/path/to/video1.mp4",
        # "/path/to/video2.mp4",
    ]
    
    if sample_videos:
        print("3. Indexing sample videos...")
        index_result = client.index_videos(sample_videos)
        print(f"Index job submitted: {index_result}")
        
        # Monitor indexing progress
        job_id = index_result.get("job_id")
        if job_id:
            while True:
                status = client.get_indexing_status(job_id)
                print(f"Indexing progress: {status['progress_percentage']:.1f}%")
                
                if status["status"] in ["completed", "completed_with_errors", "failed"]:
                    print(f"Indexing finished: {status}")
                    break
                
                time.sleep(2)
        print()
    
    # 4. Search for similar videos
    query_video = input("Enter path to query video (or press Enter to skip): ").strip()
    if query_video and os.path.exists(query_video):
        print("4. Searching for similar videos...")
        search_result = client.search_video(query_video, top_k=10)
        print(f"Search results: {json.dumps(search_result, indent=2)}\n")
    
    # 5. Final stats
    print("5. Final system stats...")
    final_stats = client.get_stats()
    print(f"Final stats: {final_stats}")

if __name__ == "__main__":
    demo_workflow()