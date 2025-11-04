#!/usr/bin/env python3
"""Example usage of MotionMatch MVP"""
import os
import time
import requests
import json
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000"
SAMPLE_VIDEOS_DIR = "sample_videos"

def create_sample_videos_dir():
    """Create sample videos directory"""
    Path(SAMPLE_VIDEOS_DIR).mkdir(exist_ok=True)
    
    readme_content = """# Sample Videos Directory

Place your sample video files here for testing MotionMatch.

Supported formats:
- MP4 (recommended)
- AVI
- MOV
- WebM

Example files you can add:
- sports_basketball_dunk.mp4
- sports_soccer_kick.mp4
- dance_ballet_spin.mp4
- action_person_jumping.mp4

The system works best with:
- Videos 5-30 seconds long
- Clear motion/action content
- Good video quality (720p+)
- Stable camera (not too shaky)

To test the system:
1. Add 3-5 sample videos to this directory
2. Run: python example.py
"""
    
    readme_path = Path(SAMPLE_VIDEOS_DIR) / "README.md"
    if not readme_path.exists():
        readme_path.write_text(readme_content)

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        health = response.json()
        print(f"✓ API Health: {health['status']}")
        print(f"  Model loaded: {health['model_loaded']}")
        print(f"  Device: {health['device']}")
        if health.get('gpu_memory_mb'):
            print(f"  GPU Memory: {health['gpu_memory_mb']:.1f} MB")
        return health['status'] == 'healthy'
    except Exception as e:
        print(f"✗ API not available: {e}")
        return False

def get_system_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE}/stats")
        stats = response.json()
        print(f"📊 System Stats:")
        print(f"  Total videos indexed: {stats['total_videos']}")
        print(f"  Model: {stats['model_name']}")
        print(f"  Vector dimension: {stats['vector_dim']}")
        return stats
    except Exception as e:
        print(f"✗ Failed to get stats: {e}")
        return None

def find_sample_videos():
    """Find sample videos in the directory"""
    video_extensions = {'.mp4', '.avi', '.mov', '.webm'}
    sample_dir = Path(SAMPLE_VIDEOS_DIR)
    
    videos = []
    if sample_dir.exists():
        for file_path in sample_dir.iterdir():
            if file_path.suffix.lower() in video_extensions:
                videos.append(str(file_path))
    
    return videos

def index_sample_videos(video_paths):
    """Index sample videos"""
    print(f"\n📚 Indexing {len(video_paths)} videos...")
    
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"  [{i}/{len(video_paths)}] Indexing: {Path(video_path).name}")
        
        try:
            response = requests.post(
                f"{API_BASE}/index/single",
                params={"video_path": video_path},
                timeout=60
            )
            
            if response.status_code == 200:
                print(f"    ✓ Success")
                successful += 1
            else:
                print(f"    ✗ Failed: {response.json().get('detail', 'Unknown error')}")
                failed += 1
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            failed += 1
    
    print(f"\n📈 Indexing Results: {successful} successful, {failed} failed")
    return successful > 0

def search_similar_videos(query_video_path):
    """Search for similar videos"""
    print(f"\n🔍 Searching for videos similar to: {Path(query_video_path).name}")
    
    try:
        with open(query_video_path, 'rb') as f:
            files = {'file': f}
            data = {
                'top_k': 5,
                'enable_reranking': False
            }
            
            response = requests.post(
                f"{API_BASE}/search/upload",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Processing time: {result['processing_time_ms']:.0f}ms")
            print(f"  Found {len(result['results'])} similar videos:")
            
            for i, item in enumerate(result['results'], 1):
                score = item['similarity_score'] * 100
                print(f"    {i}. {item['video_id']} (similarity: {score:.1f}%)")
                print(f"       Distance: {item['distance']:.3f}")
                if 'duration' in item['metadata']:
                    print(f"       Duration: {item['metadata']['duration']:.1f}s")
            
            return result
        else:
            print(f"  ✗ Search failed: {response.json().get('detail', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"  ✗ Search error: {e}")
        return None

def run_demo():
    """Run the complete demo"""
    print("🎬 MotionMatch MVP Demo")
    print("=" * 50)
    
    # Create sample videos directory
    create_sample_videos_dir()
    
    # Check API health
    if not check_api_health():
        print("\n💡 Make sure the API server is running:")
        print("   python start.py")
        return
    
    # Get initial stats
    get_system_stats()
    
    # Find sample videos
    sample_videos = find_sample_videos()
    
    if not sample_videos:
        print(f"\n📁 No sample videos found in '{SAMPLE_VIDEOS_DIR}' directory")
        print("   Please add some sample videos and run the demo again")
        print("   See sample_videos/README.md for guidance")
        return
    
    print(f"\n📹 Found {len(sample_videos)} sample videos:")
    for video in sample_videos:
        print(f"  - {Path(video).name}")
    
    # Index videos
    if index_sample_videos(sample_videos):
        # Wait a moment for indexing to complete
        time.sleep(2)
        
        # Get updated stats
        print("\n📊 Updated stats after indexing:")
        get_system_stats()
        
        # Perform search with first video as query
        query_video = sample_videos[0]
        search_result = search_similar_videos(query_video)
        
        if search_result and len(search_result['results']) > 1:
            print("\n🎉 Demo completed successfully!")
            print("   The system found similar videos based on motion patterns")
        else:
            print("\n⚠ Demo completed but no similar videos found")
            print("   Try adding more diverse sample videos")
    
    print("\n🌐 You can also use the web interface at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")

if __name__ == "__main__":
    run_demo()