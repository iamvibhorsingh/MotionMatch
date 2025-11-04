#!/usr/bin/env python3
"""Test setup script for MotionMatch MVP - initializes system with test videos"""
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import requests
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE = "http://localhost:8000"
TEST_VIDEO_DIR = "testvideo"
SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
MAX_WAIT_TIME = 300  # 5 minutes max wait for services
INDEXING_TIMEOUT = 600  # 10 minutes max for indexing

class TestSetup:
    """Test setup manager for MotionMatch MVP"""
    
    def __init__(self):
        self.test_video_dir = Path(TEST_VIDEO_DIR)
        self.api_base = API_BASE
        self.session = requests.Session()
        self.session.timeout = 30
    
    def create_test_video_directory(self):
        """Create test video directory with instructions"""
        self.test_video_dir.mkdir(exist_ok=True)
        
        readme_content = """# Test Video Directory

This directory contains test videos for MotionMatch MVP validation.

## Recommended Test Videos

For best testing results, include videos with different types of motion:

### Sports & Athletics
- Basketball shots/dunks
- Soccer kicks/goals  
- Tennis serves/swings
- Running/sprinting
- Jumping/leaping

### Dance & Movement
- Ballet spins/leaps
- Hip-hop moves
- Martial arts forms
- Gymnastics routines

### Everyday Actions
- Walking/jogging
- Waving hands
- Throwing objects
- Climbing stairs
- Opening doors

### Action Sequences
- Parkour movements
- Skateboarding tricks
- Swimming strokes
- Cycling

## Video Requirements

- **Duration**: 3-30 seconds (optimal: 5-15 seconds)
- **Format**: MP4, AVI, MOV, WebM, MKV
- **Quality**: 480p minimum, 1080p recommended
- **Content**: Clear motion with minimal camera shake
- **Size**: Under 100MB per video

## File Naming Convention

Use descriptive names that indicate the motion type:
- `basketball_dunk_01.mp4`
- `soccer_kick_penalty.mp4`
- `dance_ballet_spin.mp4`
- `person_jumping_high.mp4`

## Testing Strategy

The test setup will:
1. Index all videos in this directory
2. Test search functionality using each video as a query
3. Validate that similar motions are found
4. Generate a comprehensive test report

## Sample Video Sources

- **Free Stock Videos**: Pexels, Pixabay, Unsplash
- **Creative Commons**: Wikimedia Commons
- **Personal Videos**: Your own recordings (ensure appropriate content)

Place your test videos in this directory and run:
```bash
python testsetup.py
```
"""
        
        readme_path = self.test_video_dir / "README.md"
        if not readme_path.exists():
            readme_path.write_text(readme_content)
            logger.info(f"Created {readme_path}")
    
    def find_test_videos(self) -> List[Path]:
        """Find all test videos in the directory"""
        if not self.test_video_dir.exists():
            logger.warning(f"Test video directory {self.test_video_dir} does not exist")
            return []
        
        videos = []
        for file_path in self.test_video_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                videos.append(file_path)
        
        videos.sort()  # Sort for consistent ordering
        return videos
    
    def wait_for_api(self, max_wait: int = MAX_WAIT_TIME) -> bool:
        """Wait for API to become available"""
        logger.info("Waiting for API to become available...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = self.session.get(f"{self.api_base}/health")
                if response.status_code == 200:
                    health = response.json()
                    if health.get("status") == "healthy":
                        logger.info("✓ API is healthy and ready")
                        return True
                    else:
                        logger.info(f"API responding but not healthy: {health.get('status')}")
                
            except requests.exceptions.RequestException:
                pass
            
            logger.info("Waiting for API... (will retry in 5 seconds)")
            time.sleep(5)
        
        logger.error(f"API did not become available within {max_wait} seconds")
        return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            response = self.session.get(f"{self.api_base}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def clear_existing_index(self) -> bool:
        """Clear existing video index (for clean testing)"""
        logger.info("Checking for existing videos in index...")
        
        try:
            stats = self.get_system_stats()
            video_count = stats.get("total_videos", 0)
            
            if video_count > 0:
                logger.warning(f"Found {video_count} existing videos in index")
                response = input("Clear existing index for clean testing? (y/N): ").strip().lower()
                
                if response == 'y':
                    logger.info("Clearing all videos from index...")
                    clear_response = self.session.delete(f"{self.api_base}/v1/videos")
                    
                    if clear_response.status_code == 200:
                        logger.info("✓ Index cleared successfully")
                        return True
                    else:
                        logger.error(f"Failed to clear index: {clear_response.text}")
                        return False
                else:
                    logger.info("Continuing with existing index")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check/clear existing index: {e}")
            return False
    
    def index_test_videos(self, video_paths: List[Path]) -> Dict[str, Any]:
        """Index all test videos"""
        logger.info(f"Starting indexing of {len(video_paths)} test videos...")
        
        results = {
            "total": len(video_paths),
            "successful": 0,
            "failed": 0,
            "errors": [],
            "indexed_videos": []
        }
        
        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"[{i}/{len(video_paths)}] Indexing: {video_path.name}")
            
            try:
                response = self.session.post(
                    f"{self.api_base}/index/single",
                    params={"video_path": str(video_path.absolute())},
                    timeout=120  # 2 minutes per video
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"  ✓ Success: {result.get('message', 'Indexed successfully')}")
                    results["successful"] += 1
                    results["indexed_videos"].append({
                        "path": str(video_path),
                        "name": video_path.name,
                        "size_mb": video_path.stat().st_size / (1024 * 1024)
                    })
                else:
                    error_msg = response.json().get("detail", f"HTTP {response.status_code}")
                    logger.error(f"  ✗ Failed: {error_msg}")
                    results["failed"] += 1
                    results["errors"].append({
                        "video": video_path.name,
                        "error": error_msg
                    })
                
            except requests.exceptions.Timeout:
                error_msg = "Indexing timeout (>2 minutes)"
                logger.error(f"  ✗ {error_msg}")
                results["failed"] += 1
                results["errors"].append({
                    "video": video_path.name,
                    "error": error_msg
                })
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"  ✗ Error: {error_msg}")
                results["failed"] += 1
                results["errors"].append({
                    "video": video_path.name,
                    "error": error_msg
                })
            
            # Small delay between videos to avoid overwhelming the system
            if i < len(video_paths):
                time.sleep(2)
        
        return results
    
    def test_search_functionality(self, video_paths: List[Path]) -> Dict[str, Any]:
        """Test search functionality using indexed videos as queries"""
        logger.info("Testing search functionality...")
        
        # Use first few videos as test queries (limit to avoid long test times)
        test_queries = video_paths[:min(5, len(video_paths))]
        
        results = {
            "total_queries": len(test_queries),
            "successful_searches": 0,
            "failed_searches": 0,
            "search_results": [],
            "errors": []
        }
        
        for i, query_video in enumerate(test_queries, 1):
            logger.info(f"[{i}/{len(test_queries)}] Testing search with: {query_video.name}")
            
            try:
                with open(query_video, 'rb') as f:
                    files = {'file': f}
                    data = {
                        'top_k': 5,
                        'enable_reranking': False
                    }
                    
                    response = self.session.post(
                        f"{self.api_base}/search/upload",
                        files=files,
                        data=data,
                        timeout=60
                    )
                
                if response.status_code == 200:
                    search_result = response.json()
                    processing_time = search_result.get("processing_time_ms", 0)
                    num_results = len(search_result.get("results", []))
                    
                    logger.info(f"  ✓ Found {num_results} results in {processing_time:.0f}ms")
                    
                    results["successful_searches"] += 1
                    results["search_results"].append({
                        "query_video": query_video.name,
                        "processing_time_ms": processing_time,
                        "num_results": num_results,
                        "top_result": search_result["results"][0] if search_result["results"] else None
                    })
                    
                    # Log top result for verification
                    if search_result["results"]:
                        top_result = search_result["results"][0]
                        similarity = top_result.get("similarity_score", 0) * 100
                        logger.info(f"    Top result: {top_result.get('video_id', 'Unknown')} (similarity: {similarity:.1f}%)")
                
                else:
                    error_msg = response.json().get("detail", f"HTTP {response.status_code}")
                    logger.error(f"  ✗ Search failed: {error_msg}")
                    results["failed_searches"] += 1
                    results["errors"].append({
                        "query_video": query_video.name,
                        "error": error_msg
                    })
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"  ✗ Search error: {error_msg}")
                results["failed_searches"] += 1
                results["errors"].append({
                    "query_video": query_video.name,
                    "error": error_msg
                })
            
            # Small delay between searches
            time.sleep(1)
        
        return results
    
    def generate_test_report(self, indexing_results: Dict, search_results: Dict, video_paths: List[Path]):
        """Generate comprehensive test report"""
        report_path = Path("test_report.json")
        
        # Get final system stats
        final_stats = self.get_system_stats()
        
        report = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_video_directory": str(self.test_video_dir),
            "system_stats": final_stats,
            "test_videos": {
                "total_found": len(video_paths),
                "video_list": [
                    {
                        "name": v.name,
                        "path": str(v),
                        "size_mb": round(v.stat().st_size / (1024 * 1024), 2)
                    } for v in video_paths
                ]
            },
            "indexing_results": indexing_results,
            "search_results": search_results,
            "summary": {
                "indexing_success_rate": (indexing_results["successful"] / indexing_results["total"] * 100) if indexing_results["total"] > 0 else 0,
                "search_success_rate": (search_results["successful_searches"] / search_results["total_queries"] * 100) if search_results["total_queries"] > 0 else 0,
                "avg_search_time_ms": sum(r.get("processing_time_ms", 0) for r in search_results["search_results"]) / len(search_results["search_results"]) if search_results["search_results"] else 0
            }
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 TEST SUMMARY")
        print("="*60)
        print(f"Test Videos Found: {len(video_paths)}")
        print(f"Indexing Success: {indexing_results['successful']}/{indexing_results['total']} ({report['summary']['indexing_success_rate']:.1f}%)")
        print(f"Search Success: {search_results['successful_searches']}/{search_results['total_queries']} ({report['summary']['search_success_rate']:.1f}%)")
        print(f"Average Search Time: {report['summary']['avg_search_time_ms']:.0f}ms")
        print(f"Total Videos in Index: {final_stats.get('total_videos', 0)}")
        
        if indexing_results["errors"]:
            print(f"\n❌ Indexing Errors ({len(indexing_results['errors'])}):")
            for error in indexing_results["errors"][:3]:  # Show first 3
                print(f"  - {error['video']}: {error['error']}")
            if len(indexing_results["errors"]) > 3:
                print(f"  ... and {len(indexing_results['errors']) - 3} more")
        
        if search_results["errors"]:
            print(f"\n❌ Search Errors ({len(search_results['errors'])}):")
            for error in search_results["errors"][:3]:  # Show first 3
                print(f"  - {error['query_video']}: {error['error']}")
        
        print(f"\n📊 Detailed report: {report_path}")
        print("="*60)
    
    def run_full_test_setup(self):
        """Run the complete test setup process"""
        print("🚀 MotionMatch MVP Test Setup")
        print("="*50)
        
        # Step 1: Create test directory and find videos
        self.create_test_video_directory()
        video_paths = self.find_test_videos()
        
        if not video_paths:
            print(f"\n❌ No test videos found in {self.test_video_dir}")
            print("Please add test videos and run again.")
            print("See testvideo/README.md for guidance.")
            return False
        
        print(f"\n📹 Found {len(video_paths)} test videos:")
        for video in video_paths:
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"  - {video.name} ({size_mb:.1f} MB)")
        
        # Step 2: Wait for API
        if not self.wait_for_api():
            print("\n❌ API is not available. Please start the system first:")
            print("   python start.py")
            return False
        
        # Step 3: Check existing index
        if not self.clear_existing_index():
            print("\n⚠ Continuing with existing index state")
        
        # Step 4: Index test videos
        print(f"\n📚 Indexing {len(video_paths)} test videos...")
        indexing_results = self.index_test_videos(video_paths)
        
        if indexing_results["successful"] == 0:
            print("\n❌ No videos were successfully indexed")
            return False
        
        # Step 5: Test search functionality
        print(f"\n🔍 Testing search functionality...")
        search_results = self.test_search_functionality(video_paths)
        
        # Step 6: Generate report
        self.generate_test_report(indexing_results, search_results, video_paths)
        
        # Success check
        success = (
            indexing_results["successful"] > 0 and
            search_results["successful_searches"] > 0
        )
        
        if success:
            print("\n✅ Test setup completed successfully!")
            print("🌐 You can now use the web interface at: http://localhost:8000")
        else:
            print("\n⚠ Test setup completed with issues")
            print("Check the test report for details")
        
        return success

def main():
    """Main function"""
    test_setup = TestSetup()
    
    try:
        success = test_setup.run_full_test_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Test setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()