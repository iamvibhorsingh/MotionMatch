#!/usr/bin/env python3
"""Download sample videos for MotionMatch testing"""
import os
import sys
import requests
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample video URLs (free/public domain videos)
SAMPLE_VIDEOS = [
    {
        "name": "person_walking.mp4",
        "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
        "description": "Person walking - basic locomotion"
    },
    {
        "name": "basketball_dribble.mp4", 
        "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_2mb.mp4",
        "description": "Basketball dribbling motion"
    },
    {
        "name": "dance_movement.mp4",
        "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_5mb.mp4", 
        "description": "Dance movement sequence"
    }
]

# Note: The above URLs are examples. In practice, you would need actual URLs
# to free/public domain videos or create your own sample videos.

def download_video(url: str, filename: str, target_dir: Path) -> bool:
    """Download a video file"""
    target_path = target_dir / filename
    
    if target_path.exists():
        logger.info(f"✓ {filename} already exists, skipping")
        return True
    
    try:
        logger.info(f"Downloading {filename}...")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  Progress: {progress:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        logger.info(f"✓ Downloaded {filename} ({downloaded / 1024 / 1024:.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download {filename}: {e}")
        if target_path.exists():
            target_path.unlink()  # Remove partial file
        return False

def create_sample_videos():
    """Create simple sample videos using OpenCV (if available)"""
    try:
        import cv2
        import numpy as np
        
        logger.info("Creating synthetic sample videos...")
        
        target_dir = Path("testvideo")
        target_dir.mkdir(exist_ok=True)
        
        # Create a simple moving circle video
        def create_moving_circle_video(filename: str, motion_type: str):
            video_path = target_dir / filename
            
            if video_path.exists():
                logger.info(f"✓ {filename} already exists, skipping")
                return True
            
            # Video properties
            width, height = 640, 480
            fps = 30
            duration = 5  # seconds
            total_frames = fps * duration
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            for frame_num in range(total_frames):
                # Create blank frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Calculate circle position based on motion type
                t = frame_num / total_frames
                
                if motion_type == "horizontal":
                    x = int(50 + (width - 100) * t)
                    y = height // 2
                elif motion_type == "vertical":
                    x = width // 2
                    y = int(50 + (height - 100) * t)
                elif motion_type == "circular":
                    angle = t * 4 * np.pi  # 2 full circles
                    radius = min(width, height) // 4
                    x = int(width // 2 + radius * np.cos(angle))
                    y = int(height // 2 + radius * np.sin(angle))
                else:  # diagonal
                    x = int(50 + (width - 100) * t)
                    y = int(50 + (height - 100) * t)
                
                # Draw circle
                cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
                
                # Add motion type text
                cv2.putText(frame, f"{motion_type.title()} Motion", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(frame)
            
            out.release()
            logger.info(f"✓ Created {filename}")
            return True
        
        # Create different motion types
        samples = [
            ("motion_horizontal.mp4", "horizontal"),
            ("motion_vertical.mp4", "vertical"), 
            ("motion_circular.mp4", "circular"),
            ("motion_diagonal.mp4", "diagonal")
        ]
        
        success_count = 0
        for filename, motion_type in samples:
            if create_moving_circle_video(filename, motion_type):
                success_count += 1
        
        logger.info(f"Created {success_count}/{len(samples)} synthetic videos")
        return success_count > 0
        
    except ImportError:
        logger.warning("OpenCV not available, cannot create synthetic videos")
        return False
    except Exception as e:
        logger.error(f"Failed to create synthetic videos: {e}")
        return False

def main():
    """Main function"""
    print("📥 MotionMatch Sample Video Downloader")
    print("="*50)
    
    target_dir = Path("testvideo")
    target_dir.mkdir(exist_ok=True)
    
    print(f"Target directory: {target_dir.absolute()}")
    
    # Option 1: Try to create synthetic videos
    print("\n🎨 Creating synthetic sample videos...")
    if create_sample_videos():
        print("✅ Synthetic videos created successfully!")
        print("\nYou can now run the test setup:")
        print("  python testsetup.py")
        return
    
    # Option 2: Provide instructions for manual download
    print("\n📋 Manual Sample Video Setup")
    print("-" * 30)
    print("Since automatic download is not available, please manually add test videos:")
    print()
    print("1. Visit free video sites:")
    print("   - Pexels: https://www.pexels.com/videos/")
    print("   - Pixabay: https://pixabay.com/videos/")
    print("   - Unsplash: https://unsplash.com/videos")
    print()
    print("2. Download short videos (5-15 seconds) with clear motion:")
    print("   - Sports activities (basketball, soccer, tennis)")
    print("   - Dance movements")
    print("   - People walking, running, jumping")
    print("   - Object movements")
    print()
    print("3. Save videos to the testvideo/ directory with descriptive names:")
    print("   - basketball_shot.mp4")
    print("   - person_walking.mp4") 
    print("   - dance_spin.mp4")
    print()
    print("4. Run the test setup:")
    print("   python testsetup.py")
    print()
    print("💡 Tip: You need at least 3-5 videos for meaningful testing")

if __name__ == "__main__":
    main()