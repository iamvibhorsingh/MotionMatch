"""Setup requirements checker for MotionMatch features"""
import logging
import os
import subprocess
import sys
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class SetupChecker:
    """Check and setup requirements for MotionMatch features"""
    
    def __init__(self):
        self.requirements = {
            "celery": self._check_celery,
            "postgresql": self._check_postgresql,
            "shot_segmentation": self._check_shot_segmentation,
            "roi_detection": self._check_roi_detection,
            "ffmpeg": self._check_ffmpeg
        }
    
    def check_all_requirements(self) -> Dict[str, Tuple[bool, str]]:
        """Check all feature requirements"""
        results = {}
        
        for feature, check_func in self.requirements.items():
            try:
                success, message = check_func()
                results[feature] = (success, message)
            except Exception as e:
                results[feature] = (False, f"Check failed: {e}")
        
        return results
    
    def _check_celery(self) -> Tuple[bool, str]:
        """Check Celery setup"""
        try:
            import celery
            import redis
            
            # Test Redis connection
            r = redis.Redis.from_url(os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"))
            r.ping()
            
            return True, f"Celery {celery.__version__} with Redis broker available"
        except ImportError:
            return False, "Celery or Redis not installed"
        except Exception as e:
            return False, f"Redis broker not available: {e}"
    
    def _check_postgresql(self) -> Tuple[bool, str]:
        """Check PostgreSQL setup"""
        try:
            import psycopg2
            from sqlalchemy import create_engine, text
            
            # Test database connection
            db_url = os.getenv("DATABASE_URL", "postgresql://motionmatch:password@localhost:5432/motionmatch")
            engine = create_engine(db_url)
            
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return True, "PostgreSQL connection successful"
        except ImportError:
            return False, "psycopg2 not installed"
        except Exception as e:
            return False, f"PostgreSQL connection failed: {e}"
    
    def _check_shot_segmentation(self) -> Tuple[bool, str]:
        """Check shot segmentation requirements"""
        try:
            import scenedetect
            
            # Check if ffmpeg is available (required by scenedetect)
            ffmpeg_ok, ffmpeg_msg = self._check_ffmpeg()
            if not ffmpeg_ok:
                return False, f"FFmpeg required for shot segmentation: {ffmpeg_msg}"
            
            return True, f"PySceneDetect {scenedetect.__version__} with FFmpeg available"
        except ImportError:
            return False, "PySceneDetect not installed"
    
    def _check_roi_detection(self) -> Tuple[bool, str]:
        """Check ROI detection requirements"""
        try:
            import ultralytics
            import supervision
            
            # Test model loading (will download if needed)
            from ultralytics import YOLO
            
            # Use smallest model for testing
            model = YOLO('yolov8n.pt')
            
            return True, f"YOLOv8 (ultralytics {ultralytics.__version__}) and supervision available"
        except ImportError as e:
            return False, f"ROI detection dependencies not installed: {e}"
        except Exception as e:
            return False, f"YOLOv8 model loading failed: {e}"
    
    def _check_ffmpeg(self) -> Tuple[bool, str]:
        """Check FFmpeg availability"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                return True, f"FFmpeg available: {version_line}"
            else:
                return False, "FFmpeg command failed"
        except FileNotFoundError:
            return False, "FFmpeg not found in PATH"
        except subprocess.TimeoutExpired:
            return False, "FFmpeg check timed out"
        except Exception as e:
            return False, f"FFmpeg check error: {e}"
    
    def print_setup_report(self):
        """Print comprehensive setup report"""
        print("🔍 MotionMatch Setup Requirements Check")
        print("=" * 50)
        
        results = self.check_all_requirements()
        
        # Core features
        core_features = ["celery", "postgresql"]
        optional_features = ["shot_segmentation", "roi_detection", "ffmpeg"]
        
        print("\n📋 Core Features:")
        for feature in core_features:
            success, message = results.get(feature, (False, "Not checked"))
            status = "✅" if success else "❌"
            print(f"  {status} {feature.replace('_', ' ').title()}: {message}")
        
        print("\n🎯 Optional Features:")
        for feature in optional_features:
            success, message = results.get(feature, (False, "Not checked"))
            status = "✅" if success else "⚠️"
            print(f"  {status} {feature.replace('_', ' ').title()}: {message}")
        
        # Installation instructions
        failed_features = [f for f, (success, _) in results.items() if not success]
        
        if failed_features:
            print(f"\n🔧 Setup Instructions:")
            
            if "celery" in failed_features:
                print("  Celery setup:")
                print("    - Start Redis: docker run -d -p 6379:6379 redis:alpine")
                print("    - Or use docker-compose up redis")
            
            if "postgresql" in failed_features:
                print("  PostgreSQL setup:")
                print("    - Start PostgreSQL: docker-compose up postgres")
                print("    - Or set DATABASE_URL environment variable")
            
            if "ffmpeg" in failed_features:
                print("  FFmpeg installation:")
                print("    - Windows: Download from https://ffmpeg.org/download.html")
                print("    - macOS: brew install ffmpeg")
                print("    - Linux: apt-get install ffmpeg")
            
            if "shot_segmentation" in failed_features:
                print("  Shot segmentation:")
                print("    - Install: pip install scenedetect")
                print("    - Requires FFmpeg (see above)")
            
            if "roi_detection" in failed_features:
                print("  ROI detection:")
                print("    - Install: pip install ultralytics supervision")
                print("    - Models will auto-download on first use")
        
        # Summary
        total_features = len(results)
        successful_features = sum(1 for success, _ in results.values() if success)
        
        print(f"\n📊 Summary: {successful_features}/{total_features} features available")
        
        if successful_features == total_features:
            print("🎉 All features ready!")
        elif successful_features >= len(core_features):
            print("✅ Core features ready, optional features can be enabled later")
        else:
            print("⚠️ Some core features need setup")

def main():
    """Run setup check"""
    checker = SetupChecker()
    checker.print_setup_report()

if __name__ == "__main__":
    main()