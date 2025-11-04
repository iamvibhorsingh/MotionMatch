#!/usr/bin/env python3
"""Clear all indexed videos from MotionMatch"""
import os
import sys
import shutil
from pathlib import Path
from pymilvus import connections, utility

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from motionmatch.core.config import config

def clear_all_indexed_videos():
    """Clear all indexed videos from Milvus, PostgreSQL, and storage"""
    
    print("🗑️  Clearing MotionMatch Index")
    print("=" * 50)
    
    # 1. Clear Milvus collection
    try:
        print("\n1. Connecting to Milvus...")
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        
        if utility.has_collection(config.COLLECTION_NAME):
            print(f"   Dropping collection: {config.COLLECTION_NAME}")
            utility.drop_collection(config.COLLECTION_NAME)
            print("   ✓ Milvus collection dropped")
        else:
            print("   ℹ No Milvus collection found")
        
        connections.disconnect("default")
    except Exception as e:
        print(f"   ✗ Milvus error: {e}")
    
    # 2. Clear PostgreSQL database
    try:
        print("\n2. Clearing PostgreSQL database...")
        from motionmatch.db.postgres import SessionLocal, Video, IndexingJob, SearchQuery, SearchClick
        
        db = SessionLocal()
        
        # Count records
        video_count = db.query(Video).count()
        job_count = db.query(IndexingJob).count()
        query_count = db.query(SearchQuery).count()
        click_count = db.query(SearchClick).count()
        
        print(f"   Found: {video_count} videos, {job_count} jobs, {query_count} queries, {click_count} clicks")
        
        # Delete all records
        db.query(SearchClick).delete()
        db.query(SearchQuery).delete()
        db.query(IndexingJob).delete()
        db.query(Video).delete()
        db.commit()
        
        print("   ✓ PostgreSQL database cleared")
        db.close()
    except Exception as e:
        print(f"   ✗ PostgreSQL error: {e}")
    
    # 3. Clear temporal features storage
    try:
        print("\n3. Clearing temporal features...")
        features_dir = os.path.join(config.STORAGE_PATH, "temporal_features")
        
        if os.path.exists(features_dir):
            file_count = len([f for f in os.listdir(features_dir) if f.endswith('.npy')])
            print(f"   Found {file_count} temporal feature files")
            shutil.rmtree(features_dir)
            os.makedirs(features_dir, exist_ok=True)
            print("   ✓ Temporal features cleared")
        else:
            print("   ℹ No temporal features directory found")
    except Exception as e:
        print(f"   ✗ Storage error: {e}")
    
    # 4. Clear temp files
    try:
        print("\n4. Clearing temp files...")
        if os.path.exists(config.TEMP_PATH):
            temp_files = [f for f in os.listdir(config.TEMP_PATH) if os.path.isfile(os.path.join(config.TEMP_PATH, f))]
            for f in temp_files:
                os.remove(os.path.join(config.TEMP_PATH, f))
            print(f"   ✓ Cleared {len(temp_files)} temp files")
        else:
            print("   ℹ No temp directory found")
    except Exception as e:
        print(f"   ✗ Temp cleanup error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Index cleared successfully!")
    print("\nTo re-index videos, run: python testsetup.py")

if __name__ == "__main__":
    import sys
    
    # Confirmation prompt
    response = input("\n⚠️  This will delete ALL indexed videos. Continue? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        clear_all_indexed_videos()
    else:
        print("❌ Cancelled")
        sys.exit(0)
