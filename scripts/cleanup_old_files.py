"""
Cleanup script to remove old files after refactoring
Only removes files that have been successfully moved to new structure
"""
from pathlib import Path
import os


class OldFileCleanup:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.root = Path(".")
        
        # Files that should be removed from root after refactoring
        self.files_to_remove = [
            # Core service files (moved to src/motionmatch/)
            "api.py",
            "encoder_service.py",
            "search_service.py",
            "indexing_service.py",
            "vector_db.py",
            "database.py",
            "models.py",
            "config.py",
            "celery_app.py",
            "tasks.py",
            "shot_segmentation.py",
            "roi_detection.py",
            
            # Test files (moved to tests/)
            "test_client.py",
            "test_validation.py",
            "test_vjepa2_integration.py",
            "testsetup.py",
            "teststart.py",
            
            # Setup files (moved to scripts/)
            "setup.py",
            "start.py",
            "download_samples.py",
            "setup_checker.py",
            "validate_setup.py",
            "setup_features.py",
            
            # Docker files (moved to docker/)
            "Dockerfile",
            "docker-compose.yml",
        ]
        
        # Files to keep in root
        self.keep_files = [
            "README.md",
            "requirements.txt",
            ".env",
            ".env.example",
            ".gitignore",
            "pyproject.toml",
            "IMPROVEMENTS.md",
            "REFACTOR_PLAN.md",
            "QUICK_START_BENCHMARKS.md",
            # Config files
            "pg_hba_scram.conf",
            "pg_hba_simple.conf",
        ]
    
    def verify_new_files_exist(self) -> bool:
        """Verify that new structure has the files before deleting old ones"""
        critical_paths = [
            "src/motionmatch/api/main.py",
            "src/motionmatch/services/encoder.py",
            "src/motionmatch/db/vector_db.py",
            "scripts/start_services.py",
        ]
        
        missing = []
        for path in critical_paths:
            if not (self.root / path).exists():
                missing.append(path)
        
        if missing:
            print("⚠️  WARNING: New structure incomplete!")
            print("Missing files:")
            for m in missing:
                print(f"  - {m}")
            return False
        
        return True
    
    def cleanup(self):
        """Remove old files from root directory"""
        print("=" * 60)
        print("Old File Cleanup")
        print("=" * 60)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}\n")
        
        # Safety check
        if not self.dry_run:
            if not self.verify_new_files_exist():
                print("\n❌ Aborting: New structure not complete")
                print("Run the refactor script first: python scripts/refactor_codebase.py --execute")
                return
        
        removed_count = 0
        skipped_count = 0
        
        for filename in self.files_to_remove:
            file_path = self.root / filename
            
            if not file_path.exists():
                continue
            
            if self.dry_run:
                print(f"[DRY RUN] Would remove: {filename}")
                removed_count += 1
            else:
                try:
                    os.remove(file_path)
                    print(f"✓ Removed: {filename}")
                    removed_count += 1
                except Exception as e:
                    print(f"✗ Failed to remove {filename}: {e}")
                    skipped_count += 1
        
        print(f"\n{'Would remove' if self.dry_run else 'Removed'}: {removed_count} files")
        if skipped_count > 0:
            print(f"Skipped: {skipped_count} files")
        
        # Show what will remain
        print("\n📁 Files remaining in root:")
        root_files = [f for f in os.listdir(self.root) if os.path.isfile(self.root / f)]
        remaining = [f for f in root_files if f not in self.files_to_remove or f in self.keep_files]
        for f in sorted(remaining)[:10]:  # Show first 10
            print(f"  • {f}")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")
        
        print("\n" + "=" * 60)
        if self.dry_run:
            print("DRY RUN COMPLETE")
            print("Run with --execute to remove files")
        else:
            print("CLEANUP COMPLETE")
            print("\nYour root directory is now clean!")
            print("All code is organized in:")
            print("  • src/motionmatch/  - Main package")
            print("  • tests/            - Test suite")
            print("  • scripts/          - Utility scripts")
            print("  • benchmarks/       - Performance tests")
        print("=" * 60)


def main():
    import sys
    
    dry_run = "--execute" not in sys.argv
    
    if dry_run:
        print("Running in DRY RUN mode (no files will be deleted)")
        print("Use --execute flag to actually remove files\n")
    else:
        print("⚠️  WARNING: This will DELETE old files from root directory")
        print("Make sure you've run the refactor script first!\n")
        response = input("Continue with cleanup? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
    
    cleanup = OldFileCleanup(dry_run=dry_run)
    cleanup.cleanup()


if __name__ == "__main__":
    main()
