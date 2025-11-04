"""
Fix imports in refactored codebase
Updates old flat imports to new package structure
"""
from pathlib import Path
import re


def fix_imports_in_file(file_path: Path) -> bool:
    """Fix imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Define import replacements
        replacements = {
            # Services
            r'from encoder_service import': 'from motionmatch.services.encoder import',
            r'from search_service import': 'from motionmatch.services.search import',
            r'from indexing_service import': 'from motionmatch.services.indexing import',
            r'from shot_segmentation import': 'from motionmatch.services.preprocessing.shot_segmentation import',
            r'from roi_detection import': 'from motionmatch.services.preprocessing.roi_detection import',
            
            # Database
            r'from vector_db import': 'from motionmatch.db.vector_db import',
            r'from database import': 'from motionmatch.db.postgres import',
            r'from models import': 'from motionmatch.db.models import',
            
            # Core
            r'from config import': 'from motionmatch.core.config import',
            
            # Workers
            r'from celery_app import': 'from motionmatch.workers.celery_app import',
            r'from tasks import': 'from motionmatch.workers.tasks import',
            
            # Standalone imports
            r'^import config$': 'from motionmatch.core import config',
            r'^import models$': 'from motionmatch.db import models',
        }
        
        # Apply replacements
        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Fix imports in all Python files in src/motionmatch"""
    root = Path("src/motionmatch")
    
    if not root.exists():
        print("Error: src/motionmatch directory not found")
        print("Make sure you've run the refactor script first")
        return
    
    print("=" * 60)
    print("Fixing Imports in Refactored Code")
    print("=" * 60)
    
    # Find all Python files
    python_files = list(root.rglob("*.py"))
    
    print(f"\nFound {len(python_files)} Python files")
    print("\nProcessing files...")
    
    fixed_count = 0
    for file_path in python_files:
        if file_path.name == "__init__.py":
            continue
        
        if fix_imports_in_file(file_path):
            print(f"✓ Fixed: {file_path.relative_to('src/motionmatch')}")
            fixed_count += 1
        else:
            print(f"  Skipped: {file_path.relative_to('src/motionmatch')}")
    
    print(f"\n{'=' * 60}")
    print(f"Fixed imports in {fixed_count} files")
    print("=" * 60)
    
    if fixed_count > 0:
        print("\n✅ Import fixes complete!")
        print("\nNext steps:")
        print("1. Test the imports: python -c 'from motionmatch.api.main import app'")
        print("2. Run diagnostics to check for errors")
        print("3. Update any remaining manual imports if needed")
    else:
        print("\n⚠️  No files were modified")
        print("Imports may already be correct or files not found")


if __name__ == "__main__":
    main()
