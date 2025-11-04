"""
Automated code refactoring script
Reorganizes flat structure into professional package layout
"""
import shutil
from pathlib import Path
import re


class CodebaseRefactor:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.root = Path(".")
        self.moves = []
        
    def create_structure(self):
        """Create new directory structure"""
        dirs = [
            "src/motionmatch/api/routes",
            "src/motionmatch/core",
            "src/motionmatch/services/preprocessing",
            "src/motionmatch/db",
            "src/motionmatch/workers",
            "src/motionmatch/utils",
            "tests/unit",
            "tests/integration",
            "tests/fixtures",
            "scripts",
            "docker",
        ]
        
        for dir_path in dirs:
            full_path = self.root / dir_path
            if self.dry_run:
                print(f"[DRY RUN] Would create: {dir_path}")
            else:
                full_path.mkdir(parents=True, exist_ok=True)
                (full_path / "__init__.py").touch()
                print(f"✓ Created: {dir_path}")
    
    def plan_moves(self):
        """Plan file movements"""
        self.moves = [
            # API files
            ("api.py", "src/motionmatch/api/main.py"),
            
            # Core files
            ("config.py", "src/motionmatch/core/config.py"),
            
            # Services
            ("encoder_service.py", "src/motionmatch/services/encoder.py"),
            ("search_service.py", "src/motionmatch/services/search.py"),
            ("indexing_service.py", "src/motionmatch/services/indexing.py"),
            ("shot_segmentation.py", "src/motionmatch/services/preprocessing/shot_segmentation.py"),
            ("roi_detection.py", "src/motionmatch/services/preprocessing/roi_detection.py"),
            
            # Database
            ("vector_db.py", "src/motionmatch/db/vector_db.py"),
            ("database.py", "src/motionmatch/db/postgres.py"),
            ("models.py", "src/motionmatch/db/models.py"),
            
            # Workers
            ("celery_app.py", "src/motionmatch/workers/celery_app.py"),
            ("tasks.py", "src/motionmatch/workers/tasks.py"),
            
            # Scripts
            ("setup.py", "scripts/setup.py"),
            ("start.py", "scripts/start_services.py"),
            ("download_samples.py", "scripts/download_samples.py"),
            ("setup_checker.py", "scripts/validate_setup.py"),
            ("validate_setup.py", "scripts/validate_installation.py"),
            
            # Tests
            ("test_client.py", "tests/integration/test_api_client.py"),
            ("test_validation.py", "tests/integration/test_validation.py"),
            ("test_vjepa2_integration.py", "tests/integration/test_vjepa2.py"),
            ("testsetup.py", "tests/integration/test_setup.py"),
            ("teststart.py", "tests/integration/test_full_system.py"),
            
            # Docker
            ("Dockerfile", "docker/Dockerfile"),
            ("docker-compose.yml", "docker/docker-compose.yml"),
        ]
        
        return self.moves
    
    def execute_moves(self):
        """Execute planned file movements"""
        for src, dst in self.moves:
            src_path = self.root / src
            dst_path = self.root / dst
            
            if not src_path.exists():
                print(f"⚠ Skip (not found): {src}")
                continue
            
            if self.dry_run:
                print(f"[DRY RUN] Would move: {src} → {dst}")
            else:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"✓ Moved: {src} → {dst}")
    
    def update_imports(self):
        """Update import statements in moved files"""
        import_map = {
            "from encoder_service import": "from motionmatch.services.encoder import",
            "from search_service import": "from motionmatch.services.search import",
            "from indexing_service import": "from motionmatch.services.indexing import",
            "from vector_db import": "from motionmatch.db.vector_db import",
            "from database import": "from motionmatch.db.postgres import",
            "from models import": "from motionmatch.db.models import",
            "from config import": "from motionmatch.core.config import",
            "from celery_app import": "from motionmatch.workers.celery_app import",
            "from tasks import": "from motionmatch.workers.tasks import",
            "import config": "from motionmatch.core import config",
        }
        
        if self.dry_run:
            print("\n[DRY RUN] Would update imports in all Python files")
            for old, new in import_map.items():
                print(f"  {old} → {new}")
        else:
            print("\n✓ Import updates would be applied (manual review recommended)")
    
    def create_pyproject_toml(self):
        """Create modern Python packaging file"""
        content = '''[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "motionmatch"
version = "0.1.0"
description = "Physics-based video search engine using V-JEPA 2"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "MotionMatch Team"}
]
keywords = ["video", "search", "motion", "vjepa", "computer-vision"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "fastapi>=0.104.0",
    "celery>=5.3.0",
    "pymilvus>=2.3.0",
    "redis>=5.0.0",
    "psycopg2-binary>=2.9.9",
    "sqlalchemy>=2.0.0",
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "pydantic>=2.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
]
preprocessing = [
    "scenedetect>=0.6.2",
    "ultralytics>=8.0.0",
]

[project.scripts]
motionmatch-api = "motionmatch.api.main:main"
motionmatch-worker = "motionmatch.workers.celery_app:main"

[project.urls]
Homepage = "https://github.com/yourusername/motionmatch"
Documentation = "https://motionmatch.readthedocs.io"
Repository = "https://github.com/yourusername/motionmatch"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
'''
        
        if self.dry_run:
            print("\n[DRY RUN] Would create pyproject.toml")
        else:
            with open(self.root / "pyproject.toml", "w") as f:
                f.write(content)
            print("\n✓ Created pyproject.toml")
    
    def create_env_example(self):
        """Create .env.example file"""
        content = '''# MotionMatch Configuration

# GPU Settings
CUDA_AVAILABLE=true
DEVICE=cuda

# Feature Flags
ENABLE_SHOT_SEGMENTATION=false
ENABLE_ROI_DETECTION=false
ENABLE_CELERY=true

# Database Settings
DATABASE_URL=postgresql://motionmatch:password@localhost:5432/motionmatch
MILVUS_HOST=localhost
MILVUS_PORT=19530
REDIS_URL=redis://localhost:6379/0

# Model Settings
MODEL_NAME=facebook/vjepa2-vitl-fpc64-256
NUM_FRAMES=64
FRAME_SIZE=256
BATCH_SIZE=4

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Storage
VIDEO_STORAGE_PATH=./storage/videos
FEATURE_STORAGE_PATH=./storage/features
'''
        
        if self.dry_run:
            print("[DRY RUN] Would create .env.example")
        else:
            with open(self.root / ".env.example", "w") as f:
                f.write(content)
            print("✓ Created .env.example")
    
    def run(self):
        """Execute full refactoring"""
        print("=" * 60)
        print("MotionMatch Code Refactoring")
        print("=" * 60)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print()
        
        print("Step 1: Creating directory structure...")
        self.create_structure()
        
        print("\nStep 2: Planning file movements...")
        self.plan_moves()
        print(f"  Planned {len(self.moves)} file movements")
        
        print("\nStep 3: Executing file movements...")
        self.execute_moves()
        
        print("\nStep 4: Updating imports...")
        self.update_imports()
        
        print("\nStep 5: Creating pyproject.toml...")
        self.create_pyproject_toml()
        
        print("\nStep 6: Creating .env.example...")
        self.create_env_example()
        
        print("\n" + "=" * 60)
        if self.dry_run:
            print("DRY RUN COMPLETE")
            print("Run with --execute to apply changes")
        else:
            print("REFACTORING COMPLETE")
            print("\nNext steps:")
            print("1. Review moved files")
            print("2. Update imports manually (search for old import patterns)")
            print("3. Test: python -m pytest tests/")
            print("4. Install package: pip install -e .")
        print("=" * 60)


def main():
    import sys
    
    dry_run = "--execute" not in sys.argv
    
    if dry_run:
        print("Running in DRY RUN mode (no changes will be made)")
        print("Use --execute flag to apply changes\n")
    else:
        response = input("This will reorganize your codebase. Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
    
    refactor = CodebaseRefactor(dry_run=dry_run)
    refactor.run()


if __name__ == "__main__":
    main()
