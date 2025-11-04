# MotionMatch Quick Start

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode (optional)
pip install -e .
```

## Starting the System

### Option 1: Simple Start (Recommended)
```bash
python start.py
```

### Option 2: Using Docker
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Option 3: Direct Module Run
```bash
python -m uvicorn motionmatch.api.main:app --host 0.0.0.0 --port 8000
```

## Running Benchmarks

```bash
python benchmarks/benchmark_runner.py
```

## Clearing Index

```bash
python clear_index.py
```

## Troubleshooting

### "No module named 'motionmatch'"
Install the package: `pip install -e .`

### Services Not Running
Make sure Docker services are up:
```bash
docker-compose -f docker/docker-compose.yml up -d
```

Wait 30-60 seconds for services to initialize.
