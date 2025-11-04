# MotionMatch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)

A physics-based video search engine using Meta's V-JEPA 2 world model to find videos with similar motion dynamics.

> **Note**: This project explores **zero-shot video retrieval** using V-JEPA 2 embeddings directly for similarity search, without any fine-tuning or training. This approach provides a baseline for motion-based search but has limitations in cross-domain scenarios. For production use, fine-tuning on domain-specific data would significantly improve results.

## Features

- **Motion-Aware Search**: Find videos by motion patterns, not just visual similarity
- **V-JEPA 2 Integration**: Uses Meta's state-of-the-art video world model
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **REST API**: Production-ready HTTP API with documentation
- **Web Interface**: Drag-and-drop video search interface

## Tech Stack

- **ML/AI**: V-JEPA 2 (Meta), PyTorch, Transformers
- **Backend**: FastAPI, Celery, Python 3.11+
- **Databases**: Milvus (vector), PostgreSQL (metadata), Redis (cache)
- **Infrastructure**: Docker, CUDA

## Quick Start

### Prerequisites
- Python 3.11+
- Docker
- NVIDIA GPU (optional, for acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/iamvibhorsingh/motionmatch.git
cd motionmatch

# Install dependencies
pip install -r requirements.txt

# Start infrastructure services
docker-compose -f docker/docker-compose.yml up -d

# Start the application
python start.py
```

### Usage

Access the web interface at `http://localhost:8000` or use the REST API:

```bash
# Index a video
curl -X POST "http://localhost:8000/v1/index/single?video_path=/path/to/video.mp4"

# Search for similar videos
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query_video_url": "/path/to/query.mp4", "top_k": 10}'
```

## Performance

### System Performance
- **GPU**: 2-3 seconds per video encoding
- **CPU**: 15-30 seconds per video encoding  
- **Search**: <1 second for 100K videos
- **Throughput**: 15-25 videos/minute (GPU)

### Search Quality (Zero-Shot)
This project uses **zero-shot retrieval** with pre-trained V-JEPA 2 embeddings:
- ✅ **Good**: Same-domain video search (same dataset/source)
- ⚠️ **Limited**: Cross-domain retrieval (different sources/contexts)
- ❌ **Poor**: Fine-grained action discrimination

**Why?** V-JEPA 2 was trained for masked video prediction, not similarity learning. The embeddings capture motion patterns but aren't optimized for retrieval tasks. Fine-tuning with metric learning (triplet loss, contrastive learning) would significantly improve results.

## Configuration

Environment variables (create `.env` file):
```env
CUDA_AVAILABLE=true
DATABASE_URL=postgresql://motionmatch:password@localhost:5432/motionmatch
MILVUS_HOST=localhost
BATCH_SIZE=4
```

## Troubleshooting

**Docker issues**: `docker-compose -f docker/docker-compose.yml ps`  
**GPU not detected**: Check `nvidia-smi` and PyTorch CUDA  
**Slow performance**: Reduce batch size or use SSD storage

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Meta AI for the V-JEPA 2 model
- Milvus for vector database capabilities
- The open-source community for various dependencies

## Documentation

- [Technical Notes](docs/TECHNICAL_NOTES.md) - Zero-shot retrieval approach and limitations
- [System Architecture](docs/ARCHITECTURE.md) - Technical overview and component design
- [API Reference](docs/API.md) - Complete API documentation with examples
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment and scaling

## Support

For questions or issues, please:
- Check the [documentation](docs/)
- Open a new issue if needed

## Citation

If you use MotionMatch in your research, please cite:

```bibtex
@software{motionmatch2025,
  title={MotionMatch: Physics-based Video Search Engine},
  author={Vibhor Singh},
  year={2025},
  url={https://github.com/iamvibhorsingh/motionmatch}
}
```