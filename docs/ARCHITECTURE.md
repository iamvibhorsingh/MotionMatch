# System Architecture

## Overview

MotionMatch is built with a modular architecture that separates video processing, feature extraction, and search functionality.

### Zero-Shot Retrieval Approach

This project implements **zero-shot video retrieval** using V-JEPA 2 embeddings without any fine-tuning:

**Approach:**
- Extract embeddings from pre-trained V-JEPA 2 model
- Store embeddings in vector database (Milvus)
- Use cosine similarity for retrieval
- No training, fine-tuning, or metric learning

**Implications:**
- ✅ **Fast to deploy**: No training required
- ✅ **General purpose**: Works on any video domain
- ❌ **Limited accuracy**: Embeddings not optimized for similarity
- ❌ **Domain sensitivity**: Performance varies by video source

**Future Improvements:**
- Fine-tune with triplet/contrastive loss on domain-specific data
- Add metric learning head for better similarity representation
- Train on labeled motion similarity datasets

## Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   V-JEPA 2      │    │   Milvus        │
│   Web Server    │───▶│   Encoder       │───▶│   Vector DB     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Search        │    │   Indexing      │    │   PostgreSQL    │
│   Service       │    │   Service       │    │   Metadata      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 1. FastAPI Web Server
- REST API endpoints
- File upload handling
- Request validation
- Response formatting

### 2. V-JEPA 2 Encoder
- Video feature extraction
- GPU acceleration (CUDA)
- Batch processing support
- Caching for performance

### 3. Vector Database (Milvus)
- High-dimensional vector storage
- Similarity search
- Indexing and retrieval
- Scalable to millions of videos

### 4. Search Service
- Query processing
- Result ranking
- Similarity scoring
- Response formatting

### 5. Indexing Service
- Batch video processing
- Feature extraction pipeline
- Database updates
- Progress tracking

### 6. PostgreSQL
- Video metadata storage
- Search analytics
- User data (if applicable)
- System configuration

## Data Flow

### Indexing Flow
1. Video uploaded or path provided
2. Video processed by V-JEPA 2 encoder
3. Features extracted and normalized
4. Features stored in Milvus vector database
5. Metadata stored in PostgreSQL

### Search Flow
1. Query video uploaded
2. Features extracted from query video
3. Vector similarity search in Milvus
4. Results ranked and filtered
5. Response returned with similarity scores

## Deployment

### Development
- Docker Compose for local development
- All services run on single machine
- SQLite for lightweight metadata storage

### Production
- Kubernetes for orchestration
- Separate GPU nodes for encoding
- Distributed Milvus cluster
- PostgreSQL with replication

## Scalability

### Horizontal Scaling
- API servers: Multiple FastAPI instances behind load balancer
- Encoding: GPU worker pool with job queue
- Database: Milvus sharding and PostgreSQL read replicas

### Performance Optimization
- Feature caching for repeated queries
- Batch processing for bulk indexing
- GPU memory management
- Connection pooling

## Limitations & Future Work

### Current Limitations (Zero-Shot Approach)
1. **No Fine-Tuning**: Uses pre-trained V-JEPA 2 embeddings as-is
2. **Suboptimal Similarity**: Embeddings not trained for retrieval tasks
3. **High Similarity Scores**: Most results cluster around 95-98% similarity
4. **Domain Dependency**: Works best within consistent video sources

### Recommended Improvements
1. **Metric Learning**: Fine-tune with triplet/contrastive loss
2. **Domain Adaptation**: Train on specific use cases (sports, surveillance, etc.)
3. **Hybrid Approach**: Combine motion embeddings with visual features
4. **Re-ranking**: Add learned re-ranking model for better precision