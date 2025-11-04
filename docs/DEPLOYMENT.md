# Deployment Guide

## Development Deployment

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- NVIDIA GPU (optional, for acceleration)

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/motionmatch.git
cd motionmatch

# Start infrastructure services
docker-compose -f docker/docker-compose.yml up -d

# Install dependencies
pip install -r requirements.txt

# Start the application
python start.py
```

The application will be available at `http://localhost:8000`

## Production Deployment

### Docker Deployment

1. **Build the image:**
```bash
docker build -t motionmatch:latest .
```

2. **Run with Docker Compose:**
```bash
# Copy and modify environment file
cp .env.example .env
# Edit .env with production values

# Start all services
docker-compose -f docker/docker-compose.yml --profile gpu up -d
```

### Kubernetes Deployment

1. **Create namespace:**
```bash
kubectl create namespace motionmatch
```

2. **Deploy infrastructure:**
```bash
# PostgreSQL
kubectl apply -f k8s/postgres.yaml

# Redis
kubectl apply -f k8s/redis.yaml

# Milvus
kubectl apply -f k8s/milvus.yaml
```

3. **Deploy application:**
```bash
# API server
kubectl apply -f k8s/api.yaml

# Worker nodes (GPU)
kubectl apply -f k8s/workers.yaml
```

### Environment Variables

#### Required
```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
MILVUS_HOST=milvus-host
REDIS_URL=redis://redis-host:6379/0
```

#### Optional
```bash
CUDA_AVAILABLE=true
BATCH_SIZE=4
NUM_FRAMES=64
ENABLE_CELERY=true
```

## Scaling

### Horizontal Scaling

**API Servers:**
```bash
# Scale API replicas
kubectl scale deployment motionmatch-api --replicas=3
```

**GPU Workers:**
```bash
# Scale worker replicas
kubectl scale deployment motionmatch-worker --replicas=2
```

### Vertical Scaling

**GPU Memory:**
- Adjust `BATCH_SIZE` based on available VRAM
- Monitor GPU utilization with `nvidia-smi`

**Database:**
- Use read replicas for PostgreSQL
- Configure Milvus sharding for large datasets

## Monitoring

### Health Checks
```bash
# Application health
curl http://localhost:8000/health

# Database connectivity
curl http://localhost:8000/v1/stats
```

### Metrics
- Prometheus metrics available at `/metrics`
- Grafana dashboards for visualization
- GPU metrics via nvidia-dcgm-exporter

### Logging
- Structured JSON logging
- Centralized with ELK stack or similar
- Log levels: DEBUG, INFO, WARNING, ERROR

## Security

### Production Checklist
- [ ] Change default passwords
- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up authentication
- [ ] Enable audit logging
- [ ] Regular security updates

### Network Security
```bash
# Restrict database access
iptables -A INPUT -p tcp --dport 5432 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 5432 -j DROP

# API rate limiting
# Configure in nginx or API gateway
```

## Backup and Recovery

### Database Backup
```bash
# PostgreSQL backup
pg_dump -h localhost -U motionmatch motionmatch > backup.sql

# Restore
psql -h localhost -U motionmatch motionmatch < backup.sql
```

### Vector Database Backup
```bash
# Milvus backup (requires Milvus Backup tool)
milvus-backup create --collection-name motion_vectors
```

### Disaster Recovery
1. Regular automated backups
2. Cross-region replication
3. Recovery time objective (RTO): 4 hours
4. Recovery point objective (RPO): 1 hour

## Performance Tuning

### GPU Optimization
```bash
# Optimal batch size for different GPUs
export BATCH_SIZE=8    # RTX 4090 (24GB)
export BATCH_SIZE=4    # RTX 4070 Ti (12GB)
export BATCH_SIZE=2    # RTX 3060 (8GB)
```

### Database Optimization
```sql
-- PostgreSQL indexes
CREATE INDEX idx_videos_created_at ON videos(created_at);
CREATE INDEX idx_videos_status ON videos(status);

-- Milvus index parameters
{
  "index_type": "IVF_FLAT",
  "metric_type": "L2",
  "params": {"nlist": 1024}
}
```

## Troubleshooting

### Common Issues

**GPU Out of Memory:**
```bash
# Reduce batch size
export BATCH_SIZE=1

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**Slow Search Performance:**
```bash
# Check Milvus index status
curl -X GET "http://milvus:9091/api/v1/collection/motion_vectors/index"

# Rebuild index if needed
curl -X POST "http://milvus:9091/api/v1/collection/motion_vectors/index"
```

**Database Connection Issues:**
```bash
# Test PostgreSQL connection
psql -h localhost -U motionmatch -d motionmatch -c "SELECT 1;"

# Check Redis connection
redis-cli -h localhost ping
```