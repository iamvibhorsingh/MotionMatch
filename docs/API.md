# API Documentation

## Base URL
```
http://localhost:8000
```

## Important Note: Zero-Shot Retrieval

This API uses **zero-shot video retrieval** with pre-trained V-JEPA 2 embeddings. The model was not fine-tuned for similarity search, which affects result quality:

- **Similarity scores** are typically high (95-98%) even for dissimilar videos
- **Best results** when query and indexed videos are from the same source/domain
- **Limited discrimination** between different motion types
- **No training required** - works out-of-the-box but with moderate accuracy

For production use, consider fine-tuning the model with metric learning on your specific video domain.

## Authentication
Currently no authentication required for development. Add API keys for production use.

## Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

### System Statistics
```http
GET /v1/stats
```

**Response:**
```json
{
  "total_videos": 1500,
  "total_features": 1500,
  "disk_usage_gb": 2.5,
  "last_indexed": "2024-01-01T11:30:00Z"
}
```

### Search Videos
```http
POST /v1/search
```

**Request Body:**
```json
{
  "query_video_url": "/path/to/video.mp4",
  "top_k": 10,
  "threshold": 0.7
}
```

**Response:**
```json
{
  "query_id": "uuid-string",
  "processing_time_ms": 1250.5,
  "results": [
    {
      "video_id": "video_001",
      "similarity_score": 0.95,
      "distance": 0.15,
      "video_path": "/path/to/similar_video.mp4",
      "metadata": {
        "duration": 12.3,
        "created_at": 1640995200.0
      }
    }
  ],
  "total_results": 1
}
```

### Search with Upload
```http
POST /v1/search/upload
```

**Form Data:**
- `file`: Video file (multipart/form-data)
- `top_k`: Number of results (optional, default: 10)

### Index Single Video
```http
POST /v1/index/single
```

**Query Parameters:**
- `video_path`: Path to video file

**Response:**
```json
{
  "video_id": "video_001",
  "status": "indexed",
  "processing_time_ms": 2500.0,
  "features_extracted": true
}
```

### Batch Index Videos
```http
POST /v1/index
```

**Request Body:**
```json
{
  "video_directory": "/path/to/videos/",
  "file_patterns": ["*.mp4", "*.avi"],
  "recursive": true
}
```

**Response:**
```json
{
  "job_id": "job_uuid",
  "status": "started",
  "total_videos": 150,
  "estimated_time_minutes": 45
}
```

### Check Indexing Status
```http
GET /v1/index/status/{job_id}
```

**Response:**
```json
{
  "job_id": "job_uuid",
  "status": "processing",
  "progress": {
    "completed": 75,
    "total": 150,
    "percentage": 50.0
  },
  "estimated_remaining_minutes": 22
}
```

### Delete Video
```http
DELETE /v1/videos/{video_id}
```

**Response:**
```json
{
  "video_id": "video_001",
  "status": "deleted"
}
```

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": "Error description",
  "code": "ERROR_CODE",
  "details": {
    "additional": "context"
  }
}
```

### Common Error Codes
- `VIDEO_NOT_FOUND`: Video file doesn't exist
- `ENCODING_FAILED`: Feature extraction failed
- `INVALID_FORMAT`: Unsupported video format
- `PROCESSING_ERROR`: Internal processing error
- `RATE_LIMITED`: Too many requests

## Rate Limits
- Search: 60 requests per minute
- Index: 10 requests per minute
- Upload: 5 requests per minute

## File Formats
Supported video formats:
- MP4 (recommended)
- AVI
- MOV
- MKV
- WebM

## Limits
- Max file size: 500MB
- Max video duration: 10 minutes
- Max batch size: 1000 videos