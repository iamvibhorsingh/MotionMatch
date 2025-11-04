"""Vector database service using Milvus for MotionMatch MVP"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, MilvusException
)
from motionmatch.core.config import config
from motionmatch.db.models import SearchResult

logger = logging.getLogger(__name__)

class VectorDBService:
    """Milvus vector database service"""
    
    def __init__(self):
        self.collection = None
        self.connected = False
        try:
            self._connect()
            self._setup_collection()
            self.connected = True
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            # Don't crash the application, allow graceful degradation
    
    def _connect(self):
        """Connect to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT
            )
            logger.info(f"Connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _setup_collection(self):
        """Setup Milvus collection"""
        try:
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.VECTOR_DIM),
                FieldSchema(name="video_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="duration", dtype=DataType.FLOAT),
                FieldSchema(name="created_at", dtype=DataType.FLOAT)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="MotionMatch video embeddings"
            )
            
            # Create or get collection
            if utility.has_collection(config.COLLECTION_NAME):
                self.collection = Collection(config.COLLECTION_NAME)
                logger.info(f"Using existing collection: {config.COLLECTION_NAME}")
            else:
                self.collection = Collection(
                    name=config.COLLECTION_NAME,
                    schema=schema
                )
                logger.info(f"Created new collection: {config.COLLECTION_NAME}")
            
            # Create index if not exists
            if not self.collection.has_index():
                # Use IP (Inner Product) for normalized vectors - equivalent to cosine similarity
                index_params = {
                    "metric_type": "IP",  # Inner Product (cosine similarity for normalized vectors)
                    "index_type": "HNSW",
                    "params": {
                        "M": 16,
                        "efConstruction": 200
                    }
                }
                
                self.collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
                logger.info("Created HNSW index with IP metric")
            
            # Load collection
            self.collection.load()
            logger.info("Collection loaded and ready")
            
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise
    
    def insert_video(
        self,
        video_id: str,
        embedding: np.ndarray,
        video_path: str,
        duration: float = 0.0,
        created_at: float = None
    ) -> bool:
        """Insert video embedding into database"""
        if not self.connected or self.collection is None:
            logger.error("Vector database not connected")
            return False
            
        try:
            if created_at is None:
                import time
                created_at = time.time()
            
            # Milvus expects column-wise data format
            data = [
                {
                    "video_id": video_id,
                    "embedding": embedding.tolist(),
                    "video_path": video_path,
                    "duration": duration,
                    "created_at": created_at
                }
            ]
            
            result = self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Inserted video {video_id} with {result.insert_count} records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert video {video_id}: {e}")
            return False
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar videos with optimized parameters"""
        if not self.connected or self.collection is None:
            logger.error("Vector database not connected")
            return []
            
        try:
            # Optimized search parameters for speed
            # ef must be >= top_k for HNSW index
            ef = max(top_k, 64)  # Use at least 64, but scale with top_k if needed
            
            search_params = {
                "metric_type": "IP",  # Inner Product (cosine similarity)
                "params": {
                    "ef": ef
                }
            }
            
            # Build filter expression if provided
            expr = None
            if filters:
                conditions = []
                if "duration_min" in filters:
                    conditions.append(f"duration >= {filters['duration_min']}")
                if "duration_max" in filters:
                    conditions.append(f"duration <= {filters['duration_max']}")
                if conditions:
                    expr = " and ".join(conditions)
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["video_id", "video_path", "duration", "created_at"]
            )
            
            # Parse results
            search_results = []
            for hits in results:
                for hit in hits:
                    # For IP metric, distance is actually the inner product (similarity)
                    # Higher is better, range is [-1, 1] for normalized vectors
                    similarity_score = hit.distance  # Already a similarity score with IP
                    distance = 1.0 - similarity_score  # Convert to distance for compatibility
                    
                    result = SearchResult(
                        video_id=hit.entity.get("video_id"),
                        similarity_score=similarity_score,
                        distance=distance,
                        video_path=hit.entity.get("video_path"),
                        metadata={
                            "duration": hit.entity.get("duration"),
                            "created_at": hit.entity.get("created_at")
                        }
                    )
                    search_results.append(result)
            
            logger.info(f"Found {len(search_results)} similar videos")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_video_count(self) -> int:
        """Get total number of videos in database"""
        try:
            return self.collection.num_entities
        except Exception as e:
            logger.error(f"Failed to get video count: {e}")
            return 0
    
    def video_exists(self, video_id: str) -> bool:
        """Check if a video is already indexed"""
        if not self.connected or self.collection is None:
            logger.warning("Collection not connected, cannot check if video exists")
            return False
            
        try:
            # Escape backslashes for Milvus query
            escaped_id = video_id.replace("\\", "\\\\")
            expr = f'video_id == "{escaped_id}"'
            logger.debug(f"Checking existence with query: {expr}")
            
            results = self.collection.query(
                expr=expr,
                output_fields=["video_id"],
                limit=1
            )
            exists = len(results) > 0
            
            logger.info(f"Video {video_id} exists check: {exists} (found {len(results)} results)")
            
            return exists
        except Exception as e:
            logger.warning(f"Failed to check if video {video_id} exists (assuming not): {e}")
            return False  # If check fails, assume doesn't exist and try to index
    
    def get_all_video_ids(self) -> List[str]:
        """Get all indexed video IDs"""
        try:
            # Query all videos (limit to reasonable number)
            results = self.collection.query(
                expr="video_id != ''",
                output_fields=["video_id"],
                limit=10000
            )
            video_ids = [r["video_id"] for r in results]
            logger.info(f"Found {len(video_ids)} video IDs in database")
            if video_ids and len(video_ids) <= 10:
                logger.info(f"Sample IDs: {video_ids[:10]}")
            return video_ids
        except Exception as e:
            logger.error(f"Failed to get video IDs: {e}")
            return []
    
    def delete_video(self, video_id: str) -> bool:
        """Delete video from database"""
        try:
            expr = f'video_id == "{video_id}"'
            self.collection.delete(expr)
            self.collection.flush()
            logger.info(f"Deleted video {video_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete video {video_id}: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all videos from the collection"""
        try:
            # Drop and recreate collection for clean slate
            if self.collection:
                self.collection.release()
                utility.drop_collection(config.COLLECTION_NAME)
                logger.info(f"Dropped collection {config.COLLECTION_NAME}")
            
            # Recreate collection
            self._setup_collection()
            logger.info("Collection recreated and ready")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def health_check(self) -> dict:
        """Health check for vector database"""
        try:
            count = self.get_video_count()
            return {
                "status": "healthy",
                "collection_name": config.COLLECTION_NAME,
                "video_count": count,
                "connected": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False
            }

# Global vector database instance
vector_db = VectorDBService()