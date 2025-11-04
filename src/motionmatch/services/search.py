"""Search service for MotionMatch MVP"""
import os
import time
import uuid
import logging
import threading
from typing import List, Optional, Dict, Any
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from motionmatch.services.encoder import encoder_service
from motionmatch.db.vector_db import vector_db
from motionmatch.db.models import SearchRequest, SearchResponse, SearchResult
from motionmatch.core.config import config

logger = logging.getLogger(__name__)

class SearchService:
    """Main search service orchestrating query processing"""
    
    def __init__(self):
        self.query_cache = {}  # Simple in-memory cache for demo
        self.cache_lock = threading.Lock()  # Thread safety
    
    def search(self, request: SearchRequest) -> SearchResponse:
        """Process search request end-to-end"""
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Processing search query {query_id}")
            
            # Step 1: Encode query video
            query_features = self._encode_query_video(request.query_video_url)
            
            # Step 2: Vector search
            candidates = self._vector_search(
                query_features.global_features,
                top_k=config.SEARCH_TOP_K,
                filters=request.filters
            )
            
            # Step 3: Optional re-ranking
            enable_reranking = request.options.enable_reranking if request.options else False
            if enable_reranking and len(candidates) > 0:
                candidates = self._rerank_results(
                    query_features.temporal_features,
                    candidates,
                    top_k=request.top_k
                )
            else:
                candidates = candidates[:request.top_k]
            
            processing_time = (time.time() - start_time) * 1000
            
            return SearchResponse(
                query_id=query_id,
                processing_time_ms=processing_time,
                results=candidates,
                total_results=len(candidates)
            )
            
        except Exception as e:
            logger.error(f"Search failed for query {query_id}: {e}")
            raise
    
    def _encode_query_video(self, video_path: str):
        """Encode query video with persistent disk caching"""
        import hashlib
        from pathlib import Path
        
        # Create cache key from file content hash for better cache hits
        try:
            with open(video_path, 'rb') as f:
                # Read first 1MB for hash (faster than full file)
                file_sample = f.read(1024 * 1024)
                file_hash = hashlib.md5(file_sample).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash file, using path hash: {e}")
            file_hash = hashlib.md5(video_path.encode()).hexdigest()
        
        # Check in-memory cache first (fastest)
        cache_key = f"query_{file_hash}"
        with self.cache_lock:
            if cache_key in self.query_cache:
                logger.info("Using in-memory cached query features")
                return self.query_cache[cache_key]
        
        # Check disk cache (fast)
        cache_dir = Path(config.STORAGE_PATH) / "query_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{file_hash}.npz"
        
        if cache_file.exists():
            try:
                logger.info("Using disk cached query features")
                data = np.load(cache_file, allow_pickle=True)
                
                # Reconstruct VideoFeatures object
                from motionmatch.db.models import VideoFeatures
                features = VideoFeatures(
                    video_id=str(data['video_id'].item()) if hasattr(data['video_id'], 'item') else str(data['video_id']),
                    global_features=data['global_features'],
                    temporal_features=data['temporal_features'],
                    metadata=dict(data['metadata'].item()) if hasattr(data['metadata'], 'item') else dict(data['metadata']),
                    created_at=float(data['created_at'].item()) if hasattr(data['created_at'], 'item') else float(data['created_at'])
                )
                
                # Store in memory cache for next time
                with self.cache_lock:
                    self.query_cache[cache_key] = features
                
                return features
            except Exception as e:
                logger.warning(f"Failed to load cache file, re-encoding: {e}")
                # Delete corrupted cache file
                try:
                    cache_file.unlink()
                except:
                    pass
        
        # Encode video (slowest path)
        logger.info("Encoding query video (no cache hit)")
        features = encoder_service.encode_video(video_path)
        
        # Save to disk cache
        try:
            np.savez(
                cache_file,
                video_id=features.video_id,
                global_features=features.global_features,
                temporal_features=features.temporal_features,
                metadata=features.metadata,
                created_at=features.created_at
            )
            logger.info(f"Saved query features to disk cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache file: {e}")
        
        # Store in memory cache
        with self.cache_lock:
            self.query_cache[cache_key] = features
        
        return features
    
    def _vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform vector similarity search"""
        return vector_db.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
    
    def _rerank_results(
        self,
        query_temporal: np.ndarray,
        candidates: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Re-rank results using advanced temporal similarity"""
        try:
            logger.info(f"Re-ranking {len(candidates)} candidates using temporal features")
            
            # Load temporal features for candidates and compute similarity
            reranked_candidates = []
            
            for candidate in candidates:
                try:
                    # Load candidate's temporal features from storage
                    features_dir = os.path.join(config.STORAGE_PATH, "temporal_features")
                    temporal_path = os.path.join(features_dir, f"{candidate.video_id}_temporal.npy")
                    
                    if os.path.exists(temporal_path):
                        candidate_temporal = np.load(temporal_path)
                        
                        # Compute multiple temporal similarity metrics
                        # 1. DTW distance (captures temporal alignment)
                        dtw_dist = self._compute_dtw_distance(query_temporal, candidate_temporal)
                        dtw_similarity = 1.0 / (1.0 + dtw_dist)
                        
                        # 2. Cosine similarity on temporal average (captures overall motion)
                        query_avg = query_temporal.mean(axis=0)
                        candidate_avg = candidate_temporal.mean(axis=0)
                        cosine_sim = np.dot(query_avg, candidate_avg) / (
                            np.linalg.norm(query_avg) * np.linalg.norm(candidate_avg) + 1e-8
                        )
                        
                        # 3. Temporal variance similarity (captures motion dynamics)
                        query_var = query_temporal.var(axis=0).mean()
                        candidate_var = candidate_temporal.var(axis=0).mean()
                        var_similarity = 1.0 - abs(query_var - candidate_var) / (query_var + candidate_var + 1e-8)
                        
                        # Weighted combination of metrics
                        temporal_score = (
                            0.5 * dtw_similarity +      # DTW is most important
                            0.3 * cosine_sim +          # Overall similarity
                            0.2 * var_similarity        # Motion dynamics
                        )
                        
                        # Combine with original global similarity (70% temporal, 30% global)
                        combined_score = 0.7 * temporal_score + 0.3 * candidate.similarity_score
                        
                        candidate.similarity_score = combined_score
                        candidate.metadata['temporal_score'] = float(temporal_score)
                        candidate.metadata['dtw_similarity'] = float(dtw_similarity)
                        
                    else:
                        # No temporal features available, keep original score
                        logger.warning(f"No temporal features found for {candidate.video_id}")
                    
                    reranked_candidates.append(candidate)
                    
                except Exception as e:
                    logger.warning(f"Failed to rerank {candidate.video_id}: {e}")
                    reranked_candidates.append(candidate)
            
            # Sort by updated similarity scores
            reranked_candidates.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(f"Re-ranking complete, top score: {reranked_candidates[0].similarity_score:.3f}")
            return reranked_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            # Return original results if re-ranking fails
            return candidates[:top_k]
    
    def _compute_dtw_distance(
        self,
        query_temporal: np.ndarray,
        candidate_temporal: np.ndarray,
        use_gpu: bool = False
    ) -> float:
        """Compute DTW distance between temporal features
        
        Args:
            query_temporal: Query temporal features
            candidate_temporal: Candidate temporal features
            use_gpu: Use GPU-accelerated computation (experimental)
        """
        try:
            if use_gpu and config.DEVICE == "cuda":
                # GPU-accelerated approximate DTW using cosine similarity
                import torch
                q = torch.from_numpy(query_temporal).cuda()
                c = torch.from_numpy(candidate_temporal).cuda()
                
                # Compute pairwise cosine similarity
                q_norm = q / (q.norm(dim=1, keepdim=True) + 1e-8)
                c_norm = c / (c.norm(dim=1, keepdim=True) + 1e-8)
                sim_matrix = torch.mm(q_norm, c_norm.t())
                
                # Approximate DTW with diagonal path
                distance = (1 - sim_matrix.diag()).sum().item()
                return distance
            else:
                # CPU-based FastDTW
                distance, _ = fastdtw(
                    query_temporal,
                    candidate_temporal,
                    radius=10,  # Constrained DTW for speed
                    dist=euclidean
                )
                return distance
        except Exception as e:
            logger.error(f"DTW computation failed: {e}")
            return float('inf')

# Global search service instance
search_service = SearchService()