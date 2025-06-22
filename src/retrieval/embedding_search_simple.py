"""
Simplified Embedding Search Service for MVP
Uses sentence-transformers for local embedding generation
"""
import logging
from typing import List, Tuple, Optional, Dict, Any
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
import time

logger = logging.getLogger(__name__)

class EmbeddingSearch:
    """Simplified embedding search for MVP using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a local sentence transformer model
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}  # memory_id -> embedding
        self.ready = False
    
    async def initialize(self):
        """Initialize the embedding model asynchronously"""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.model_name)
            )
            self.ready = True
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if the embedding service is ready"""
        return self.ready and self.model is not None
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not self.is_ready():
            raise RuntimeError("Embedding service not initialized")
        
        try:
            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode(text, convert_to_numpy=True)
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.is_ready():
            raise RuntimeError("Embedding service not initialized")
        
        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts, convert_to_numpy=True)
            )
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def store_embedding(self, memory_id: str, embedding: List[float]):
        """Store an embedding in the cache"""
        self.embeddings_cache[memory_id] = np.array(embedding)
    
    def get_stored_embedding(self, memory_id: str) -> Optional[List[float]]:
        """Get a stored embedding from cache"""
        embedding = self.embeddings_cache.get(memory_id)
        return embedding.tolist() if embedding is not None else None
    
    async def find_similar(self, 
                          query_embedding: List[float], 
                          k: int = 5, 
                          threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Find most similar embeddings in cache
        
        Args:
            query_embedding: Embedding to search with
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        if not self.embeddings_cache:
            return []
        
        query_vec = np.array(query_embedding)
        similarities = []
        
        for memory_id, stored_embedding in self.embeddings_cache.items():
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_vec, stored_embedding)
            if similarity >= threshold:
                similarities.append((memory_id, float(similarity)))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    async def search_by_text(self, 
                           query: str, 
                           k: int = 5, 
                           threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Search by text query
        
        Args:
            query: Text query to search with
            k: Number of results
            threshold: Minimum similarity
            
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        # Generate embedding for query
        query_embedding = await self.get_embedding(query)
        
        # Find similar embeddings
        return await self.find_similar(query_embedding, k, threshold)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up embedding service")
        self.model = None
        self.ready = False
        self.embeddings_cache.clear()