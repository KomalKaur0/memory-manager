"""
Mock Embedding Search Service for quick testing without dependencies
"""
import logging
from typing import List, Tuple, Optional, Dict, Any
import asyncio
import random
import time

logger = logging.getLogger(__name__)

class EmbeddingSearch:
    """Mock embedding search for testing without heavy dependencies"""
    
    def __init__(self, model_name: str = "mock-model"):
        """
        Initialize with a mock model
        
        Args:
            model_name: Name of the mock model
        """
        self.model_name = model_name
        self.embeddings_cache = {}  # memory_id -> embedding
        self.ready = False
    
    async def initialize(self):
        """Initialize the mock embedding model"""
        try:
            logger.info(f"Loading mock embedding model: {self.model_name}")
            # Simulate loading time
            await asyncio.sleep(0.1)
            self.ready = True
            logger.info("Mock embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load mock embedding model: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if the embedding service is ready"""
        return self.ready
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate mock embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Mock embedding vector as list of floats
        """
        if not self.is_ready():
            raise RuntimeError("Embedding service not initialized")
        
        try:
            # Generate deterministic mock embedding based on text
            random.seed(hash(text) % 2**32)
            embedding = [random.uniform(-1, 1) for _ in range(384)]
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate mock embedding: {e}")
            raise
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate mock embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of mock embedding vectors
        """
        if not self.is_ready():
            raise RuntimeError("Embedding service not initialized")
        
        try:
            embeddings = []
            for text in texts:
                embedding = await self.get_embedding(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate batch mock embeddings: {e}")
            raise
    
    def store_embedding(self, memory_id: str, embedding: List[float]):
        """Store an embedding in the cache"""
        import numpy as np
        self.embeddings_cache[memory_id] = np.array(embedding) if hasattr(np, 'array') else embedding
    
    def get_stored_embedding(self, memory_id: str) -> Optional[List[float]]:
        """Get a stored embedding from cache"""
        embedding = self.embeddings_cache.get(memory_id)
        if embedding is not None:
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            else:
                return embedding
        return None
    
    async def find_similar(self, 
                          query_embedding: List[float], 
                          k: int = 5, 
                          threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Find most similar embeddings in cache using mock similarity
        
        Args:
            query_embedding: Embedding to search with
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        if not self.embeddings_cache:
            return []
        
        similarities = []
        
        for memory_id, stored_embedding in self.embeddings_cache.items():
            # Mock cosine similarity calculation
            similarity = self._mock_cosine_similarity(query_embedding, stored_embedding)
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
        Search by text query using mock embeddings
        
        Args:
            query: Text query to search with
            k: Number of results
            threshold: Minimum similarity
            
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        # Generate mock embedding for query
        query_embedding = await self.get_embedding(query)
        
        # Find similar embeddings
        return await self.find_similar(query_embedding, k, threshold)
    
    def _mock_cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate mock cosine similarity between two vectors"""
        try:
            # Simple mock similarity - for testing, return higher similarity for similar text
            # In a real implementation, this would be actual cosine similarity
            
            # Convert to string representations to check for similarity
            a_hash = hash(str(a[:10]))  # Use first 10 elements
            b_hash = hash(str(b[:10]))
            
            # If hashes are close, return high similarity (for same text)
            if abs(a_hash - b_hash) < 1000:
                return random.uniform(0.8, 0.95)
            else:
                return random.uniform(0.6, 0.8)  # Still return reasonable scores
        except:
            return random.uniform(0.6, 0.8)
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up mock embedding service")
        self.ready = False
        self.embeddings_cache.clear()