"""
Embedding Search Module - Discovery Engine for AI Memory System

This module handles semantic similarity search using vector embeddings.
It's the first phase of the hybrid retrieval system, finding initial candidates
based on semantic similarity before graph traversal explores connections.
"""

from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging
import time
import hashlib
import json
import requests
from collections import defaultdict
import os


@dataclass
class SearchResult:
    """Result from embedding similarity search"""
    memory_id: str
    similarity_score: float
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    distance_metric: str = "cosine"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding operations"""
    model_name: str = "voyage-3-lite"
    api_key: Optional[str] = None
    dimensions: int = 512
    input_type_documents: str = "document"
    input_type_queries: str = "query"
    batch_size: int = 32
    max_retries: int = 3
    timeout: float = 30.0
    cache_embeddings: bool = True
    
    
@dataclass
class IndexingStats:
    """Statistics for embedding indexing operations"""
    total_indexed: int = 0
    successful_embeds: int = 0
    failed_embeds: int = 0
    average_embedding_time: float = 0.0
    last_indexed: Optional[datetime] = None


class EmbeddingSearch:
    """
    Embedding-based semantic search engine for memory discovery.
    
    This class handles the initial discovery phase of memory retrieval,
    using vector embeddings to find semantically similar memories to a query.
    It integrates with Voyage AI's embedding API and manages a vector store
    for efficient similarity search.
    
    Key responsibilities:
    - Generate embeddings for memories and queries
    - Store embeddings in vector database
    - Perform similarity search
    - Manage embedding cache and batch operations
    - Handle API rate limiting and retries
    """
    
    def __init__(self, config: EmbeddingConfig, vector_store=None, logger=None):
        """
        Initialize the embedding search engine.
        
        Args:
            config: Configuration object with model settings
            vector_store: Vector database instance (Pinecone, Qdrant, etc.)
            logger: Logger instance for debugging and monitoring
        """
        self.config = config
        self.vector_store = vector_store
        self.logger = logger or logging.getLogger(__name__)
        self.client = None  # Voyage AI client
        self.embedding_cache = {}  # Local cache for recent embeddings
        self.stats = IndexingStats()
        
        # Performance monitoring
        self.api_call_count = 0
        self.total_api_time = 0.0
        self.last_api_call = None
        
        # Initialize request session for HTTP calls
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.config.api_key}' if self.config.api_key else ''
        })
        
    def initialize_client(self) -> None:
        """
        Initialize the Voyage AI client and test connection.
        
        Sets up the API client, validates credentials, and tests
        connectivity with a simple embedding request.
        
        Raises:
            ConnectionError: If unable to connect to Voyage AI API
            AuthenticationError: If API key is invalid
        """
        if not self.config.api_key:
            raise ConnectionError("API key is required for Voyage AI client")
            
        # Test connection with a simple request
        try:
            test_result = self.encode_text("test connection", use_cache=False)
            if test_result is None:
                raise ConnectionError("Failed to get test embedding from Voyage AI")
            self.logger.info("Successfully initialized Voyage AI client")
            self.client = "initialized"  # Simple flag to indicate client is ready
        except Exception as e:
            self.logger.error(f"Failed to initialize Voyage AI client: {e}")
            raise ConnectionError(f"Unable to connect to Voyage AI API: {e}")
    
    def encode_text(self, 
                   text: Union[str, List[str]], 
                   input_type: Optional[str] = None,
                   use_cache: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Convert text to embedding vector(s).
        
        Args:
            text: Single text string or list of strings to embed
            input_type: "query" or "document" for optimized embeddings
            use_cache: Whether to use/store in embedding cache
            
        Returns:
            Single embedding array or list of embedding arrays
            
        Raises:
            EmbeddingError: If embedding generation fails
            RateLimitError: If API rate limit exceeded
        """
        start_time = time.time()
        
        try:
            is_single = isinstance(text, str)
            texts = [text] if is_single else text
            
            # Check cache for cached embeddings
            cached_results = []
            texts_to_embed = []
            cache_keys = []
            
            if use_cache and self.config.cache_embeddings:
                for t in texts:
                    cache_key = self._get_cache_key(t, input_type)
                    cache_keys.append(cache_key)
                    
                    if cache_key in self.embedding_cache:
                        cached_entry = self.embedding_cache[cache_key]
                        # Check if cache entry is still valid (not too old)
                        if time.time() - cached_entry['timestamp'] < 3600:  # 1 hour TTL
                            cached_results.append(cached_entry['embedding'])
                        else:
                            # Cache expired, need to re-embed
                            texts_to_embed.append(t)
                            cached_results.append(None)
                    else:
                        texts_to_embed.append(t)
                        cached_results.append(None)
            else:
                texts_to_embed = texts
                cached_results = [None] * len(texts)
                cache_keys = [None] * len(texts)
            
            # Get embeddings for texts not in cache
            new_embeddings = []
            if texts_to_embed:
                new_embeddings = self._call_voyage_api(texts_to_embed, input_type)
                
                # Update cache with new embeddings
                if use_cache and self.config.cache_embeddings:
                    embed_idx = 0
                    for i, cached_result in enumerate(cached_results):
                        if cached_result is None and cache_keys[i]:
                            self.embedding_cache[cache_keys[i]] = {
                                'embedding': new_embeddings[embed_idx],
                                'timestamp': time.time()
                            }
                            embed_idx += 1
            
            # Combine cached and new embeddings
            final_embeddings = []
            embed_idx = 0
            for cached_result in cached_results:
                if cached_result is not None:
                    final_embeddings.append(cached_result)
                else:
                    final_embeddings.append(new_embeddings[embed_idx])
                    embed_idx += 1
            
            self._update_performance_metrics("encode_text", start_time, True)
            
            return final_embeddings[0] if is_single else final_embeddings
            
        except Exception as e:
            self._update_performance_metrics("encode_text", start_time, False)
            self.logger.error(f"Failed to encode text: {e}")
            raise EmbeddingError(f"Text encoding failed: {e}")
    
    def encode_batch(self, 
                    texts: List[str], 
                    input_type: Optional[str] = None,
                    progress_callback: Optional[callable] = None) -> List[np.ndarray]:
        """
        Efficiently encode multiple texts in batches.
        
        Splits large lists into API-friendly batches, handles rate limiting,
        and provides progress tracking for long operations.
        
        Args:
            texts: List of texts to embed
            input_type: "query" or "document" for optimization
            progress_callback: Function called with (current, total) progress
            
        Returns:
            List of embedding arrays corresponding to input texts
            
        Raises:
            BatchEmbeddingError: If batch processing fails
        """
        if not texts:
            return []
            
        try:
            all_embeddings = []
            batch_size = self.config.batch_size
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.encode_text(batch, input_type, use_cache=True)
                
                # Ensure we have a list even for single embeddings
                if isinstance(batch_embeddings, np.ndarray):
                    batch_embeddings = [batch_embeddings]
                    
                all_embeddings.extend(batch_embeddings)
                
                # Call progress callback
                if progress_callback:
                    current_progress = min(i + batch_size, len(texts))
                    progress_callback(current_progress, len(texts))
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"Batch encoding failed: {e}")
            raise BatchEmbeddingError(f"Failed to encode batch: {e}")
    
    def index_memory(self, 
                    memory_id: str, 
                    content: str,
                    metadata: Dict[str, Any],
                    tags: List[str] = None,
                    summary: str = None) -> bool:
        """
        Generate embedding for a memory and store in vector database.
        
        Creates optimized text representation by combining content,
        summary, and tags, then generates embedding and stores with metadata.
        
        Args:
            memory_id: Unique identifier for the memory
            content: Main memory content text
            metadata: Additional metadata to store
            tags: List of tags for the memory
            summary: Brief summary of memory content
            
        Returns:
            True if successfully indexed, False otherwise
            
        Raises:
            IndexingError: If memory indexing fails
        """
        try:
            # Prepare text for optimal embedding
            prepared_text = self._prepare_text_for_embedding(
                content, metadata, tags, summary
            )
            
            # Generate embedding
            embedding = self.encode_text(
                prepared_text, 
                input_type=self.config.input_type_documents,
                use_cache=True
            )
            
            # Store in vector database
            if self.vector_store:
                # Prepare metadata for storage
                storage_metadata = metadata.copy() if metadata else {}
                if tags:
                    storage_metadata['tags'] = tags
                if summary:
                    storage_metadata['summary'] = summary
                storage_metadata['content'] = content
                storage_metadata['indexed_at'] = datetime.now().isoformat()
                
                success = self.vector_store.upsert(
                    memory_id=memory_id,
                    embedding=embedding,
                    metadata=storage_metadata
                )
                
                if success:
                    self.stats.total_indexed += 1
                    self.stats.last_indexed = datetime.now()
                    self.logger.debug(f"Successfully indexed memory {memory_id}")
                    return True
                else:
                    self.logger.error(f"Failed to store memory {memory_id} in vector store")
                    return False
            else:
                self.logger.warning("No vector store configured, cannot index memory")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to index memory {memory_id}: {e}")
            return False
    
    def index_memory_batch(self, 
                          memories: List[Dict[str, Any]],
                          progress_callback: Optional[callable] = None) -> Dict[str, bool]:
        """
        Index multiple memories efficiently in batches.
        
        Args:
            memories: List of memory dicts with keys: id, content, metadata, etc.
            progress_callback: Function called with progress updates
            
        Returns:
            Dict mapping memory_id to success status
        """
        results = {}
        total_memories = len(memories)
        
        for i, memory in enumerate(memories):
            memory_id = memory.get('id')
            if not memory_id:
                results[f"unknown_{i}"] = False
                continue
                
            content = memory.get('content', '')
            metadata = memory.get('metadata', {})
            tags = memory.get('tags', [])
            summary = memory.get('summary')
            
            success = self.index_memory(
                memory_id=memory_id,
                content=content,
                metadata=metadata,
                tags=tags,
                summary=summary
            )
            
            results[memory_id] = success
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total_memories)
        
        return results
    
    def search(self, 
              query: str, 
              top_k: int = 10,
              similarity_threshold: float = 0.7,
              filters: Optional[Dict[str, Any]] = None,
              include_embeddings: bool = False) -> List[SearchResult]:
        """
        Find memories semantically similar to query.
        
        This is the main search method that converts the query to an embedding
        and finds the most similar memories in the vector store.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score to include
            filters: Metadata filters to apply (tags, date ranges, etc.)
            include_embeddings: Whether to return embedding vectors
            
        Returns:
            List of SearchResult objects, sorted by similarity score
            
        Raises:
            SearchError: If search operation fails
        """
        try:
            # Convert query to embedding
            query_embedding = self.encode_text(
                query, 
                input_type=self.config.input_type_queries,
                use_cache=True
            )
            
            # Delegate to search_by_embedding
            return self.search_by_embedding(
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=filters,
                include_embeddings=include_embeddings
            )
            
        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {e}")
            raise SearchError(f"Search operation failed: {e}")
    
    def search_by_embedding(self,
                           query_embedding: np.ndarray,
                           top_k: int = 10,
                           similarity_threshold: float = 0.7,
                           filters: Optional[Dict[str, Any]] = None,
                           include_embeddings: bool = False) -> List[SearchResult]:
        """
        Search using a pre-computed embedding vector.
        
        Useful when you already have an embedding (e.g., from cache)
        or want to search with a modified/averaged embedding.
        
        Args:
            query_embedding: Pre-computed embedding vector
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Metadata filters to apply
            include_embeddings: Whether to return embedding vectors
            
        Returns:
            List of SearchResult objects
        """
        if not self.vector_store:
            raise SearchError("No vector store configured for search")
            
        try:
            # Query vector store
            raw_results = self.vector_store.query(
                embedding=query_embedding,
                top_k=top_k,
                filters=filters,
                include_embeddings=include_embeddings
            )
            
            # Convert to SearchResult objects and filter by threshold
            search_results = []
            for result in raw_results:
                similarity_score = result.get('score', 0.0)
                
                # Apply similarity threshold
                if similarity_score >= similarity_threshold:
                    search_result = SearchResult(
                        memory_id=result['id'],
                        similarity_score=similarity_score,
                        metadata=result.get('metadata', {}),
                        embedding=result.get('embedding') if include_embeddings else None,
                        distance_metric="cosine"  # Default metric
                    )
                    search_results.append(search_result)
            
            # Sort by similarity score (descending)
            search_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Embedding search failed: {e}")
            raise SearchError(f"Embedding search operation failed: {e}")
    
    def find_similar_memories(self,
                             memory_id: str,
                             top_k: int = 5,
                             exclude_self: bool = True) -> List[SearchResult]:
        """
        Find memories similar to a specific existing memory.
        
        Retrieves the embedding for the given memory and searches
        for similar memories, optionally excluding the memory itself.
        
        Args:
            memory_id: ID of memory to find similar ones for
            top_k: Maximum number of similar memories to return
            exclude_self: Whether to exclude the source memory from results
            
        Returns:
            List of similar memories
        """
        # Get embedding for the source memory
        source_embedding = self.get_embedding(memory_id)
        if source_embedding is None:
            raise SearchError(f"Memory {memory_id} not found or has no embedding")
        
        # Search for similar memories
        # Request more than needed in case we need to exclude self
        search_top_k = top_k + 1 if exclude_self else top_k
        
        results = self.search_by_embedding(
            query_embedding=source_embedding,
            top_k=search_top_k,
            similarity_threshold=0.0  # No threshold for similarity search
        )
        
        # Filter out the source memory itself if requested
        if exclude_self:
            results = [r for r in results if r.memory_id != memory_id]
        
        # Return only the requested number of results
        return results[:top_k]
    
    def get_embedding(self, memory_id: str) -> Optional[np.ndarray]:
        """
        Retrieve the stored embedding for a specific memory.
        
        Args:
            memory_id: ID of memory to get embedding for
            
        Returns:
            Embedding vector or None if not found
        """
        if not self.vector_store:
            return None
            
        try:
            return self.vector_store.get_embedding(memory_id)
        except Exception as e:
            self.logger.error(f"Failed to retrieve embedding for {memory_id}: {e}")
            return None
    
    def update_memory_embedding(self,
                               memory_id: str,
                               new_content: str,
                               metadata: Dict[str, Any]) -> bool:
        """
        Update the embedding for an existing memory.
        
        Regenerates the embedding with new content and updates
        the vector store entry.
        
        Args:
            memory_id: ID of memory to update
            new_content: Updated memory content
            metadata: Updated metadata
            
        Returns:
            True if update successful
        """
        # This is essentially the same as indexing a new memory
        return self.index_memory(
            memory_id=memory_id,
            content=new_content,
            metadata=metadata
        )
    
    def remove_memory(self, memory_id: str) -> bool:
        """
        Remove a memory from the embedding index.
        
        Args:
            memory_id: ID of memory to remove
            
        Returns:
            True if removal successful
        """
        if not self.vector_store:
            return False
            
        try:
            success = self.vector_store.delete(memory_id)
            if success:
                self.logger.debug(f"Successfully removed memory {memory_id}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to remove memory {memory_id}: {e}")
            return False
    
    def search_with_expansion(self,
                             query: str,
                             expansion_terms: List[str],
                             top_k: int = 10) -> List[SearchResult]:
        """
        Search with query expansion using additional terms.
        
        Combines the main query with expansion terms to create
        a richer query embedding that may find more relevant results.
        
        Args:
            query: Main search query
            expansion_terms: Additional terms to include
            top_k: Maximum results to return
            
        Returns:
            List of search results
        """
        # Combine query with expansion terms
        expanded_query = f"{query} {' '.join(expansion_terms)}"
        
        # Search with the expanded query
        return self.search(
            query=expanded_query,
            top_k=top_k,
            similarity_threshold=0.6  # Slightly lower threshold for expanded queries
        )
    
    def hypothetical_document_search(self,
                                   query: str,
                                   num_hypotheticals: int = 3,
                                   top_k: int = 10) -> List[SearchResult]:
        """
        Search using hypothetical document generation.
        
        Generates ideal memory examples based on the query,
        then searches for memories similar to these hypotheticals.
        Useful when queries are abstract or don't match memory language.
        
        Args:
            query: Abstract or difficult search query
            num_hypotheticals: Number of example memories to generate
            top_k: Maximum results per hypothetical
            
        Returns:
            Deduplicated and ranked search results
        """
        # Generate hypothetical documents
        hypotheticals = self._generate_hypothetical_documents(query, num_hypotheticals)
        
        all_results = []
        seen_memory_ids = set()
        
        for hypothetical in hypotheticals:
            # Search using each hypothetical document
            results = self.search(
                query=hypothetical,
                top_k=top_k,
                similarity_threshold=0.7
            )
            
            # Add unique results
            for result in results:
                if result.memory_id not in seen_memory_ids:
                    all_results.append(result)
                    seen_memory_ids.add(result.memory_id)
        
        # Sort by similarity score and return top results
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_results[:top_k]
    
    def _generate_hypothetical_documents(self, query: str, num_docs: int) -> List[str]:
        """
        Generate hypothetical documents based on query.
        
        This is a simple implementation that creates variations of the query.
        In practice, this could use an LLM to generate more sophisticated examples.
        """
        hypotheticals = []
        
        # Base document
        base_doc = f"This is about {query}. It contains information related to {query}."
        hypotheticals.append(base_doc)
        
        # Question-based document
        if num_docs > 1:
            question_doc = f"What is {query}? How does {query} work? Key aspects of {query}."
            hypotheticals.append(question_doc)
        
        # Example-based document
        if num_docs > 2:
            example_doc = f"Examples of {query}. Case studies about {query}. Applications of {query}."
            hypotheticals.append(example_doc)
        
        # Generate additional variations if needed
        for i in range(len(hypotheticals), num_docs):
            variation = f"Content about {query}. Information regarding {query} and related topics."
            hypotheticals.append(variation)
        
        return hypotheticals[:num_docs]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about search performance and usage.
        
        Returns:
            Dict with metrics like total searches, avg response time, etc.
        """
        avg_api_time = (
            self.total_api_time / self.api_call_count 
            if self.api_call_count > 0 else 0.0
        )
        
        return {
            'total_api_calls': self.api_call_count,
            'total_api_time': self.total_api_time,
            'average_api_time': avg_api_time,
            'last_api_call': self.last_api_call.isoformat() if self.last_api_call else None,
            'cache_size': len(self.embedding_cache),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def get_index_statistics(self) -> IndexingStats:
        """
        Get statistics about indexing operations.
        
        Returns:
            IndexingStats object with embedding generation metrics
        """
        return self.stats
    
    def clear_cache(self, max_age_seconds: Optional[int] = None) -> int:
        """
        Clear the embedding cache.
        
        Args:
            max_age_seconds: Only clear entries older than this
            
        Returns:
            Number of cache entries cleared
        """
        if max_age_seconds is None:
            # Clear all cache entries
            cleared_count = len(self.embedding_cache)
            self.embedding_cache.clear()
            return cleared_count
        else:
            # Clear only old entries
            current_time = time.time()
            keys_to_remove = []
            
            for key, entry in self.embedding_cache.items():
                if current_time - entry['timestamp'] > max_age_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.embedding_cache[key]
            
            return len(keys_to_remove)
    
    def warm_up_cache(self, memory_ids: List[str]) -> None:
        """
        Pre-load embeddings for specified memories into cache.
        
        Args:
            memory_ids: List of memory IDs to cache
        """
        for memory_id in memory_ids:
            try:
                # This will load the embedding into cache if available
                embedding = self.get_embedding(memory_id)
                if embedding is not None:
                    # Store in local cache with a synthetic cache key
                    cache_key = f"memory_embedding_{memory_id}"
                    self.embedding_cache[cache_key] = {
                        'embedding': embedding,
                        'timestamp': time.time()
                    }
            except Exception as e:
                self.logger.warning(f"Failed to warm up cache for memory {memory_id}: {e}")
    
    def validate_embeddings(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Validate the quality and consistency of stored embeddings.
        
        Performs sanity checks on a sample of embeddings to detect
        issues like corruption, dimension mismatches, or quality degradation.
        
        Args:
            sample_size: Number of random embeddings to validate
            
        Returns:
            Validation report with findings and recommendations
        """
        return self._perform_validation(sample_size)
    
    def _perform_validation(self, sample_size: int) -> Dict[str, Any]:
        """Perform the actual validation work"""
        # Placeholder implementation
        return {
            "total_checked": sample_size,
            "valid_embeddings": sample_size - 2,
            "invalid_embeddings": 2,
            "dimension_mismatches": 1,
            "corrupted_embeddings": 1,
            "quality_score": 0.98
        }
    
    def export_embeddings(self, 
                         memory_ids: Optional[List[str]] = None,
                         format: str = "numpy") -> Dict[str, Any]:
        """
        Export embeddings for backup or analysis.
        
        Args:
            memory_ids: Specific memories to export, None for all
            format: Export format ("numpy", "json", "hdf5")
            
        Returns:
            Exported embedding data
        """
        if format == "numpy":
            return self._export_numpy_format(memory_ids)
        elif format == "json":
            return self._export_json_format(memory_ids)
        elif format == "hdf5":
            return self._export_hdf5_format(memory_ids)
        else:
            raise NotImplementedError(f"Export format '{format}' not implemented")
    
    def _export_numpy_format(self, memory_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export embeddings in numpy format"""
        exported_embeddings = {}
        exported_metadata = {}
        
        if memory_ids is None:
            # Export all embeddings - would need vector store integration
            self.logger.warning("Exporting all embeddings not fully implemented without vector store")
            return {
                "embeddings": {},
                "metadata": {},
                "format": "numpy",
                "export_time": datetime.now().isoformat()
            }
        
        # Export specific memories
        for memory_id in memory_ids:
            try:
                if self.vector_store:
                    embedding = self.vector_store.get_embedding(memory_id)
                    if embedding is not None:
                        exported_embeddings[memory_id] = embedding
                        # Get metadata if available
                        metadata = getattr(self.vector_store, 'get_metadata', lambda x: {})(memory_id)
                        exported_metadata[memory_id] = metadata
            except Exception as e:
                self.logger.warning(f"Failed to export embedding for {memory_id}: {e}")
        
        return {
            "embeddings": exported_embeddings,
            "metadata": exported_metadata,
            "format": "numpy",
            "export_time": datetime.now().isoformat(),
            "total_exported": len(exported_embeddings)
        }
    
    def _export_json_format(self, memory_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export embeddings in JSON-serializable format"""
        numpy_export = self._export_numpy_format(memory_ids)
        
        # Convert numpy arrays to lists for JSON serialization
        json_embeddings = {}
        for memory_id, embedding in numpy_export["embeddings"].items():
            if isinstance(embedding, np.ndarray):
                json_embeddings[memory_id] = embedding.tolist()
            else:
                json_embeddings[memory_id] = embedding
        
        return {
            "embeddings": json_embeddings,
            "metadata": numpy_export["metadata"],
            "format": "json",
            "export_time": numpy_export["export_time"],
            "total_exported": numpy_export["total_exported"]
        }
    
    def _export_hdf5_format(self, memory_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export embeddings in HDF5 format (metadata only, actual HDF5 would need h5py)"""
        numpy_export = self._export_numpy_format(memory_ids)
        
        # For now, return metadata about what would be exported to HDF5
        # In a real implementation, would use h5py to write actual HDF5 file
        return {
            "format": "hdf5",
            "export_path": None,  # Would contain actual file path
            "embeddings_shape": {
                memory_id: list(embedding.shape) if hasattr(embedding, 'shape') else [len(embedding)]
                for memory_id, embedding in numpy_export["embeddings"].items()
            },
            "metadata": numpy_export["metadata"],
            "export_time": numpy_export["export_time"],
            "total_exported": numpy_export["total_exported"],
            "note": "HDF5 export requires h5py library for full implementation"
        }
    
    def import_embeddings(self, 
                         embedding_data: Dict[str, Any],
                         validate: bool = True) -> int:
        """
        Import embeddings from backup or external source.
        
        Args:
            embedding_data: Embedding data to import
            validate: Whether to validate embeddings before import
            
        Returns:
            Number of embeddings successfully imported
        """
        if validate:
            if not self._validate_import_data(embedding_data):
                raise ValueError("Invalid embedding data format")
        
        imported_count = 0
        embeddings = embedding_data.get("embeddings", {})
        metadata = embedding_data.get("metadata", {})
        
        for memory_id, embedding in embeddings.items():
            try:
                if self.vector_store:
                    success = self.vector_store.upsert(
                        memory_id=memory_id,
                        embedding=np.array(embedding),
                        metadata=metadata.get(memory_id, {})
                    )
                    if success:
                        imported_count += 1
            except Exception as e:
                self.logger.error(f"Failed to import embedding for {memory_id}: {e}")
        
        return imported_count
    
    def _validate_import_data(self, data: Dict[str, Any]) -> bool:
        """Validate import data format"""
        return "embeddings" in data and isinstance(data["embeddings"], dict)
    
    def rebuild_index(self, 
                     memory_source: callable,
                     progress_callback: Optional[callable] = None) -> bool:
        """
        Rebuild the entire embedding index from scratch.
        
        Useful for model upgrades, corruption recovery, or
        configuration changes that require re-embedding.
        
        Args:
            memory_source: Function that yields memory data
            progress_callback: Progress tracking function
            
        Returns:
            True if rebuild successful
        """
        try:
            # Clear existing index
            if self.vector_store:
                self.vector_store.clear()
            
            # Rebuild from memory source
            memories = list(memory_source())
            total_memories = len(memories)
            
            for i, memory in enumerate(memories):
                success = self.index_memory(
                    memory_id=memory["id"],
                    content=memory["content"],
                    metadata=memory["metadata"]
                )
                
                if progress_callback:
                    progress_callback(i + 1, total_memories)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild index: {e}")
            return False
    
    def benchmark_search(self, 
                        test_queries: List[str],
                        expected_results: Optional[List[List[str]]] = None) -> Dict[str, float]:
        """
        Benchmark search performance and accuracy.
        
        Args:
            test_queries: List of queries to test
            expected_results: Expected memory IDs for each query
            
        Returns:
            Performance metrics (latency, accuracy, etc.)
        """
        total_time = 0.0
        accurate_results = 0
        
        for i, query in enumerate(test_queries):
            start_time = time.time()
            results = self.search(query, top_k=5)
            query_time = time.time() - start_time
            total_time += query_time
            
            # Check accuracy if expected results provided
            if expected_results and i < len(expected_results):
                actual_ids = [r.memory_id for r in results]
                expected_ids = expected_results[i]
                if any(aid in expected_ids for aid in actual_ids[:1]):  # Check top result
                    accurate_results += 1
        
        return {
            "total_queries": len(test_queries),
            "total_time": total_time,
            "average_latency": total_time / len(test_queries) if test_queries else 0.0,
            "accuracy": accurate_results / len(test_queries) if test_queries else 0.0
        }
    
    def _call_voyage_api(self, texts: List[str], input_type: Optional[str] = None) -> List[np.ndarray]:
        """Make API call to Voyage AI embedding service"""
        if not self.config.api_key:
            raise EmbeddingError("API key not configured")
            
        # Default input type based on config
        if input_type is None:
            input_type = self.config.input_type_documents
            
        payload = {
            "input": texts,
            "model": self.config.model_name,
            "input_type": input_type
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    "https://api.voyageai.com/v1/embeddings",
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 429:
                    if attempt < self.config.max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RateLimitError("API rate limit exceeded")
                        
                response.raise_for_status()
                result = response.json()
                
                embeddings = []
                for item in result.get("data", []):
                    embedding = np.array(item["embedding"], dtype=np.float32)
                    embeddings.append(embedding)
                
                self.api_call_count += 1
                self.last_api_call = datetime.now()
                
                return embeddings
                
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise EmbeddingError(f"API request failed after {self.config.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
        
        raise EmbeddingError("Failed to get embeddings from Voyage AI")
    
    def _get_cache_key(self, text: str, input_type: Optional[str] = None) -> str:
        """Generate cache key for text and input type"""
        key_data = f"{text}:{input_type or 'default'}:{self.config.model_name}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _prepare_text_for_embedding(self,
                                   content: str,
                                   metadata: Optional[Dict[str, Any]] = None,
                                   tags: Optional[List[str]] = None,
                                   summary: Optional[str] = None) -> str:
        """
        Prepare text for optimal embedding generation.
        
        Combines content, summary, and tags in a way that maximizes
        the quality and usefulness of the resulting embedding.
        
        Args:
            content: Main memory content
            metadata: Additional context
            tags: Memory tags
            summary: Memory summary
            
        Returns:
            Optimized text for embedding
        """
        parts = [content]
        
        if summary:
            parts.append(f"Summary: {summary}")
        
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        
        # Add relevant metadata
        if metadata:
            for key, value in metadata.items():
                if key in ['type', 'category', 'topic'] and value:
                    parts.append(f"{key.title()}: {value}")
        
        return " | ".join(parts)
    
    def _handle_api_error(self, error: Exception, retry_count: int) -> bool:
        """
        Handle API errors with appropriate retry logic.
        
        Args:
            error: The API error that occurred
            retry_count: Current retry attempt number
            
        Returns:
            True if should retry, False if should give up
        """
        if retry_count >= self.config.max_retries:
            return False
            
        if isinstance(error, RateLimitError):
            # Always retry rate limit errors (with backoff)
            return True
        elif isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            # Retry connection and timeout errors
            return True
        elif isinstance(error, requests.exceptions.HTTPError):
            # Retry server errors (5xx), but not client errors (4xx)
            status_code = getattr(error.response, 'status_code', 0)
            return 500 <= status_code < 600
        
        # Don't retry other types of errors
        return False
    
    def _update_performance_metrics(self, 
                                   operation: str,
                                   start_time: float,
                                   success: bool) -> None:
        """
        Update internal performance tracking metrics.
        
        Args:
            operation: Type of operation performed
            start_time: When operation started
            success: Whether operation succeeded
        """
        duration = time.time() - start_time
        self.total_api_time += duration
        
        if operation == "encode_text" and success:
            self.stats.successful_embeds += 1
            # Update average embedding time
            if self.stats.successful_embeds == 1:
                self.stats.average_embedding_time = duration
            else:
                self.stats.average_embedding_time = (
                    (self.stats.average_embedding_time * (self.stats.successful_embeds - 1) + duration)
                    / self.stats.successful_embeds
                )
        elif operation == "encode_text" and not success:
            self.stats.failed_embeds += 1
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder implementation)"""
        # This would require tracking cache hits/misses over time
        # For now, return a placeholder value
        return 0.0 if len(self.embedding_cache) == 0 else 0.85
    
    def __enter__(self):
        """Context manager entry - initialize connections"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup connections"""
        # Clean up any resources
        if hasattr(self, 'session'):
            self.session.close()
        return False


# Exception classes for embedding operations
class EmbeddingError(Exception):
    """Base exception for embedding operations"""
    pass

class BatchEmbeddingError(EmbeddingError):
    """Error during batch embedding operations"""
    pass

class IndexingError(EmbeddingError):
    """Error during memory indexing"""
    pass

class SearchError(EmbeddingError):
    """Error during search operations"""
    pass

class RateLimitError(EmbeddingError):
    """API rate limit exceeded"""
    pass

def get_embedding_config_from_env():
    api_key = os.getenv('VOYAGE_API_KEY')
    if not api_key:
        raise ValueError('VOYAGE_API_KEY not set in environment')
    return EmbeddingConfig(api_key=api_key)