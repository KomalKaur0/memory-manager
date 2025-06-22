"""
Vector Store - Weaviate Integration
Handles storage and retrieval of memory embeddings for semantic search.
"""

from typing import List, Dict, Optional, Any, Tuple
import logging
# from datetime import datetime
import uuid

import weaviate
from weaviate.exceptions import WeaviateConnectionError
from sentence_transformers import SentenceTransformer
import numpy as np

from ..core.memory_node import MemoryNode
from ...config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MemoryVectorStore:
    """
    Weaviate vector database interface for storing and searching memory embeddings.
    """
    
    def __init__(
        self,
        weaviate_url: str,
        weaviate_api_key: str,
        embedding_model: str 
    ):
        """Initialize Weaviate connection and embedding model."""
        self.weaviate_url = weaviate_url or settings.weaviate_url
        self.weaviate_api_key = weaviate_api_key or settings.weaviate_api_key
        self.embedding_model_name = embedding_model or settings.embedding_model
        
        self.client = None  # Will be WeaviateClient after connection
        self.embedding_model: Optional[SentenceTransformer] = None
        self.class_name = "MemoryEmbedding" 
        
    def connect(self):
        """Establish connection to Weaviate and load embedding model."""
        try:
            # For Weaviate v4+ client
            import weaviate.classes as wvc
            
            if self.weaviate_api_key:
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.weaviate_url,
                    auth_credentials=wvc.init.Auth.api_key(self.weaviate_api_key)
                )
            else:
                # Extract host from URL for local connection
                host = self.weaviate_url.replace('http://', '').replace('https://', '').split(':')[0]
                port = 8080  # default port
                if ':' in self.weaviate_url.replace('http://', '').replace('https://', ''):
                    port = int(self.weaviate_url.split(':')[-1])
                
                self.client = weaviate.connect_to_local(
                    host=host,
                    port=port
                )
            
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            logger.info(f"Connected to Weaviate at {self.weaviate_url}")
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    
    def disconnect(self):
        """Close Weaviate connection."""
        if self.client:
            # Weaviate client doesn't have explicit disconnect
            self.client = None
            logger.info("Disconnected from Weaviate")
    
    def initialize_schema(self):
        """Create Weaviate schema for memory embeddings."""
        if not self.client:
            self.connect()
        
        if not self.client:
            raise RuntimeError("Failed to establish Weaviate connection")
        
        try:
            from weaviate.classes.config import Configure, Property, DataType
            from weaviate.exceptions import WeaviateConnectionError
            
            # Check if collection already exists
            if self.client.collections.exists(self.class_name):
                logger.info(f"Collection {self.class_name} already exists")
                return
            
            # Create collection with schema
            self.client.collections.create(
                name=self.class_name,
                description="Memory embeddings for semantic search",
                properties=[
                    Property(
                        name="memory_id",
                        data_type=DataType.TEXT,
                        description="Reference to memory node in Neo4j",
                        index_filterable=True,
                        index_searchable=True
                    ),
                    Property(
                        name="concept",
                        data_type=DataType.TEXT,
                        description="Main concept of the memory",
                        index_filterable=True,
                        index_searchable=True
                    ),
                    Property(
                        name="keywords",
                        data_type=DataType.TEXT_ARRAY,
                        description="Keywords associated with the memory",
                        index_filterable=True,
                        index_searchable=True
                    ),
                    Property(
                        name="tags",
                        data_type=DataType.TEXT_ARRAY,
                        description="Tags for categorization",
                        index_filterable=True,
                        index_searchable=True
                    ),
                    Property(
                        name="summary",
                        data_type=DataType.TEXT,
                        description="Brief summary of the memory",
                        index_searchable=True
                    ),
                    Property(
                        name="importance_score",
                        data_type=DataType.NUMBER,
                        description="Calculated importance score",
                        index_filterable=True
                    ),
                    Property(
                        name="created_at",
                        data_type=DataType.DATE,
                        description="When the memory was created",
                        index_filterable=True
                    ),
                    Property(
                        name="last_accessed",
                        data_type=DataType.DATE,
                        description="When the memory was last accessed",
                        index_filterable=True
                    ),
                    Property(
                        name="embedding_text",
                        data_type=DataType.TEXT,
                        description="The text that was embedded",
                        index_searchable=True
                    )
                ],
                # Configure for manual vector input
                vectorizer_config=Configure.Vectorizer.none()
            )
            
            logger.info(f"Created Weaviate collection: {self.class_name}")
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not loaded. Call connect() first.")
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def store_memory_embedding(self, memory_node: MemoryNode) -> bool:
        """
        Store or update memory embedding in Weaviate.
        
        Args:
            memory_node: MemoryNode to create embedding for
            
        Returns:
            bool: True if successful
        """
        if not self.client:
            self.connect()
        
        if not self.client:
            raise RuntimeError("Failed to establish Weaviate connection")
        
        try:
            # Create embedding text (combination of concept, summary, and keywords)
            embedding_text = f"{memory_node.concept}. {memory_node.summary}. {' '.join(memory_node.keywords)}"
            
            # Generate embedding
            vector = self.generate_embedding(embedding_text)
            
            # Prepare data object
            data_object = {
                "memory_id": memory_node.id,
                "concept": memory_node.concept,
                "keywords": memory_node.keywords,
                "tags": memory_node.tags,
                "summary": memory_node.summary,
                "importance_score": memory_node.importance_score,
                "embedding_text": embedding_text
            }
            
            # Get the collection
            collection = self.client.collections.get(self.class_name)
            
            # Check if embedding already exists
            existing = collection.query.fetch_object_by_id(memory_node.id)
            
            if existing:
                # Update existing embedding
                collection.data.update(
                    uuid=memory_node.id,
                    properties=data_object,
                    vector=vector
                )
                logger.debug(f"Updated embedding for memory: {memory_node.id}")
            else:
                # Create new embedding
                collection.data.insert(
                    properties=data_object,
                    vector=vector,
                    uuid=memory_node.id
                )
                logger.debug(f"Created embedding for memory: {memory_node.id}")
            
            return True
            
        except Exception as e:  # Changed from WeaviateException
            logger.error(f"Failed to store embedding for memory {memory_node.id}: {e}")
            return False
    
    def get_embedding_by_memory_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get embedding object by memory ID."""
        if not self.client:
            self.connect()
        
        if not self.client:
            raise RuntimeError("Failed to establish Weaviate connection")
        
        try:
            from weaviate.classes.query import Filter
            
            collection = self.client.collections.get(self.class_name)
            
            # Get all objects that match the filter
            response = collection.query.fetch_objects(
                limit=1
            )
            
            # Filter in memory since fetch_objects doesn't support where directly
            for obj in response.objects:
                if obj.properties.get("memory_id") == memory_id:
                    return {
                        **obj.properties,
                        "_additional": {"id": str(obj.uuid)}
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get embedding for memory {memory_id}: {e}")
            return None
    
    def semantic_search(
        self,
        query_text: str,
        limit: int = 10,
        min_certainty: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform semantic search for similar memories.
        """
        if not self.client:
            self.connect()
        
        if not self.client:
            raise RuntimeError("Failed to establish Weaviate connection")
        
        try:
            # Generate query embedding
            query_vector = self.generate_embedding(query_text)
            
            collection = self.client.collections.get(self.class_name)
            
            # Use hybrid search as an alternative
            response = collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=0.75,  # Weight towards vector search (0.5 = balanced, 1.0 = pure vector)
                limit=limit
            )
            
            # Process results
            results = []
            for obj in response.objects:
                memory_id = obj.properties.get("memory_id", "")
                certainty = min_certainty  # Default
                
                metadata = {
                    "concept": obj.properties.get("concept", ""),
                    "keywords": obj.properties.get("keywords", []),
                    "tags": obj.properties.get("tags", []),
                    "summary": obj.properties.get("summary", ""),
                    "importance_score": obj.properties.get("importance_score", 0.0),
                    "created_at": obj.properties.get("created_at"),
                    "last_accessed": obj.properties.get("last_accessed")
                }
                results.append((memory_id, certainty, metadata))
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def find_similar_memories(
        self,
        memory_id: str,
        limit: int = 5,
        min_certainty: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find memories similar to a given memory."""
        if not self.client:
            self.connect()
        
        try:
            # Get the reference memory's vector
            reference_obj = self.get_embedding_by_memory_id(memory_id)
            if not reference_obj:
                logger.warning(f"Memory {memory_id} not found in vector store")
                return []
            
            # Use the reference memory's embedding text for similarity search
            reference_text = reference_obj["embedding_text"]
            
            # Perform similarity search, excluding the reference memory itself
            filters = {
                "path": ["memory_id"],
                "operator": "NotEqual",
                "valueString": memory_id
            }
            
            return self.semantic_search(
                query_text=reference_text,
                limit=limit,
                min_certainty=min_certainty,
                filters=filters
            )
            
        except Exception as e:
            logger.error(f"Failed to find similar memories for {memory_id}: {e}")
            return []
    
    def delete_memory_embedding(self, memory_id: str) -> bool:
        """Delete memory embedding from Weaviate."""
        if not self.client:
            self.connect()
        
        if not self.client:
            raise RuntimeError("Failed to establish Weaviate connection")
        
        try:
            collection = self.client.collections.get(self.class_name)
            
            # First, find the object to get its UUID
            existing = self.get_embedding_by_memory_id(memory_id)
            if not existing:
                logger.warning(f"Memory embedding {memory_id} not found")
                return False
            
            # Delete using the UUID from the _additional field
            object_uuid = existing["_additional"]["id"]
            collection.data.delete_by_id(object_uuid)
            
            logger.info(f"Deleted embedding for memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embedding for memory {memory_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        if not self.client:
            self.connect()
        
        if not self.client:
            raise RuntimeError("Failed to establish Weaviate connection")
        
        try:
            collection = self.client.collections.get(self.class_name)
            
            # Get total count
            response = collection.aggregate.over_all()
            total_count = response.total_count if hasattr(response, 'total_count') else 0
            
            # Get embedding dimension by generating a test embedding
            embedding_dim = 0
            if self.embedding_model:
                try:
                    test_embedding = self.generate_embedding("test")
                    embedding_dim = len(test_embedding)
                except:
                    pass
            
            return {
                "total_embeddings": total_count,
                "avg_importance_score": 0.0,  # Would need separate query
                "embedding_dimension": embedding_dim,
                "model_name": self.embedding_model_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "total_embeddings": 0,
                "avg_importance_score": 0.0,
                "embedding_dimension": 0,
                "model_name": self.embedding_model_name
            }
    
    def batch_store_embeddings(self, memory_nodes: List[MemoryNode]) -> Dict[str, bool]:
        """
        Store multiple memory embeddings in batch for efficiency.
        
        Args:
            memory_nodes: List of MemoryNode objects
            
        Returns:
            Dict mapping memory_id to success status
        """
        if not self.client:
            self.connect()
        
        if not self.client:
            raise RuntimeError("Failed to establish Weaviate connection")
        
        results = {}
        
        try:
            collection = self.client.collections.get(self.class_name)
            
            # Prepare batch data
            batch_objects = []
            
            for memory_node in memory_nodes:
                try:
                    # Create embedding text
                    embedding_text = f"{memory_node.concept}. {memory_node.summary}. {' '.join(memory_node.keywords)}"
                    
                    # Generate embedding
                    vector = self.generate_embedding(embedding_text)
                    
                    # Prepare data object
                    data_object = {
                        "memory_id": memory_node.id,
                        "concept": memory_node.concept,
                        "keywords": memory_node.keywords,
                        "tags": memory_node.tags,
                        "summary": memory_node.summary,
                        "importance_score": memory_node.importance_score,
                        "embedding_text": embedding_text
                    }
                    
                    # Add to batch
                    batch_objects.append({
                        "properties": data_object,
                        "vector": vector,
                        "uuid": memory_node.id  # Use memory_id as UUID
                    })
                    
                    results[memory_node.id] = True
                    
                except Exception as e:
                    logger.error(f"Failed to prepare embedding for memory {memory_node.id}: {e}")
                    results[memory_node.id] = False
            
            # Insert batch
            if batch_objects:
                # Insert all objects at once
                response = collection.data.insert_many(batch_objects)
                
                # Check for any errors in the response
                if hasattr(response, 'errors') and response.errors:
                    for error in response.errors:
                        logger.error(f"Batch insert error: {error}")
            
            logger.info(f"Batch stored {sum(results.values())} out of {len(memory_nodes)} embeddings")
            
        except Exception as e:
            logger.error(f"Batch store operation failed: {e}")
            # Mark all as failed if batch operation fails
            for memory_node in memory_nodes:
                if memory_node.id not in results:
                    results[memory_node.id] = False
        
        return results