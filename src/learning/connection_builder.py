# src/learning/connection_builder.py
"""
Automatically builds and strengthens connections between memories based on usage patterns.
This is the key component that makes the memory system "learn" from interactions.
"""

import logging
from typing import List, Set, Tuple, Optional
from src.core.memory_node import MemoryNode, ConnectionType
from src.core.memory_graph import MemoryGraph
from src.storage.graph_database import MemoryGraphDatabase
from src.storage.vector_store import MemoryVectorStore
from src.core.memory_node import MemoryNode, Connection, ConnectionType
from config.settings import Settings

logger = logging.getLogger(__name__)


class ConnectionBuilder:
    """Builds and strengthens memory connections based on usage patterns."""
    
    def __init__(self, graph_db: MemoryGraphDatabase, vector_store: MemoryVectorStore):
        """
        Initialize the connection builder.
        
        Args:
            graph_db: Graph database instance
            vector_store: Vector store instance for similarity calculations
        """
        self.graph_db = graph_db
        self.vector_store = vector_store
        self.similarity_threshold = Settings.import_similarity_threshold
        self.connection_increment = Settings.connection_strength_increment
    
    def process_query_results(self, query: str, used_memories: List[MemoryNode], 
                            response: str) -> None:
        """
        Process memories that were used in a query response to strengthen connections.
        
        Args:
            query: The original query
            used_memories: Memories that contributed to the response
            response: The generated response (optional, for creating new memories)
        """
        if len(used_memories) < 2:
            return  # Need at least 2 memories to create connections
        
        # Update access patterns
        for memory in used_memories:
            self._update_memory_access(memory)
        
        # Strengthen existing connections between used memories
        self._strengthen_coactivated_connections(used_memories)
        
        # Find and create new connections based on co-activation
        self._create_new_connections(used_memories)
        
        # If response provided, consider creating a new memory
        if response and Settings.enable_auto_connections:
            self._consider_new_memory(query, response, used_memories)
    
    def _update_memory_access(self, memory: MemoryNode) -> None:
        """Update access statistics for a memory."""
        memory.access_count += 1
        # memory.update_importance_score()
        self.graph_db.store_memory_node(memory)
    
    def _strengthen_coactivated_connections(self, memories: List[MemoryNode]) -> None:
        """
        Strengthen existing connections between memories that were used together.
        """
        # Create pairs of all memories used together
        for i, memory1 in enumerate(memories):
            for memory2 in memories[i+1:]:
                # Check for existing connections in both directions
                self._strengthen_connection_if_exists(memory1, memory2)
                self._strengthen_connection_if_exists(memory2, memory1)
    
    def _strengthen_connection_if_exists(self, from_node: MemoryNode, 
                                       to_node: MemoryNode) -> None:
        """Strengthen a connection if it exists."""
        connections = self.graph_db.get_connections_from_node(from_node.id)
        
        for id, c in connections:
            if from_node.id == to_node.id:
                # Strengthen the connection
                current_weight = c.weight
                new_weight = min(1.0, current_weight + self.connection_increment)
                c.weight = new_weight
                self.graph_db.store_connection(
                    from_node.id, to_node.id, c
                )
                logger.info(f"Strengthened connection {from_node.concept} -> "
                          f"{to_node.concept} to {new_weight:.3f}")
    
    def _create_new_connections(self, memories: List[MemoryNode]) -> None:
        """
        Create new connections between memories based on co-activation and similarity.
        """
        for i, memory1 in enumerate(memories):
            for memory2 in memories[i+1:]:
                # Skip if connection already exists
                if self._connection_exists(memory1, memory2):
                    continue
                
                # Determine connection type based on content
                conn_type = self._determine_connection_type(memory1, memory2)
                
                if conn_type:
                    # Calculate initial weight based on similarity
                    similarity = self._calculate_similarity(memory1, memory2)
                    
                    if similarity > self.similarity_threshold:
                        initial_weight = similarity * 0.5  # Start at half the similarity
                        
                        # Create bidirectional connections with appropriate types
                        self._create_bidirectional_connection(
                            memory1, memory2, conn_type, initial_weight
                        )
    
    def _connection_exists(self, node1: MemoryNode, node2: MemoryNode) -> bool:
        """Check if any connection exists between two nodes."""
        connections = self.graph_db.get_connections_from_node(node1.id)
        return any(target_id == node2.id for target_id, _ in connections)
    
    def _determine_connection_type(self, node1: MemoryNode, 
                                 node2: MemoryNode) -> Optional[ConnectionType]:
        """
        Determine the appropriate connection type between two memories.
        Uses simple heuristics based on content analysis.
        """
        # Check for general/specific relationships
        if self._is_general_specific(node1, node2):
            return ConnectionType.GENERAL_SPECIFIC
        elif self._is_general_specific(node2, node1):
            return ConnectionType.SPECIFIC_GENERAL
        
        # Check for high similarity
        similarity = self._calculate_similarity(node1, node2)
        if similarity > 0.8:
            return ConnectionType.SIMILARITY
        
        # Check for contrast (if tags indicate opposition)
        if self._is_contrast(node1, node2):
            return ConnectionType.CONTRAST
        
        # Default to context if memories were used together
        return ConnectionType.CONTEXT
    
    def _is_general_specific(self, general: MemoryNode, specific: MemoryNode) -> bool:
        """Check if one concept is more general than another."""
        # Simple heuristic: check if specific concept contains general concept
        return (general.concept.lower() in specific.concept.lower() and
                len(specific.concept) > len(general.concept))
    
    def _is_contrast(self, node1: MemoryNode, node2: MemoryNode) -> bool:
        """Check if two nodes represent contrasting concepts."""
        contrast_indicators = [
            ("advantage", "disadvantage"),
            ("pro", "con"),
            ("benefit", "drawback"),
            ("positive", "negative"),
            ("increase", "decrease")
        ]
        
        text1 = f"{node1.concept} {node1.summary}".lower()
        text2 = f"{node2.concept} {node2.summary}".lower()
        
        for ind1, ind2 in contrast_indicators:
            if (ind1 in text1 and ind2 in text2) or (ind2 in text1 and ind1 in text2):
                return True
        return False
    
    def _calculate_similarity(self, node1: MemoryNode, node2: MemoryNode) -> float:
        """Calculate semantic similarity between two nodes using embeddings."""
        try:
            # Get embeddings from vector store
            embedding1 = self.vector_store.get_embedding_by_memory_id(node1.id)
            embedding2 = self.vector_store.get_embedding_by_memory_id(node2.id)
            
            if embedding1 is not None and embedding2 is not None:
                # Cosine similarity
                import numpy as np
                dot_product = np.dot(np.array(embedding1.values), np.array(embedding2.values))
                norm1 = np.linalg.norm(np.array(embedding1.values))
                norm2 = np.linalg.norm(np.array(embedding2.values))
                return dot_product / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
        
        # Fallback to keyword overlap
        keywords1 = set(node1.keywords)
        keywords2 = set(node2.keywords)
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _create_bidirectional_connection(self, node1: MemoryNode, node2: MemoryNode,
                                       conn_type: ConnectionType, weight: float) -> None:
        """Create appropriate bidirectional connections based on type."""
        # Map connection types to their inverses
        inverse_map = {
            ConnectionType.GENERAL_SPECIFIC: ConnectionType.SPECIFIC_GENERAL,
            ConnectionType.SPECIFIC_GENERAL: ConnectionType.GENERAL_SPECIFIC,
            ConnectionType.CAUSE_EFFECT: ConnectionType.EFFECT_CAUSE,
            ConnectionType.EFFECT_CAUSE: ConnectionType.CAUSE_EFFECT,
            ConnectionType.TEMPORAL_BEFORE: ConnectionType.TEMPORAL_AFTER,
            ConnectionType.TEMPORAL_AFTER: ConnectionType.TEMPORAL_BEFORE,
            ConnectionType.SIMILARITY: ConnectionType.SIMILARITY,
            ConnectionType.CONTRAST: ConnectionType.CONTRAST,
            ConnectionType.CONTEXT: ConnectionType.CONTEXT
        }
        
        # Create forward connection
        c = Connection(target_node_id=node2.id, connection_type=conn_type, weight=weight)
        self.graph_db.store_connection(node1.id, node2.id, c)
        
        # Create inverse connection
        inverse_type = inverse_map.get(conn_type, conn_type)
        c_inv = Connection(target_node_id=node2.id, connection_type=inverse_type, weight=weight)
        self.graph_db.store_connection(node2.id, node1.id, c_inv)
        
        logger.info(f"Created new connection: {node1.concept} "
                   f"-[{conn_type.value}:{weight:.3f}]-> {node2.concept}")
    
    def _consider_new_memory(self, query: str, response: str, 
                           context_memories: List[MemoryNode]) -> Optional[MemoryNode]:
        """
        Consider creating a new memory from a query-response pair.
        """
        # Extract key concepts from query and response
        keywords = self._extract_keywords(query, response)
        
        # Check if this information is already captured
        if self._is_duplicate_information(keywords, context_memories):
            return None
        
        # Create new memory
        new_memory = MemoryNode(
            concept=self._generate_concept(query, response),
            keywords=keywords,
            summary=f"Q: {query[:100]}... A: {response[:100]}...",
            full_content=f"Query: {query}\n\nResponse: {response}",
            # tags={"generated", "query-response"}
            # metadata={
            #     "source": "query-response",
            #     "query": query,
            # }
        )
        
        # Store in database
        self.graph_db.store_memory_node(new_memory)
        
        # Create connections to context memories
        for context_memory in context_memories:
            self._create_bidirectional_connection(
                new_memory, context_memory,
                ConnectionType.CONTEXT,
                0.3  # Moderate initial weight
            )
        
        logger.info(f"Created new memory: {new_memory.concept}")
        return new_memory
    
    def _extract_keywords(self, query: str, response: str) -> List[str]:
        """Extract keywords from query and response."""
        # Simple extraction - in production, use NLP
        import re
        text = f"{query} {response}".lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Filter common words and return top keywords
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                       'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                       'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                       'does', 'did', 'will', 'would', 'could', 'should', 'may',
                       'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        keywords = [w for w in words if w not in common_words and len(w) > 3]
        
        # Return unique keywords, preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:10]  # Top 10 keywords
    
    def _generate_concept(self, query: str, response: str) -> str:
        """Generate a concept name for a query-response pair."""
        # Take first few significant words from query
        words = query.split()[:5]
        return " ".join(words)
    
    def _is_duplicate_information(self, keywords: List[str], 
                                 existing_memories: List[MemoryNode]) -> bool:
        """Check if the information already exists in memory."""
        keyword_set = set(keywords)
        
        for memory in existing_memories:
            memory_keywords = set(memory.keywords)
            overlap = len(keyword_set.intersection(memory_keywords))
            
            # If more than 70% overlap, consider it duplicate
            if overlap / len(keyword_set) > 0.7:
                return True
        
        return False