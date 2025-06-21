from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from collections import defaultdict, deque

import numpy as np

from .memory_node import MemoryNode, Connection, ConnectionType

logger = logging.getLogger(__name__)

class MemoryGraph:
    """
    Core memory graph that manages nodes and their connections.
    Handles creation, retrieval, and maintenance of the memory network.
    """
    
    def __init__(self, decay_rate: float = 0.01):
        """
        Initialize memory graph with decay settings
        
        Args:
            decay_rate: How much to weaken connections on each decay cycle
        """
        # TODO: Initialize data structures
        # - self.nodes: Dict[str, MemoryNode] = {}
        # - self.decay_rate = decay_rate
        # - self.max_connections_per_node = max_connections_per_node
        # - Create indexes for fast lookups:
        #   - self._concept_index: Dict[str, Set[str]] (concept -> node_ids)
        #   - self._keyword_index: Dict[str, Set[str]] (keyword -> node_ids) 
        #   - self._tag_index: Dict[str, Set[str]] (tag -> node_ids)
        
        self.nodes: Dict[str, MemoryNode] = {}
        self.decay_rate = decay_rate
        self._concept_index: Dict[str, Set[str]] = defaultdict(set)  # concept -> node_ids
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)  # keyword -> node_ids
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)      # tag -> node_ids

    def add_node(self, node: MemoryNode) -> str:
        """
        Add a new memory node to the graph and update indexes
        
        Args:
            node: MemoryNode to add
            
        Returns:
            node_id: The ID of the added node
            
        TODO:
        - Add node to self.nodes
        - Update concept/keyword/tag indexes
        - Log the addition
        """
        self.nodes[node.id] = node
        
        # Update indexes
        self._concept_index[node.concept.lower()].add(node.id)
        for keyword in node.keywords:
            self._keyword_index[keyword.lower()].add(node.id)
        for tag in node.tags:
            self._tag_index[tag.lower()].add(node.id)
        
        logger.info(f"Added node {node.id}: {node.concept}")
        return node.id
    
    def get_node(self, node_id: str):
        """
        Retrieve a node by ID and update its access metadata
        
        Args:
            node_id: ID of node to retrieve
            
        Returns:
            MemoryNode if found, None otherwise
            
        TODO:
        - Check if node exists
        - Call node.update_access() to increment access count
        - Return the node
        """
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.update_access()
            return node
        else:
            print('get_node returned None')
            return None
            
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and clean up all references to it
        
        Args:
            node_id: ID of node to remove
            
        Returns:
            True if removed, False if not found
            
        TODO:
        - Check if node exists
        - Remove from all indexes (concept, keyword, tag)
        - Remove all connections TO this node from other nodes
        - Remove the node itself from self.nodes
        - Log the removal
        """
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        # Remove from indexes
        self._concept_index[node.concept.lower()].discard(node_id)
        for keyword in node.keywords:
            self._keyword_index[keyword.lower()].discard(node_id)
        for tag in node.tags:
            self._tag_index[tag.lower()].discard(node_id)
        
        # Remove all connections TO this node from other nodes
        for other_node in self.nodes.values():
            if node_id in other_node.connections:
                del other_node.connections[node_id]
        
        # Remove the node itself
        del self.nodes[node_id]
        logger.info(f"Removed node {node_id}")
        return True
    
    def create_connection(self, source_id: str, target_id: str, 
                         connection_type: ConnectionType, initial_weight: float = 0.0) -> bool:
        """
        Create a connection between two nodes
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            connection_type: Type of connection
            initial_weight: Starting weight (default 0.0)
            
        Returns:
            True if created, False if nodes don't exist
            
        TODO:
        - Verify both nodes exist
        - Check if source has too many connections
        - If at max, remove weakest connection to make room
        - Add new connection using source_node.add_connection()
        - Log the creation
        """
        """Create a connection between two nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        source_node = self.nodes[source_id]
        
        source_node.add_connection(target_id, connection_type, initial_weight)
        logger.debug(f"Created connection: {source_id} -> {target_id} ({connection_type})")
        return True
    
    
    def strengthen_connection(self, source_id: str, target_id: str, 
                            strength_increment: float = 0.1) -> bool:
        """
        Strengthen a connection between nodes
        
        Args:
            source_id: Source node ID
            target_id: Target node ID  
            strength_increment: How much to increase weight
            
        Returns:
            True if strengthened, False if connection doesn't exist
            
        TODO:
        - Verify source node exists
        - Call source_node.strengthen_connection()
        - Return success status
        """
        if source_id not in self.nodes:
            return False
        
        self.nodes[source_id].strengthen_connection(target_id, strength_increment)
        return True
    
    
    def get_connected_nodes(self, node_id: str, max_depth: int = 2, 
                           min_weight: float = 0.1) -> List[Tuple[str, float, int]]:
        """
        Get all connected nodes within max_depth, ordered by connection strength
        
        Args:
            node_id: Starting node
            max_depth: How many hops to traverse
            min_weight: Minimum connection weight to follow
            
        Returns:
            List of (node_id, cumulative_weight, depth) tuples, ordered by weight
            
        TODO:
        - Use BFS/DFS to traverse connections
        - Track cumulative weights (multiply weights as you go deeper)
        - Apply depth penalty (e.g., weight * 0.7^depth)
        - Filter by min_weight
        - Sort by cumulative weight descending
        - Return list of tuples
        """
        if node_id not in self.nodes:
            return []
        
        visited = set()
        results = []
        queue = deque([(node_id, 1.0, 0)])  # (node_id, cumulative_weight, depth)
        
        while queue:
            current_id, cumulative_weight, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            if current_id != node_id:  # Don't include the starting node
                results.append((current_id, cumulative_weight, depth))
            
            if depth < max_depth:
                current_node = self.nodes.get(current_id)
                if current_node:
                    for connection in current_node.connections.values():
                        if (connection.weight >= min_weight and 
                            connection.target_node_id not in visited):
                            # Weight diminishes with depth
                            new_weight = cumulative_weight * connection.weight * (0.7 ** depth)
                            queue.append((connection.target_node_id, new_weight, depth + 1))
        
        # Sort by cumulative weight (strongest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def find_nodes_by_concept(self, concept: str) -> List[str]:
        """
        Find nodes with exact concept match
        
        Args:
            concept: Concept to search for
            
        Returns:
            List of matching node IDs
            
        TODO:
        - Look up concept.lower() in self._concept_index
        - Return list of node IDs
        """
        return list(self._concept_index.get(concept.lower(), set()))
    
    def find_nodes_by_keyword(self, keyword: str) -> List[str]:
        """
        Find nodes containing the keyword
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of matching node IDs
            
        TODO:
        - Look up keyword.lower() in self._keyword_index
        - Return list of node IDs
        """
        return list(self._keyword_index.get(keyword.lower(), set()))
    
    def find_nodes_by_tag(self, tag: str) -> List[str]:
        """
        Find nodes with the specified tag
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching node IDs
            
        TODO:
        - Look up tag.lower() in self._tag_index
        - Return list of node IDs
        """
        return list(self._tag_index.get(tag.lower(), set()))
    
    def find_similar_concepts(self, concept: str, threshold: float = 0.8) -> List[str]:
        """
        Find nodes with similar concepts using string similarity
        
        Args:
            concept: Concept to find similar matches for
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            List of similar node IDs
            
        TODO:
        - Iterate through all concepts in self._concept_index
        - Calculate similarity using _string_similarity helper
        - Return node IDs where similarity >= threshold
        """
        concept_lower = concept.lower()
        similar_nodes = []
        
        for stored_concept, node_ids in self._concept_index.items():
            # Simple similarity check - could be enhanced with embedding similarity
            if self._string_similarity(concept_lower, stored_concept) >= threshold:
                similar_nodes.extend(node_ids)
        
        return similar_nodes
    
    def apply_usage_based_decay(self, accessed_nodes: List[str]):
        """
        Apply decay to connections based on which nodes were accessed
        
        Args:
            accessed_nodes: List of recently accessed node IDs
            
        TODO:
        - For each node NOT in accessed_nodes:
        - Call node.weaken_all_connections(self.decay_rate)
        - This creates organic forgetting - unused connections decay
        """
        for node_id in np.array(self.nodes.keys):
            if node_id not in accessed_nodes:
               node = self.get_node(node_id)
               if node is not None:
                node.weaken_all_connections(self.decay_rate)
            
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current graph state
        
        Returns:
            Dictionary with graph metrics
            
        TODO:
        - Count total nodes, connections
        - Calculate average connections per node
        - Calculate average connection weight
        - Count connection types
        - Return as dictionary
        """
        total_connections = sum(len(node.connections) for node in self.nodes.values())
        
        connection_weights = []
        connection_types = defaultdict(int)
        
        for node in self.nodes.values():
            for connection in node.connections.values():
                connection_weights.append(connection.weight)
                connection_types[connection.connection_type] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_connections": total_connections,
            "avg_connections_per_node": total_connections / len(self.nodes) if self.nodes else 0,
            "avg_connection_weight": sum(connection_weights) / len(connection_weights) if connection_weights else 0,
            "connection_types": dict(connection_types),
            "concepts": len(self._concept_index),
            "keywords": len(self._keyword_index),
            "tags": len(self._tag_index)
        }
    
    def export_graph(self) -> Dict[str, Any]:
        """
        Export the entire graph as a dictionary for persistence
        
        Returns:
            Dictionary representation of the graph
            
        TODO:
        - Convert all nodes to dictionaries using node.to_dict()
        - Include metadata (decay_rate, max_connections, statistics)
        - Return complete graph data
        """
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "metadata": {
                "decay_rate": self.decay_rate,
                "statistics": self.get_graph_statistics()
            }
        }
    
    def import_graph(self, graph_data: Dict[str, Any]):
        """
        Import a graph from dictionary data
        
        Args:
            graph_data: Dictionary containing graph data
            
        TODO:
        - Clear existing graph (nodes and indexes)
        - Recreate nodes from data using MemoryNode.from_dict()
        - Rebuild indexes by calling add_node() for each
        - Import metadata settings
        - Log import completion
        """
        self.nodes.clear()
        self._concept_index.clear()
        self._keyword_index.clear()
        self._tag_index.clear()
        
        # Import nodes
        for node_data in graph_data["nodes"].values():
            node = MemoryNode.from_dict(node_data)
            self.add_node(node)
        
        # Import metadata if available
        if "metadata" in graph_data:
            metadata = graph_data["metadata"]
            self.decay_rate = metadata.get("decay_rate", self.decay_rate)
            self.max_connections_per_node = metadata.get("max_connections_per_node", self.max_connections_per_node)
        
        logger.info(f"Imported graph with {len(self.nodes)} nodes")
    
    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """
        Calculate simple string similarity using character overlap
        
        Args:
            s1, s2: Strings to compare
            
        Returns:
            Similarity score (0.0-1.0)
            
        TODO:
        - Convert strings to character sets
        - Calculate intersection and union
        - Return intersection/union ratio
        - Handle edge cases (empty strings)
        """
        if not s1 or not s2:
            return 0.0
        
        # Convert to sets of characters
        set1, set2 = set(s1), set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0