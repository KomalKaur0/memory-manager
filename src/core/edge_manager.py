from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ConnectionType(str, Enum):
    GENERAL_SPECIFIC = "general_specific"
    SPECIFIC_GENERAL = "specific_general"
    CAUSE_EFFECT = "cause_effect"
    EFFECT_CAUSE = "effect_cause"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"
    SIMILARITY = "similarity"
    CONTRAST = "contrast"
    CONTEXT = "context"

@dataclass
class Edge:
    source_id: str
    target_id: str
    connection_type: ConnectionType
    weight: float = 0.0
    usage_count: int = 0
    
    def strengthen(self, increment: float = 0.1) -> float:
        """Strengthen the edge and return new weight"""
        self.weight = min(1.0, self.weight + increment)
        self.usage_count += 1
        return self.weight
    
    def weaken(self, decrement: float) -> float:
        """Weaken the edge and return new weight"""
        self.weight = max(0.0, self.weight - decrement)
        return self.weight

class EdgeManager:
    """
    Manages all edges in the memory graph.
    Handles creation, strengthening, weakening, and querying of connections.
    """
    
    def __init__(self, strength_increment: float = 0.1, relative_decay: bool = True):
        self.edges: Dict[str, Edge] = {}  # edge_id -> Edge
        self.source_index: Dict[str, Set[str]] = {}  # source_id -> set of edge_ids
        self.target_index: Dict[str, Set[str]] = {}  # target_id -> set of edge_ids
        self.strength_increment = strength_increment
        self.relative_decay = relative_decay  # If True, weights decay relative to new activity
        
        # Track total activity for relative decay
        self.total_strengthening_events = 0
    
    def _generate_edge_id(self, source_id: str, target_id: str, connection_type: ConnectionType) -> str:
        """Generate unique edge ID"""
        return f"{source_id}->{target_id}:{connection_type.value}"
    
    def create_edge(self, source_id: str, target_id: str, connection_type: ConnectionType, 
                   initial_weight: float = 0.0) -> str:
        """Create a new edge between nodes"""
        edge_id = self._generate_edge_id(source_id, target_id, connection_type)
        
        # Check if edge already exists
        if edge_id in self.edges:
            logger.debug(f"Edge {edge_id} already exists")
            return edge_id
        
        # Create new edge
        edge = Edge(source_id, target_id, connection_type, initial_weight)
        self.edges[edge_id] = edge
        
        # Update indexes
        if source_id not in self.source_index:
            self.source_index[source_id] = set()
        self.source_index[source_id].add(edge_id)
        
        if target_id not in self.target_index:
            self.target_index[target_id] = set()
        self.target_index[target_id].add(edge_id)
        
        logger.debug(f"Created edge: {edge_id} with weight {initial_weight}")
        return edge_id
    
    def strengthen_edge(self, source_id: str, target_id: str, connection_type: ConnectionType,
                       increment: Optional[float] = None) -> bool:
        """Strengthen an existing edge"""
        edge_id = self._generate_edge_id(source_id, target_id, connection_type)
        
        if edge_id not in self.edges:
            # Create edge if it doesn't exist
            self.create_edge(source_id, target_id, connection_type)
        
        # Strengthen the edge
        increment = increment or self.strength_increment
        edge = self.edges[edge_id]
        old_weight = edge.weight
        new_weight = edge.strengthen(increment)
        
        self.total_strengthening_events += 1
        
        # Apply relative decay to other edges if enabled
        if self.relative_decay:
            self._apply_relative_decay(exclude_edge_id=edge_id)
        
        logger.debug(f"Strengthened edge {edge_id}: {old_weight:.3f} -> {new_weight:.3f}")
        return True
    
    def weaken_edge(self, edge_id: str, decrement: float) -> bool:
        """Weaken a specific edge"""
        if edge_id not in self.edges:
            return False
        
        edge = self.edges[edge_id]
        old_weight = edge.weight
        new_weight = edge.weaken(decrement)
        
        logger.debug(f"Weakened edge {edge_id}: {old_weight:.3f} -> {new_weight:.3f}")
        return True
    
    def _apply_relative_decay(self, exclude_edge_id: Optional[str] = None, 
                             decay_factor: float = 0.01):
        """Apply small decay to all edges except the excluded one"""
        for edge_id, edge in self.edges.items():
            if edge_id != exclude_edge_id and edge.weight > 0:
                edge.weaken(decay_factor)
    
    def get_outgoing_edges(self, node_id: str, min_weight: float = 0.0) -> List[Edge]:
        """Get all outgoing edges from a node, ordered by weight"""
        if node_id not in self.source_index:
            return []
        
        edges = []
        for edge_id in self.source_index[node_id]:
            edge = self.edges[edge_id]
            if edge.weight >= min_weight:
                edges.append(edge)
        
        # Sort by weight (strongest first)
        edges.sort(key=lambda e: e.weight, reverse=True)
        return edges
    
    def get_incoming_edges(self, node_id: str, min_weight: float = 0.0) -> List[Edge]:
        """Get all incoming edges to a node, ordered by weight"""
        if node_id not in self.target_index:
            return []
        
        edges = []
        for edge_id in self.target_index[node_id]:
            edge = self.edges[edge_id]
            if edge.weight >= min_weight:
                edges.append(edge)
        
        # Sort by weight (strongest first)
        edges.sort(key=lambda e: e.weight, reverse=True)
        return edges
    
    def get_all_edges_for_node(self, node_id: str, min_weight: float = 0.0) -> List[Edge]:
        """Get all edges (incoming and outgoing) for a node"""
        outgoing = self.get_outgoing_edges(node_id, min_weight)
        incoming = self.get_incoming_edges(node_id, min_weight)
        
        # Combine and deduplicate
        all_edges = outgoing + incoming
        seen_ids = set()
        unique_edges = []
        
        for edge in all_edges:
            edge_id = self._generate_edge_id(edge.source_id, edge.target_id, edge.connection_type)
            if edge_id not in seen_ids:
                seen_ids.add(edge_id)
                unique_edges.append(edge)
        
        # Sort by weight
        unique_edges.sort(key=lambda e: e.weight, reverse=True)
        return unique_edges
    
    def get_edge(self, source_id: str, target_id: str, connection_type: ConnectionType) -> Optional[Edge]:
        """Get a specific edge"""
        edge_id = self._generate_edge_id(source_id, target_id, connection_type)
        return self.edges.get(edge_id)
    
    def remove_edge(self, source_id: str, target_id: str, connection_type: ConnectionType) -> bool:
        """Remove a specific edge"""
        edge_id = self._generate_edge_id(source_id, target_id, connection_type)
        
        if edge_id not in self.edges:
            return False
        
        # Remove from indexes
        self.source_index[source_id].discard(edge_id)
        self.target_index[target_id].discard(edge_id)
        
        # Clean up empty sets
        if not self.source_index[source_id]:
            del self.source_index[source_id]
        if not self.target_index[target_id]:
            del self.target_index[target_id]
        
        # Remove edge
        del self.edges[edge_id]
        
        logger.debug(f"Removed edge: {edge_id}")
        return True
    
    def remove_all_edges_for_node(self, node_id: str) -> int:
        """Remove all edges connected to a node and return count removed"""
        edges_to_remove = []
        
        # Collect outgoing edges
        if node_id in self.source_index:
            for edge_id in self.source_index[node_id]:
                edge = self.edges[edge_id]
                edges_to_remove.append((edge.source_id, edge.target_id, edge.connection_type))
        
        # Collect incoming edges
        if node_id in self.target_index:
            for edge_id in self.target_index[node_id]:
                edge = self.edges[edge_id]
                edges_to_remove.append((edge.source_id, edge.target_id, edge.connection_type))
        
        # Remove all collected edges
        removed_count = 0
        for source_id, target_id, connection_type in edges_to_remove:
            if self.remove_edge(source_id, target_id, connection_type):
                removed_count += 1
        
        logger.info(f"Removed {removed_count} edges for node {node_id}")
        return removed_count
    
    def cleanup_weak_edges(self, threshold: float = 0.01) -> int:
        """Remove edges below threshold weight"""
        edges_to_remove = []
        
        for edge_id, edge in self.edges.items():
            if edge.weight < threshold:
                edges_to_remove.append((edge.source_id, edge.target_id, edge.connection_type))
        
        removed_count = 0
        for source_id, target_id, connection_type in edges_to_remove:
            if self.remove_edge(source_id, target_id, connection_type):
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} weak edges below threshold {threshold}")
        return removed_count
    
    def get_strongest_edges(self, limit: int = 10) -> List[Edge]:
        """Get the strongest edges in the entire graph"""
        all_edges = list(self.edges.values())
        all_edges.sort(key=lambda e: e.weight, reverse=True)
        return all_edges[:limit]
    
    def get_statistics(self):
        """Get statistics about the edge manager"""
        if not self.edges:
            return {
                "total_edges": 0,
                "avg_weight": 0.0,
                "max_weight": 0.0,
                "min_weight": 0.0,
                "connection_types": {},
                "total_strengthening_events": self.total_strengthening_events
            }
        
        weights = [edge.weight for edge in self.edges.values()]
        connection_types = {}
        
        for edge in self.edges.values():
            conn_type = edge.connection_type.value
            connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
        
        return {
            "total_edges": len(self.edges),
            "avg_weight": sum(weights) / len(weights),
            "max_weight": max(weights),
            "min_weight": min(weights),
            "connection_types": connection_types,
            "total_strengthening_events": self.total_strengthening_events,
            "nodes_with_outgoing": len(self.source_index),
            "nodes_with_incoming": len(self.target_index)
        }