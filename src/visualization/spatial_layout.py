"""
3D Spatial Layout Engine for Memory Graph Visualization

This module provides algorithms for positioning memory nodes in 3D space
based on their semantic relationships, connection strengths, and usage patterns.
"""

import math
import random
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import numpy as np

from ..core.memory_node import MemoryNode, ConnectionType


class SpatialLayoutEngine:
    """
    Generates and manages 3D coordinates for memory nodes to create
    meaningful spatial representations of the memory graph.
    """
    
    def __init__(self, space_size: float = 100.0, clustering_strength: float = 0.7):
        """
        Initialize the spatial layout engine.
        
        Args:
            space_size: The size of the 3D space (cube with sides of this length)
            clustering_strength: How strongly related memories cluster together (0.0-1.0)
        """
        self.space_size = space_size
        self.clustering_strength = clustering_strength
        self.cluster_centers: Dict[str, Tuple[float, float, float]] = {}
        
    def generate_initial_layout(self, nodes: List[MemoryNode]) -> Dict[str, Tuple[float, float, float]]:
        """
        Generate initial 3D coordinates for all memory nodes using a multi-stage approach:
        1. Cluster by semantic similarity (tags/concepts)
        2. Apply force-directed layout within clusters
        3. Position clusters in 3D space
        
        Args:
            nodes: List of memory nodes to position
            
        Returns:
            Dictionary mapping node IDs to 3D coordinates
        """
        if not nodes:
            return {}
            
        # Stage 1: Cluster nodes by semantic similarity
        clusters = self._cluster_by_semantics(nodes)
        
        # Stage 2: Generate cluster centers in 3D space
        cluster_positions = self._generate_cluster_centers(clusters)
        
        # Stage 3: Position nodes within their clusters
        positions = {}
        for cluster_id, cluster_nodes in clusters.items():
            cluster_center = cluster_positions[cluster_id]
            node_positions = self._layout_cluster_nodes(cluster_nodes, cluster_center)
            positions.update(node_positions)
            
        return positions
    
    def _cluster_by_semantics(self, nodes: List[MemoryNode]) -> Dict[str, List[MemoryNode]]:
        """
        Group nodes into semantic clusters based on shared tags and concepts.
        """
        clusters = defaultdict(list)
        
        for node in nodes:
            # Primary clustering by tags
            if node.tags:
                primary_tag = node.tags[0]  # Use first tag as primary cluster
                cluster_key = f"tag:{primary_tag}"
            else:
                # Fallback to concept-based clustering
                concept_words = node.concept.lower().split()
                if concept_words:
                    cluster_key = f"concept:{concept_words[0]}"
                else:
                    cluster_key = "misc:uncategorized"
                    
            clusters[cluster_key].append(node)
            
        # Handle small clusters by merging them
        return self._merge_small_clusters(dict(clusters))
    
    def _merge_small_clusters(self, clusters: Dict[str, List[MemoryNode]], 
                            min_cluster_size: int = 2) -> Dict[str, List[MemoryNode]]:
        """
        Merge clusters that are too small into nearby larger clusters.
        """
        large_clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}
        small_clusters = {k: v for k, v in clusters.items() if len(v) < min_cluster_size}
        
        # Merge small clusters into a "misc" cluster
        if small_clusters:
            misc_nodes = []
            for nodes in small_clusters.values():
                misc_nodes.extend(nodes)
            if misc_nodes:
                large_clusters["misc:mixed"] = misc_nodes
                
        return large_clusters
    
    def _generate_cluster_centers(self, clusters: Dict[str, List[MemoryNode]]) -> Dict[str, Tuple[float, float, float]]:
        """
        Position cluster centers in 3D space using a spherical distribution.
        """
        cluster_ids = list(clusters.keys())
        num_clusters = len(cluster_ids)
        
        if num_clusters == 1:
            return {cluster_ids[0]: (0.0, 0.0, 0.0)}
            
        positions = {}
        
        # Use spherical distribution for cluster centers
        for i, cluster_id in enumerate(cluster_ids):
            if num_clusters <= 6:
                # For small numbers, use predefined positions
                positions[cluster_id] = self._get_predefined_position(i, num_clusters)
            else:
                # For larger numbers, use spherical distribution
                positions[cluster_id] = self._get_spherical_position(i, num_clusters)
                
        self.cluster_centers = positions
        return positions
    
    def _get_predefined_position(self, index: int, total: int) -> Tuple[float, float, float]:
        """
        Get predefined positions for small numbers of clusters.
        """
        radius = self.space_size * 0.3
        
        predefined = [
            (0.0, 0.0, 0.0),  # Center
            (radius, 0.0, 0.0),  # Right
            (-radius, 0.0, 0.0),  # Left
            (0.0, radius, 0.0),  # Up
            (0.0, -radius, 0.0),  # Down
            (0.0, 0.0, radius),  # Forward
            (0.0, 0.0, -radius),  # Back
        ]
        
        if index < len(predefined):
            return predefined[index]
        else:
            return self._get_spherical_position(index, total)
    
    def _get_spherical_position(self, index: int, total: int) -> Tuple[float, float, float]:
        """
        Generate position on a sphere using golden spiral distribution.
        """
        radius = self.space_size * 0.3
        
        # Golden ratio for even distribution
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        
        # Calculate spherical coordinates
        y = 1 - (index / float(total - 1)) * 2  # y goes from 1 to -1
        radius_at_y = math.sqrt(1 - y * y)
        
        theta = golden_angle * index
        
        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y
        
        return (x * radius, y * radius, z * radius)
    
    def _layout_cluster_nodes(self, nodes: List[MemoryNode], 
                            cluster_center: Tuple[float, float, float]) -> Dict[str, Tuple[float, float, float]]:
        """
        Position nodes within a cluster using force-directed layout.
        """
        if len(nodes) == 1:
            return {nodes[0].id: cluster_center}
            
        # Initialize positions randomly around cluster center
        positions = {}
        cluster_radius = min(self.space_size * 0.15, len(nodes) * 2)
        
        for i, node in enumerate(nodes):
            # Random position within cluster
            angle = (2 * math.pi * i) / len(nodes)
            radius = random.uniform(0, cluster_radius)
            height = random.uniform(-cluster_radius/2, cluster_radius/2)
            
            x = cluster_center[0] + radius * math.cos(angle)
            y = cluster_center[1] + height
            z = cluster_center[2] + radius * math.sin(angle)
            
            positions[node.id] = (x, y, z)
        
        # Apply force-directed refinement
        positions = self._apply_force_directed_layout(nodes, positions, cluster_center, cluster_radius)
        
        return positions
    
    def _apply_force_directed_layout(self, nodes: List[MemoryNode], 
                                   initial_positions: Dict[str, Tuple[float, float, float]],
                                   cluster_center: Tuple[float, float, float],
                                   cluster_radius: float) -> Dict[str, Tuple[float, float, float]]:
        """
        Refine positions using force-directed algorithm based on connections.
        """
        positions = initial_positions.copy()
        node_lookup = {node.id: node for node in nodes}
        
        # Parameters for force simulation
        iterations = 50
        cooling_factor = 0.95
        spring_length = cluster_radius * 0.3
        spring_strength = 0.1
        repulsion_strength = cluster_radius * 0.5
        
        for iteration in range(iterations):
            forces = defaultdict(lambda: [0.0, 0.0, 0.0])
            
            # Calculate forces between connected nodes (attraction)
            for node in nodes:
                node_pos = positions[node.id]
                
                for target_id, connection in node.connections.items():
                    if target_id in positions:
                        target_pos = positions[target_id]
                        
                        # Spring force based on connection weight
                        dx = target_pos[0] - node_pos[0]
                        dy = target_pos[1] - node_pos[1]
                        dz = target_pos[2] - node_pos[2]
                        
                        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                        if distance > 0:
                            force_magnitude = spring_strength * connection.weight * (distance - spring_length)
                            
                            forces[node.id][0] += (dx / distance) * force_magnitude
                            forces[node.id][1] += (dy / distance) * force_magnitude
                            forces[node.id][2] += (dz / distance) * force_magnitude
            
            # Calculate repulsion forces between all nodes
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    pos1 = positions[node1.id]
                    pos2 = positions[node2.id]
                    
                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    dz = pos1[2] - pos2[2]
                    
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if distance > 0:
                        force_magnitude = repulsion_strength / (distance * distance)
                        
                        forces[node1.id][0] += (dx / distance) * force_magnitude
                        forces[node1.id][1] += (dy / distance) * force_magnitude
                        forces[node1.id][2] += (dz / distance) * force_magnitude
                        
                        forces[node2.id][0] -= (dx / distance) * force_magnitude
                        forces[node2.id][1] -= (dy / distance) * force_magnitude
                        forces[node2.id][2] -= (dz / distance) * force_magnitude
            
            # Apply forces with cooling
            temperature = (1.0 - iteration / iterations) * cooling_factor
            
            for node in nodes:
                force = forces[node.id]
                current_pos = positions[node.id]
                
                # Apply force with temperature
                new_pos = (
                    current_pos[0] + force[0] * temperature,
                    current_pos[1] + force[1] * temperature,
                    current_pos[2] + force[2] * temperature
                )
                
                # Keep within cluster bounds
                new_pos = self._constrain_to_cluster(new_pos, cluster_center, cluster_radius)
                positions[node.id] = new_pos
        
        return positions
    
    def _constrain_to_cluster(self, position: Tuple[float, float, float],
                            cluster_center: Tuple[float, float, float],
                            cluster_radius: float) -> Tuple[float, float, float]:
        """
        Constrain a position to stay within cluster bounds.
        """
        dx = position[0] - cluster_center[0]
        dy = position[1] - cluster_center[1]
        dz = position[2] - cluster_center[2]
        
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if distance > cluster_radius:
            # Scale back to cluster boundary
            scale = cluster_radius / distance
            return (
                cluster_center[0] + dx * scale,
                cluster_center[1] + dy * scale,
                cluster_center[2] + dz * scale
            )
        
        return position
    
    def update_positions_on_access(self, accessed_node_id: str, 
                                 connected_nodes: List[str],
                                 all_positions: Dict[str, Tuple[float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
        """
        Dynamically adjust positions when nodes are accessed together.
        This creates subtle movements that show active connections.
        
        Args:
            accessed_node_id: The node that was accessed
            connected_nodes: List of connected node IDs
            all_positions: Current positions of all nodes
            
        Returns:
            Updated positions dictionary
        """
        updated_positions = all_positions.copy()
        
        if accessed_node_id not in all_positions:
            return updated_positions
            
        # Small movement towards accessed node for connected nodes
        accessed_pos = all_positions[accessed_node_id]
        movement_factor = 0.02  # Small movement (2% of distance)
        
        for connected_id in connected_nodes:
            if connected_id in all_positions:
                connected_pos = all_positions[connected_id]
                
                # Calculate movement vector
                dx = accessed_pos[0] - connected_pos[0]
                dy = accessed_pos[1] - connected_pos[1]
                dz = accessed_pos[2] - connected_pos[2]
                
                # Apply small movement
                new_pos = (
                    connected_pos[0] + dx * movement_factor,
                    connected_pos[1] + dy * movement_factor,
                    connected_pos[2] + dz * movement_factor
                )
                
                updated_positions[connected_id] = new_pos
        
        return updated_positions
    
    def get_cluster_info(self) -> Dict[str, Dict[str, any]]:
        """
        Get information about current clusters for debugging/analytics.
        """
        return {
            cluster_id: {
                "center": center,
                "type": cluster_id.split(":")[0],
                "name": cluster_id.split(":")[1] if ":" in cluster_id else cluster_id
            }
            for cluster_id, center in self.cluster_centers.items()
        }