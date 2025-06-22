"""
Graph Traversal Module - Association Engine for AI Memory System

This module handles navigation through the memory graph using learned edge weights.
It's the second phase of the hybrid retrieval system, exploring connections
between memories to find related content through associative pathways.
"""

from typing import List, Dict, Optional, Tuple, Set, Deque, Any, Generator
from dataclasses import dataclass, field
from collections import deque, defaultdict
import heapq
from datetime import datetime
from enum import Enum
import logging
import time
import random
import math


class TraversalStrategy(Enum):
    """Different graph traversal strategies"""
    WEIGHTED_BFS = "weighted_bfs"
    WEIGHTED_DFS = "weighted_dfs" 
    BIDIRECTIONAL = "bidirectional"
    BEST_FIRST = "best_first"
    RANDOM_WALK = "random_walk"


@dataclass
class TraversalResult:
    """Result from graph traversal operations"""
    memory_id: str
    connection_strength: float
    path: List[str]
    depth: int
    traversal_score: float
    discovery_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PathInfo:
    """Information about a path between memories"""
    source_id: str
    target_id: str
    path: List[str]
    total_weight: float
    hop_count: int
    bottleneck_weight: float  # Weakest connection in path
    avg_weight: float


@dataclass
class TraversalConfig:
    """Configuration for graph traversal operations"""
    max_depth: int = 3
    max_results: int = 20
    min_connection_weight: float = 0.1
    depth_decay_factor: float = 0.8
    exploration_width: int = 5  # Max neighbors to explore per node
    strategy: TraversalStrategy = TraversalStrategy.WEIGHTED_BFS
    include_reverse_edges: bool = True
    path_diversity_bonus: float = 0.1
    recency_bonus_days: int = 30


@dataclass
class TraversalStats:
    """Statistics for traversal operations"""
    total_traversals: int = 0
    avg_depth_explored: float = 0.0
    avg_results_found: int = 0
    avg_traversal_time: float = 0.0
    paths_discovered: int = 0
    unique_memories_explored: int = 0
    cache_hit_rate: float = 0.0


class GraphTraversal:
    """
    Graph traversal engine for memory association exploration.
    
    This class implements various algorithms to navigate the memory graph,
    following learned connections between memories to discover related content.
    It simulates human associative thinking by following connection weights
    that strengthen over time through usage patterns.
    
    Key responsibilities:
    - Traverse memory graph using edge weights
    - Find optimal paths between memories
    - Explore neighborhoods around seed memories
    - Handle different traversal strategies
    - Manage depth limits and performance
    - Cache traversal results for efficiency
    """
    
    def __init__(self, 
                 memory_graph,
                 config: TraversalConfig,
                 logger=None):
        """
        Initialize the graph traversal engine.
        
        Args:
            memory_graph: The main memory graph instance
            config: Traversal configuration
            logger: Logger for debugging and monitoring
        """
        self.graph = memory_graph
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.stats = TraversalStats()
        
        # Traversal caching
        self.path_cache = {}  # Cache for computed paths
        self.neighborhood_cache = {}  # Cache for neighborhood explorations
        self.cache_ttl = 300  # 5 minutes
        
        # Performance monitoring
        self.traversal_history = []
        self.hot_paths = defaultdict(int)  # Frequently used paths
        
    def explore_from_seeds(self,
                          seed_memory_ids: List[str],
                          max_results: Optional[int] = None,
                          strategy: Optional[TraversalStrategy] = None) -> List[TraversalResult]:
        """
        Explore the memory graph starting from seed memories.
        
        This is the main traversal method that starts from embedding search
        results and explores connected memories through learned associations.
        
        Args:
            seed_memory_ids: Starting points from embedding search
            max_results: Maximum memories to return (overrides config)
            strategy: Traversal strategy to use (overrides config)
            
        Returns:
            List of TraversalResult objects sorted by relevance
            
        Raises:
            TraversalError: If traversal fails
        """
        start_time = time.time()
        
        try:
            # Use provided values or defaults from config
            max_results = max_results or self.config.max_results
            strategy = strategy or self.config.strategy
            
            self.logger.debug(f"Starting traversal from {len(seed_memory_ids)} seeds using {strategy.value}")
            
            # Route to appropriate algorithm
            if strategy == TraversalStrategy.WEIGHTED_BFS:
                results = self.weighted_breadth_first_search(
                    seed_memory_ids, self.config.max_depth, max_results
                )
            elif strategy == TraversalStrategy.WEIGHTED_DFS:
                results = self.weighted_depth_first_search(
                    seed_memory_ids, self.config.max_depth, max_results
                )
            elif strategy == TraversalStrategy.BEST_FIRST:
                results = self._best_first_search(
                    seed_memory_ids, self.config.max_depth, max_results
                )
            elif strategy == TraversalStrategy.RANDOM_WALK:
                # Special case: random walk from each seed
                all_results = []
                for seed_id in seed_memory_ids:
                    walk_results = self.random_walk(seed_id, walk_length=10, num_walks=3)
                    all_results.extend(walk_results)
                results = self._deduplicate_and_rank(all_results, max_results)
            else:
                raise TraversalError(f"Unsupported traversal strategy: {strategy}")
            
            # Update statistics
            traversal_time = time.time() - start_time
            self._update_traversal_stats(len(results), traversal_time)
            
            self.logger.debug(f"Traversal completed: found {len(results)} memories in {traversal_time:.3f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Traversal failed from seeds {seed_memory_ids}: {e}")
            raise TraversalError(f"Exploration failed: {e}")
    
    def find_path(self,
                  source_id: str,
                  target_id: str,
                  max_hops: Optional[int] = None) -> Optional[PathInfo]:
        """
        Find the best path between two specific memories.
        
        Uses bidirectional search or A* to efficiently find the
        strongest connection path between two memories.
        
        Args:
            source_id: Starting memory ID
            target_id: Destination memory ID  
            max_hops: Maximum path length to consider
            
        Returns:
            PathInfo object with path details, or None if no path exists
        """
        # Check cache first
        cache_key = f"{source_id}->{target_id}"
        if cache_key in self.path_cache:
            cache_entry = self.path_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                return cache_entry['path_info']
        
        max_hops = max_hops or self.config.max_depth
        
        try:
            # Use bidirectional search for efficiency
            path_info = self.bidirectional_search(source_id, target_id, max_hops)
            
            # Cache the result
            if path_info:
                self.path_cache[cache_key] = {
                    'path_info': path_info,
                    'timestamp': time.time()
                }
                
                # Track frequently used paths
                path_key = "->".join(path_info.path)
                self.hot_paths[path_key] += 1
                
            return path_info
            
        except Exception as e:
            self.logger.error(f"Path finding failed from {source_id} to {target_id}: {e}")
            return None
    
    def find_all_paths(self,
                      source_id: str,
                      target_id: str,
                      max_paths: int = 3,
                      max_hops: int = 4) -> List[PathInfo]:
        """
        Find multiple paths between two memories.
        
        Discovers diverse paths to understand different types of
        relationships between memories.
        
        Args:
            source_id: Starting memory ID
            target_id: Destination memory ID
            max_paths: Maximum number of paths to find
            max_hops: Maximum length for each path
            
        Returns:
            List of PathInfo objects, sorted by strength
        """
        paths = []
        explored_routes = set()
        
        # Try to find diverse paths by temporarily removing edges
        for attempt in range(max_paths * 2):  # Try more attempts than needed
            try:
                # Find a path avoiding previously found routes
                path_info = self._find_path_avoiding_routes(
                    source_id, target_id, explored_routes, max_hops
                )
                
                if path_info and len(paths) < max_paths:
                    paths.append(path_info)
                    # Mark this route as explored
                    route_key = "->".join(path_info.path[1:-1])  # Exclude start and end
                    explored_routes.add(route_key)
                elif not path_info:
                    break  # No more paths available
                    
            except Exception as e:
                self.logger.debug(f"Path finding attempt {attempt} failed: {e}")
                continue
        
        # Sort by total weight (strongest paths first)
        paths.sort(key=lambda p: p.total_weight, reverse=True)
        return paths[:max_paths]
    
    def explore_neighborhood(self,
                           center_id: str,
                           radius: int = 2,
                           min_strength: Optional[float] = None) -> List[TraversalResult]:
        """
        Explore the immediate neighborhood around a memory.
        
        Gets all memories within a specified radius (hop count)
        of the center memory, useful for understanding local context.
        
        Args:
            center_id: Memory to explore around
            radius: Maximum hops from center
            min_strength: Minimum connection strength to follow
            
        Returns:
            List of memories in the neighborhood
        """
        # Check cache
        cache_key = f"neighborhood:{center_id}:{radius}:{min_strength}"
        if cache_key in self.neighborhood_cache:
            cache_entry = self.neighborhood_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                return cache_entry['results']
        
        min_strength = min_strength or self.config.min_connection_weight
        
        results = []
        visited = {center_id}
        queue = deque([(center_id, 0, [center_id])])  # (node_id, depth, path)
        
        while queue:
            current_id, depth, path = queue.popleft()
            
            if depth >= radius:
                continue
                
            # Get connections from current node
            try:
                connections = self.graph.get_connections(current_id)
                if self.config.include_reverse_edges:
                    # Also get reverse connections
                    reverse_connections = self.graph.get_reverse_connections(current_id)
                    connections.extend(reverse_connections)
                
                for neighbor_id, weight in connections:
                    if neighbor_id not in visited and weight >= min_strength:
                        visited.add(neighbor_id)
                        new_path = path + [neighbor_id]
                        
                        # Create traversal result
                        traversal_score = self._calculate_traversal_score(
                            neighbor_id, weight, depth + 1, new_path
                        )
                        
                        result = TraversalResult(
                            memory_id=neighbor_id,
                            connection_strength=weight,
                            path=new_path,
                            depth=depth + 1,
                            traversal_score=traversal_score,
                            discovery_method="neighborhood_exploration"
                        )
                        
                        results.append(result)
                        
                        # Add to queue for further exploration
                        queue.append((neighbor_id, depth + 1, new_path))
                        
            except Exception as e:
                self.logger.debug(f"Failed to get connections for {current_id}: {e}")
                continue
        
        # Sort by traversal score
        results.sort(key=lambda r: r.traversal_score, reverse=True)
        
        # Cache the results
        self.neighborhood_cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }
        
        return results
    
    def weighted_breadth_first_search(self,
                                    start_nodes: List[str],
                                    max_depth: int,
                                    max_results: int) -> List[TraversalResult]:
        """
        Perform weighted BFS prioritizing stronger connections.
        
        Uses a priority queue to explore strongest connections first,
        implementing the core associative memory retrieval algorithm.
        
        Args:
            start_nodes: Starting memory IDs
            max_depth: Maximum traversal depth
            max_results: Maximum results to return
            
        Returns:
            Ordered list of discovered memories
        """
        results = []
        visited = set(start_nodes)
        
        # Priority queue: (-score, memory_id, depth, path, connection_strength)
        priority_queue = []
        
        # Initialize with start nodes
        for start_id in start_nodes:
            heapq.heappush(priority_queue, (
                0.0,  # No score penalty for seeds
                start_id,
                0,
                [start_id],
                1.0  # Perfect strength for seeds
            ))
        
        while priority_queue and len(results) < max_results:
            neg_score, current_id, depth, path, strength = heapq.heappop(priority_queue)
            
            # Skip if we've reached max depth
            if depth >= max_depth:
                continue
            
            # Get connections from current memory
            try:
                connections = self.graph.get_connections(current_id)
                if self.config.include_reverse_edges:
                    reverse_connections = self.graph.get_reverse_connections(current_id)
                    connections.extend(reverse_connections)
                
                # Sort connections by weight and take top K
                connections.sort(key=lambda x: x[1], reverse=True)
                connections = connections[:self.config.exploration_width]
                
                for neighbor_id, weight in connections:
                    if (neighbor_id not in visited and 
                        weight >= self.config.min_connection_weight and
                        self._should_explore_connection(current_id, neighbor_id, weight, depth)):
                        
                        visited.add(neighbor_id)
                        new_path = path + [neighbor_id]
                        
                        # Calculate traversal score
                        traversal_score = self._calculate_traversal_score(
                            neighbor_id, weight, depth + 1, new_path
                        )
                        
                        # Create result
                        result = TraversalResult(
                            memory_id=neighbor_id,
                            connection_strength=weight,
                            path=new_path,
                            depth=depth + 1,
                            traversal_score=traversal_score,
                            discovery_method="weighted_bfs"
                        )
                        
                        results.append(result)
                        
                        # Add to priority queue for further exploration
                        heapq.heappush(priority_queue, (
                            -traversal_score,  # Negative for max heap behavior
                            neighbor_id,
                            depth + 1,
                            new_path,
                            weight
                        ))
                        
            except Exception as e:
                self.logger.debug(f"Failed to get connections for {current_id}: {e}")
                continue
        
        # Sort results by traversal score
        results.sort(key=lambda r: r.traversal_score, reverse=True)
        return results[:max_results]
    
    def weighted_depth_first_search(self,
                                   start_nodes: List[str],
                                   max_depth: int,
                                   max_results: int) -> List[TraversalResult]:
        """
        Perform weighted DFS following strongest paths deeply.
        
        Explores strong connection chains in depth before
        exploring alternative paths.
        
        Args:
            start_nodes: Starting memory IDs
            max_depth: Maximum traversal depth
            max_results: Maximum results to return
            
        Returns:
            Ordered list of discovered memories
        """
        results = []
        visited = set()
        
        def dfs_explore(current_id: str, depth: int, path: List[str], 
                       connection_strength: float) -> None:
            if len(results) >= max_results or depth >= max_depth:
                return
                
            visited.add(current_id)
            
            # If not a start node, add to results
            if depth > 0:
                traversal_score = self._calculate_traversal_score(
                    current_id, connection_strength, depth, path
                )
                
                result = TraversalResult(
                    memory_id=current_id,
                    connection_strength=connection_strength,
                    path=path.copy(),
                    depth=depth,
                    traversal_score=traversal_score,
                    discovery_method="weighted_dfs"
                )
                
                results.append(result)
            
            # Get and sort connections by weight
            try:
                connections = self.graph.get_connections(current_id)
                if self.config.include_reverse_edges:
                    reverse_connections = self.graph.get_reverse_connections(current_id)
                    connections.extend(reverse_connections)
                
                connections.sort(key=lambda x: x[1], reverse=True)
                
                # Explore strongest connections first
                for neighbor_id, weight in connections:
                    if (neighbor_id not in visited and 
                        weight >= self.config.min_connection_weight and
                        len(results) < max_results):
                        
                        new_path = path + [neighbor_id]
                        dfs_explore(neighbor_id, depth + 1, new_path, weight)
                        
            except Exception as e:
                self.logger.debug(f"Failed to get connections for {current_id}: {e}")
            
            visited.remove(current_id)  # Allow revisiting in different paths
        
        # Start DFS from each seed
        for start_id in start_nodes:
            if len(results) < max_results:
                dfs_explore(start_id, 0, [start_id], 1.0)
        
        # Sort results by traversal score
        results.sort(key=lambda r: r.traversal_score, reverse=True)
        return results[:max_results]
    
    def bidirectional_search(self,
                           source_id: str,
                           target_id: str,
                           max_depth: int = 4) -> Optional[PathInfo]:
        """
        Find path between memories using bidirectional search.
        
        Searches from both ends simultaneously until they meet,
        efficient for finding connections in large graphs.
        
        Args:
            source_id: Starting memory
            target_id: Target memory
            max_depth: Maximum total path length
            
        Returns:
            PathInfo if path found, None otherwise
        """
        if source_id == target_id:
            return PathInfo(
                source_id=source_id,
                target_id=target_id,
                path=[source_id],
                total_weight=1.0,
                hop_count=0,
                bottleneck_weight=1.0,
                avg_weight=1.0
            )
        
        # Forward and backward search states
        forward_visited = {source_id: (0, [source_id], 1.0)}  # depth, path, weight
        backward_visited = {target_id: (0, [target_id], 1.0)}
        
        forward_queue = deque([(source_id, 0, [source_id], 1.0)])
        backward_queue = deque([(target_id, 0, [target_id], 1.0)])
        
        max_half_depth = max_depth // 2
        
        while forward_queue or backward_queue:
            # Forward search step
            if forward_queue:
                current_id, depth, path, path_weight = forward_queue.popleft()
                
                if depth < max_half_depth:
                    try:
                        connections = self.graph.get_connections(current_id)
                        for neighbor_id, weight in connections:
                            if weight >= self.config.min_connection_weight:
                                new_path = path + [neighbor_id]
                                new_weight = path_weight * weight
                                
                                # Check if we've met the backward search
                                if neighbor_id in backward_visited:
                                    backward_depth, backward_path, backward_weight = backward_visited[neighbor_id]
                                    # Construct full path
                                    full_path = new_path + backward_path[::-1][1:]  # Reverse and skip duplicate
                                    total_weight = new_weight * backward_weight
                                    
                                    return self._create_path_info(
                                        source_id, target_id, full_path, total_weight
                                    )
                                
                                # Continue forward search
                                if neighbor_id not in forward_visited:
                                    forward_visited[neighbor_id] = (depth + 1, new_path, new_weight)
                                    forward_queue.append((neighbor_id, depth + 1, new_path, new_weight))
                                    
                    except Exception as e:
                        self.logger.debug(f"Forward search failed at {current_id}: {e}")
            
            # Backward search step
            if backward_queue:
                current_id, depth, path, path_weight = backward_queue.popleft()
                
                if depth < max_half_depth:
                    try:
                        # Get reverse connections for backward search
                        connections = self.graph.get_reverse_connections(current_id)
                        for neighbor_id, weight in connections:
                            if weight >= self.config.min_connection_weight:
                                new_path = path + [neighbor_id]
                                new_weight = path_weight * weight
                                
                                # Check if we've met the forward search
                                if neighbor_id in forward_visited:
                                    forward_depth, forward_path, forward_weight = forward_visited[neighbor_id]
                                    # Construct full path
                                    full_path = forward_path + new_path[::-1][1:]  # Reverse and skip duplicate
                                    total_weight = forward_weight * new_weight
                                    
                                    return self._create_path_info(
                                        source_id, target_id, full_path, total_weight
                                    )
                                
                                # Continue backward search
                                if neighbor_id not in backward_visited:
                                    backward_visited[neighbor_id] = (depth + 1, new_path, new_weight)
                                    backward_queue.append((neighbor_id, depth + 1, new_path, new_weight))
                                    
                    except Exception as e:
                        self.logger.debug(f"Backward search failed at {current_id}: {e}")
        
        return None  # No path found
    
    def random_walk(self,
                   start_id: str,
                   walk_length: int = 10,
                   num_walks: int = 5) -> List[TraversalResult]:
        """
        Perform random walks for serendipitous discovery.
        
        Randomly explores the graph with probability proportional
        to edge weights, useful for discovering unexpected connections.
        
        Args:
            start_id: Starting memory ID
            walk_length: Length of each walk
            num_walks: Number of walks to perform
            
        Returns:
            Memories discovered during walks
        """
        discovered = {}
        
        for walk_num in range(num_walks):
            current_id = start_id
            path = [current_id]
            
            for step in range(walk_length):
                try:
                    connections = self.graph.get_connections(current_id)
                    if not connections:
                        break
                    
                    # Choose next node with probability proportional to edge weight
                    weights = [weight for _, weight in connections]
                    total_weight = sum(weights)
                    
                    if total_weight == 0:
                        break
                    
                    # Weighted random selection
                    rand_val = random.random() * total_weight
                    cumulative_weight = 0
                    
                    for neighbor_id, weight in connections:
                        cumulative_weight += weight
                        if rand_val <= cumulative_weight:
                            current_id = neighbor_id
                            path.append(current_id)
                            
                            # Track discovery
                            if current_id != start_id and current_id not in discovered:
                                discovered[current_id] = {
                                    'connection_strength': weight,
                                    'path': path.copy(),
                                    'depth': step + 1,
                                    'walk_number': walk_num
                                }
                            break
                    
                except Exception as e:
                    self.logger.debug(f"Random walk failed at {current_id}: {e}")
                    break
        
        # Convert to TraversalResult objects
        results = []
        for memory_id, info in discovered.items():
            traversal_score = self._calculate_traversal_score(
                memory_id, info['connection_strength'], info['depth'], info['path']
            )
            
            result = TraversalResult(
                memory_id=memory_id,
                connection_strength=info['connection_strength'],
                path=info['path'],
                depth=info['depth'],
                traversal_score=traversal_score,
                discovery_method="random_walk",
                metadata={'walk_number': info['walk_number']}
            )
            
            results.append(result)
        
        # Sort by traversal score
        results.sort(key=lambda r: r.traversal_score, reverse=True)
        return results
    
    def get_strongest_connections(self,
                                memory_id: str,
                                top_k: int = 5,
                                direction: str = "both") -> List[Tuple[str, float]]:
        """
        Get the strongest direct connections for a memory.
        
        Args:
            memory_id: Memory to get connections for
            top_k: Number of top connections to return
            direction: "outbound", "inbound", or "both"
            
        Returns:
            List of (connected_memory_id, weight) tuples
        """
        connections = []
        
        try:
            if direction in ["outbound", "both"]:
                outbound = self.graph.get_connections(memory_id)
                connections.extend(outbound)
            
            if direction in ["inbound", "both"]:
                inbound = self.graph.get_reverse_connections(memory_id)
                connections.extend(inbound)
            
            # Remove duplicates and sort by weight
            unique_connections = {}
            for neighbor_id, weight in connections:
                if neighbor_id not in unique_connections or weight > unique_connections[neighbor_id]:
                    unique_connections[neighbor_id] = weight
            
            # Convert back to list and sort
            sorted_connections = [(k, v) for k, v in unique_connections.items()]
            sorted_connections.sort(key=lambda x: x[1], reverse=True)
            
            return sorted_connections[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to get connections for {memory_id}: {e}")
            return []
    
    def get_connection_strength(self,
                              source_id: str,
                              target_id: str) -> Optional[float]:
        """
        Get the connection strength between two specific memories.
        
        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            
        Returns:
            Connection weight or None if not connected
        """
        try:
            return self.graph.get_connection_weight(source_id, target_id)
        except Exception as e:
            self.logger.debug(f"Failed to get connection strength {source_id}->{target_id}: {e}")
            return None
    
    def find_bridge_memories(self,
                           cluster_a: List[str],
                           cluster_b: List[str],
                           max_bridges: int = 3) -> List[str]:
        """
        Find memories that bridge between two clusters.
        
        Identifies memories that are well-connected to both
        clusters, useful for finding conceptual links.
        
        Args:
            cluster_a: First cluster of memory IDs
            cluster_b: Second cluster of memory IDs
            max_bridges: Maximum bridge memories to return
            
        Returns:
            List of bridge memory IDs
        """
        bridge_scores = {}
        
        # Get all possible bridge candidates (memories connected to either cluster)
        candidates = set()
        
        for memory_id in cluster_a + cluster_b:
            try:
                connections = self.graph.get_connections(memory_id)
                reverse_connections = self.graph.get_reverse_connections(memory_id)
                
                for neighbor_id, _ in connections + reverse_connections:
                    if neighbor_id not in cluster_a and neighbor_id not in cluster_b:
                        candidates.add(neighbor_id)
                        
            except Exception as e:
                self.logger.debug(f"Failed to get connections for {memory_id}: {e}")
                continue
        
        # Score each candidate based on connections to both clusters
        for candidate_id in candidates:
            try:
                connections_a = 0
                connections_b = 0
                total_weight_a = 0.0
                total_weight_b = 0.0
                
                # Check connections to candidate
                all_connections = self.graph.get_connections(candidate_id)
                all_connections.extend(self.graph.get_reverse_connections(candidate_id))
                
                for neighbor_id, weight in all_connections:
                    if neighbor_id in cluster_a:
                        connections_a += 1
                        total_weight_a += weight
                    elif neighbor_id in cluster_b:
                        connections_b += 1
                        total_weight_b += weight
                
                # Bridge score based on connections to both clusters
                if connections_a > 0 and connections_b > 0:
                    bridge_score = (
                        math.sqrt(connections_a * connections_b) *
                        (total_weight_a + total_weight_b) / (connections_a + connections_b)
                    )
                    bridge_scores[candidate_id] = bridge_score
                    
            except Exception as e:
                self.logger.debug(f"Failed to score bridge candidate {candidate_id}: {e}")
                continue
        
        # Sort by bridge score and return top bridges
        sorted_bridges = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)
        return [bridge_id for bridge_id, _ in sorted_bridges[:max_bridges]]
    
    def find_central_memories(self,
                            memory_ids: List[str],
                            centrality_metric: str = "weighted_degree") -> List[Tuple[str, float]]:
        """
        Find the most central memories in a subgraph.
        
        Calculates centrality metrics to identify important
        memories within a subset of the graph.
        
        Args:
            memory_ids: Subgraph to analyze
            centrality_metric: "weighted_degree", "betweenness", "eigenvector"
            
        Returns:
            List of (memory_id, centrality_score) tuples
        """
        centrality_scores = {}
        memory_set = set(memory_ids)
        
        if centrality_metric == "weighted_degree":
            # Calculate weighted degree centrality
            for memory_id in memory_ids:
                try:
                    total_weight = 0.0
                    connections = self.graph.get_connections(memory_id)
                    reverse_connections = self.graph.get_reverse_connections(memory_id)
                    
                    for neighbor_id, weight in connections + reverse_connections:
                        if neighbor_id in memory_set:
                            total_weight += weight
                    
                    centrality_scores[memory_id] = total_weight
                    
                except Exception as e:
                    self.logger.debug(f"Failed to calculate centrality for {memory_id}: {e}")
                    centrality_scores[memory_id] = 0.0
        
        elif centrality_metric == "betweenness":
            # Simplified betweenness centrality
            for memory_id in memory_ids:
                betweenness_score = 0.0
                
                # Count how many shortest paths go through this node
                for source_id in memory_ids:
                    for target_id in memory_ids:
                        if source_id != target_id and source_id != memory_id and target_id != memory_id:
                            # Check if shortest path goes through memory_id
                            path_info = self.find_path(source_id, target_id, max_hops=4)
                            if path_info and memory_id in path_info.path[1:-1]:
                                betweenness_score += 1.0
                
                centrality_scores[memory_id] = betweenness_score
        
        else:
            # Default to weighted degree if metric not supported
            return self.find_central_memories(memory_ids, "weighted_degree")
        
        # Sort by centrality score
        sorted_centrality = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_centrality
    
    def propagate_activation(self,
                           source_ids: List[str],
                           activation_strength: float = 1.0,
                           decay_rate: float = 0.8,
                           max_iterations: int = 5) -> Dict[str, float]:
        """
        Simulate spreading activation through the memory network.
        
        Models how activation spreads through connected memories,
        similar to neural network activation propagation.
        
        Args:
            source_ids: Initial activation sources
            activation_strength: Starting activation level
            decay_rate: How much activation decays per hop
            max_iterations: Maximum propagation steps
            
        Returns:
            Dict mapping memory_id to final activation level
        """
        # Initialize activation levels
        current_activation = {}
        for source_id in source_ids:
            current_activation[source_id] = activation_strength
        
        # Propagate activation iteratively
        for iteration in range(max_iterations):
            next_activation = current_activation.copy()
            
            for memory_id, activation in current_activation.items():
                if activation > 0.01:  # Only propagate significant activation
                    try:
                        connections = self.graph.get_connections(memory_id)
                        
                        for neighbor_id, weight in connections:
                            # Calculate activation to propagate
                            propagated_activation = activation * decay_rate * weight
                            
                            # Add to neighbor's activation (accumulating from multiple sources)
                            if neighbor_id in next_activation:
                                next_activation[neighbor_id] += propagated_activation
                            else:
                                next_activation[neighbor_id] = propagated_activation
                                
                    except Exception as e:
                        self.logger.debug(f"Activation propagation failed from {memory_id}: {e}")
                        continue
            
            current_activation = next_activation
        
        # Filter out very low activation levels
        filtered_activation = {
            memory_id: activation 
            for memory_id, activation in current_activation.items()
            if activation > 0.01
        }
        
        return filtered_activation
    
    def detect_communities(self,
                          memory_ids: Optional[List[str]] = None,
                          algorithm: str = "louvain") -> Dict[str, int]:
        """
        Detect communities/clusters in the memory graph.
        
        Groups memories into communities based on connection
        patterns, useful for understanding memory organization.
        
        Args:
            memory_ids: Subset to analyze (None for full graph)
            algorithm: Community detection algorithm to use
            
        Returns:
            Dict mapping memory_id to community_id
        """
        if memory_ids is None:
            try:
                memory_ids = self.graph.get_all_memory_ids()
            except Exception as e:
                self.logger.error(f"Failed to get all memory IDs: {e}")
                return {}
        
        # Simple community detection based on connection density
        # In a real implementation, you'd use algorithms like Louvain or Leiden
        
        communities = {}
        community_id = 0
        unassigned = set(memory_ids)
        
        while unassigned:
            # Start a new community with an unassigned node
            seed = unassigned.pop()
            current_community = {seed}
            communities[seed] = community_id
            
            # Grow community by adding strongly connected neighbors
            changed = True
            while changed:
                changed = False
                new_members = set()
                
                for member_id in current_community:
                    try:
                        connections = self.graph.get_connections(member_id)
                        reverse_connections = self.graph.get_reverse_connections(member_id)
                        
                        for neighbor_id, weight in connections + reverse_connections:
                            if (neighbor_id in unassigned and 
                                weight >= self.config.min_connection_weight * 2):  # Higher threshold for communities
                                new_members.add(neighbor_id)
                                
                    except Exception as e:
                        self.logger.debug(f"Community detection failed for {member_id}: {e}")
                        continue
                
                # Add new members to community
                for new_member in new_members:
                    if new_member in unassigned:
                        unassigned.remove(new_member)
                        current_community.add(new_member)
                        communities[new_member] = community_id
                        changed = True
            
            community_id += 1
        
        return communities
    
    def calculate_graph_metrics(self,
                              subgraph_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate various graph theory metrics.
        
        Computes metrics like clustering coefficient, average path length,
        density, etc. for analysis and optimization.
        
        Args:
            subgraph_ids: Subset to analyze (None for full graph)
            
        Returns:
            Dict with various graph metrics
        """
        if subgraph_ids is None:
            try:
                subgraph_ids = self.graph.get_all_memory_ids()
            except Exception as e:
                self.logger.error(f"Failed to get all memory IDs: {e}")
                return {}
        
        num_nodes = len(subgraph_ids)
        if num_nodes < 2:
            return {"num_nodes": num_nodes, "num_edges": 0}
        
        subgraph_set = set(subgraph_ids)
        num_edges = 0
        total_weight = 0.0
        degrees = defaultdict(int)
        
        # Count edges and calculate basic metrics
        for memory_id in subgraph_ids:
            try:
                connections = self.graph.get_connections(memory_id)
                
                for neighbor_id, weight in connections:
                    if neighbor_id in subgraph_set:
                        num_edges += 1
                        total_weight += weight
                        degrees[memory_id] += 1
                        
            except Exception as e:
                self.logger.debug(f"Metrics calculation failed for {memory_id}: {e}")
                continue
        
        # Calculate metrics
        max_possible_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0
        
        average_degree = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0.0
        average_weight = total_weight / num_edges if num_edges > 0 else 0.0
        
        # Calculate clustering coefficient (simplified)
        clustering_coefficient = 0.0
        if num_nodes > 2:
            triangles = 0
            possible_triangles = 0
            
            for memory_id in subgraph_ids:
                try:
                    neighbors = [n for n, _ in self.graph.get_connections(memory_id) if n in subgraph_set]
                    degree = len(neighbors)
                    
                    if degree >= 2:
                        possible_triangles += degree * (degree - 1) // 2
                        
                        # Count actual triangles
                        for i in range(len(neighbors)):
                            for j in range(i + 1, len(neighbors)):
                                if self.graph.has_connection(neighbors[i], neighbors[j]):
                                    triangles += 1
                                    
                except Exception as e:
                    self.logger.debug(f"Clustering calculation failed for {memory_id}: {e}")
                    continue
            
            clustering_coefficient = triangles / possible_triangles if possible_triangles > 0 else 0.0
        
        # Estimate average path length (sample-based for large graphs)
        average_path_length = self._estimate_average_path_length(subgraph_ids[:50])  # Sample
        
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": density,
            "average_degree": average_degree,
            "average_weight": average_weight,
            "clustering_coefficient": clustering_coefficient,
            "average_path_length": average_path_length,
            "diameter": min(average_path_length * 2, self.config.max_depth)  # Estimate
        }
    
    def get_traversal_statistics(self) -> TraversalStats:
        """
        Get comprehensive traversal statistics.
        
        Returns:
            TraversalStats object with performance metrics
        """
        return self.stats
    
    def clear_cache(self, max_age_seconds: Optional[int] = None) -> int:
        """
        Clear traversal caches.
        
        Args:
            max_age_seconds: Only clear entries older than this
            
        Returns:
            Number of cache entries cleared
        """
        cleared_count = 0
        current_time = time.time()
        
        if max_age_seconds is None:
            # Clear all caches
            cleared_count += len(self.path_cache)
            cleared_count += len(self.neighborhood_cache)
            self.path_cache.clear()
            self.neighborhood_cache.clear()
        else:
            # Clear old entries from path cache
            keys_to_remove = []
            for key, entry in self.path_cache.items():
                if current_time - entry['timestamp'] > max_age_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.path_cache[key]
                cleared_count += 1
            
            # Clear old entries from neighborhood cache
            keys_to_remove = []
            for key, entry in self.neighborhood_cache.items():
                if current_time - entry['timestamp'] > max_age_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.neighborhood_cache[key]
                cleared_count += 1
        
        return cleared_count
    
    def validate_graph_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of the memory graph.
        
        Checks for issues like orphaned nodes, invalid weights,
        bidirectional consistency problems, etc.
        
        Returns:
            Validation report with any issues found
        """
        report = {
            "total_nodes": 0,
            "total_edges": 0,
            "orphaned_nodes": 0,
            "invalid_weights": 0,
            "bidirectional_inconsistencies": 0,
            "self_loops": 0,
            "issues_found": [],
            "overall_health": "unknown"
        }
        
        try:
            all_memory_ids = self.graph.get_all_memory_ids()
            report["total_nodes"] = len(all_memory_ids)
            
            orphaned_nodes = []
            invalid_weights = []
            self_loops = []
            bidirectional_issues = []
            
            for memory_id in all_memory_ids:
                try:
                    connections = self.graph.get_connections(memory_id)
                    reverse_connections = self.graph.get_reverse_connections(memory_id)
                    
                    # Check for orphaned nodes
                    if not connections and not reverse_connections:
                        orphaned_nodes.append(memory_id)
                    
                    # Check connections
                    for neighbor_id, weight in connections:
                        report["total_edges"] += 1
                        
                        # Check for invalid weights
                        if weight < 0 or weight > 1 or not isinstance(weight, (int, float)):
                            invalid_weights.append((memory_id, neighbor_id, weight))
                        
                        # Check for self-loops
                        if neighbor_id == memory_id:
                            self_loops.append(memory_id)
                        
                        # Check bidirectional consistency
                        reverse_weight = self.graph.get_connection_weight(neighbor_id, memory_id)
                        if reverse_weight is not None and abs(weight - reverse_weight) > 0.1:
                            bidirectional_issues.append((memory_id, neighbor_id, weight, reverse_weight))
                    
                except Exception as e:
                    self.logger.debug(f"Validation failed for {memory_id}: {e}")
                    continue
            
            # Update report
            report["orphaned_nodes"] = len(orphaned_nodes)
            report["invalid_weights"] = len(invalid_weights)
            report["self_loops"] = len(self_loops)
            report["bidirectional_inconsistencies"] = len(bidirectional_issues)
            
            # Determine issues
            issues = []
            if orphaned_nodes:
                issues.append("orphaned_nodes")
            if invalid_weights:
                issues.append("invalid_weights")
            if self_loops:
                issues.append("self_loops")
            if bidirectional_issues:
                issues.append("bidirectional_inconsistencies")
            
            report["issues_found"] = issues
            
            # Overall health assessment
            if not issues:
                report["overall_health"] = "excellent"
            elif len(issues) == 1 and issues[0] in ["self_loops", "orphaned_nodes"]:
                report["overall_health"] = "good"
            elif len(issues) <= 2:
                report["overall_health"] = "fair"
            else:
                report["overall_health"] = "poor"
            
        except Exception as e:
            self.logger.error(f"Graph integrity validation failed: {e}")
            report["overall_health"] = "unknown"
        
        return report
    
    def _calculate_traversal_score(self,
                                 memory_id: str,
                                 connection_strength: float,
                                 depth: int,
                                 path: List[str]) -> float:
        """
        Calculate overall traversal score for ranking results.
        
        Combines connection strength, depth penalty, path diversity,
        and other factors into a single ranking score.
        """
        # Apply depth decay
        decayed_strength = self._apply_depth_decay(connection_strength, depth)
        
        # Path diversity bonus (longer unique paths get bonus)
        path_diversity = 1.0 + (len(set(path)) - 1) * self.config.path_diversity_bonus
        
        # Calculate final score
        score = decayed_strength * path_diversity
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _apply_depth_decay(self, strength: float, depth: int) -> float:
        """Apply depth-based decay to connection strength."""
        return strength * (self.config.depth_decay_factor ** depth)
    
    def _should_explore_connection(self,
                                 current_id: str,
                                 neighbor_id: str,
                                 weight: float,
                                 current_depth: int) -> bool:
        """Decide whether to explore a connection during traversal."""
        # Check weight threshold
        if weight < self.config.min_connection_weight:
            return False
        
        # Check depth limit
        if current_depth >= self.config.max_depth:
            return False
        
        # Additional heuristics could be added here
        # For example, avoiding recently traversed paths, etc.
        
        return True
    
    def _update_traversal_stats(self, results_found: int, traversal_time: float) -> None:
        """Update traversal statistics"""
        self.stats.total_traversals += 1
        
        # Update averages
        if self.stats.total_traversals == 1:
            self.stats.avg_results_found = results_found
            self.stats.avg_traversal_time = traversal_time
        else:
            alpha = 0.1  # Exponential moving average factor
            self.stats.avg_results_found = (
                (1 - alpha) * self.stats.avg_results_found + alpha * results_found
            )
            self.stats.avg_traversal_time = (
                (1 - alpha) * self.stats.avg_traversal_time + alpha * traversal_time
            )
    
    def _best_first_search(self, start_nodes: List[str], max_depth: int, max_results: int) -> List[TraversalResult]:
        """Best-first search using heuristic scoring"""
        # Similar to weighted BFS but with different scoring
        return self.weighted_breadth_first_search(start_nodes, max_depth, max_results)
    
    def _deduplicate_and_rank(self, results: List[TraversalResult], max_results: int) -> List[TraversalResult]:
        """Remove duplicates and rank results"""
        seen = set()
        unique_results = []
        
        for result in results:
            if result.memory_id not in seen:
                seen.add(result.memory_id)
                unique_results.append(result)
        
        # Sort by traversal score
        unique_results.sort(key=lambda r: r.traversal_score, reverse=True)
        return unique_results[:max_results]
    
    def _find_path_avoiding_routes(self, source_id: str, target_id: str, 
                                 avoided_routes: Set[str], max_hops: int) -> Optional[PathInfo]:
        """Find path while avoiding certain intermediate routes"""
        # Simplified implementation - would need more sophisticated path finding
        return self.bidirectional_search(source_id, target_id, max_hops)
    
    def _create_path_info(self, source_id: str, target_id: str, path: List[str], total_weight: float) -> PathInfo:
        """Create PathInfo object from path data"""
        hop_count = len(path) - 1
        
        # Calculate bottleneck weight (minimum weight in path)
        bottleneck_weight = 1.0
        path_weights = []
        
        for i in range(len(path) - 1):
            weight = self.graph.get_connection_weight(path[i], path[i + 1])
            if weight is not None:
                path_weights.append(weight)
                bottleneck_weight = min(bottleneck_weight, weight)
        
        avg_weight = sum(path_weights) / len(path_weights) if path_weights else 0.0
        
        return PathInfo(
            source_id=source_id,
            target_id=target_id,
            path=path,
            total_weight=total_weight,
            hop_count=hop_count,
            bottleneck_weight=bottleneck_weight,
            avg_weight=avg_weight
        )
    
    def _estimate_average_path_length(self, sample_nodes: List[str]) -> float:
        """Estimate average path length using a sample of nodes"""
        path_lengths = []
        
        for i, source_id in enumerate(sample_nodes[:20]):  # Limit sample size
            for target_id in sample_nodes[i+1:i+11]:  # Check up to 10 targets per source
                path_info = self.find_path(source_id, target_id, max_hops=6)
                if path_info:
                    path_lengths.append(path_info.hop_count)
        
        return sum(path_lengths) / len(path_lengths) if path_lengths else 0.0


# Exception classes for graph traversal
class TraversalError(Exception):
    """Base exception for graph traversal operations"""
    pass

class PathNotFoundError(TraversalError):
    """Path between memories not found"""
    pass

class GraphIntegrityError(TraversalError):
    """Graph integrity validation failed"""
    pass

class TraversalTimeoutError(TraversalError):
    """Traversal operation exceeded time limit"""
    pass