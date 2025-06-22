from typing import Dict, List, Tuple, Optional
import math
from enum import Enum

class WeightCalculationStrategy(str, Enum):
    """Different strategies for calculating connection weights"""
    USAGE_BASED = "usage_based"
    EMBEDDING_DISTANCE = "embedding_distance"
    HYBRID = "hybrid"
    FREQUENCY_DECAY = "frequency_decay"

class WeightCalculator:
    """
    Calculates and manages connection weights between memory nodes.
    Weights are based on usage patterns and naturally decay as new memories are accessed.
    """
    
    def __init__(self, 
                 base_increment: float = 0.1,
                 max_weight: float = 1.0,
                 min_weight: float = 0.0,
                 strategy: WeightCalculationStrategy = WeightCalculationStrategy.HYBRID):
        self.base_increment = base_increment
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.strategy = strategy
        
        # Track total system usage for relative decay
        self.total_access_count = 0
        self.connection_access_history: Dict[str, int] = {}  # connection_id -> last_access_count
    
    def calculate_initial_weight(self, 
                               embedding_distance: Optional[float] = None,
                               semantic_similarity: Optional[float] = None) -> float:
        """Calculate initial weight for a new connection"""
        if self.strategy == WeightCalculationStrategy.EMBEDDING_DISTANCE and embedding_distance is not None:
            # Convert distance to similarity (closer = higher weight)
            return max(self.min_weight, 1.0 - embedding_distance)
        
        elif semantic_similarity is not None:
            return max(self.min_weight, semantic_similarity)
        
        else:
            # Default starting weight
            return self.base_increment
    
    def strengthen_connection(self, 
                            current_weight: float,
                            usage_count: int,
                            connection_id: str) -> float:
        """
        Strengthen a connection based on usage.
        Uses diminishing returns - connections get harder to strengthen as they get stronger.
        """
        if self.strategy == WeightCalculationStrategy.USAGE_BASED:
            return self._usage_based_strengthening(current_weight, usage_count)
        
        elif self.strategy == WeightCalculationStrategy.FREQUENCY_DECAY:
            return self._frequency_decay_strengthening(current_weight, usage_count, connection_id)
        
        else:  # HYBRID or default
            return self._hybrid_strengthening(current_weight, usage_count, connection_id)
    
    def apply_relative_decay(self, 
                           connections: List[Tuple[str, float, int]]) -> List[Tuple[str, float, int]]:
        """
        Apply relative decay based on system-wide activity.
        Connections that haven't been used recently get relatively weaker as new ones are strengthened.
        
        Args:
            connections: List of (connection_id, current_weight, usage_count)
        
        Returns:
            List of (connection_id, new_weight, usage_count)
        """
        if not connections:
            return connections
        
        self.total_access_count += 1
        
        decayed_connections = []
        
        for connection_id, weight, usage_count in connections:
            # Check when this connection was last accessed relative to system activity
            last_access = self.connection_access_history.get(connection_id, 0)
            access_gap = self.total_access_count - last_access
            
            # Apply decay based on how long it's been since last access
            if access_gap > 0:
                # Exponential decay - longer gaps = more decay
                decay_factor = math.exp(-access_gap * 0.01)  # Adjust 0.01 for decay rate
                new_weight = max(self.min_weight, weight * decay_factor)
            else:
                new_weight = weight
            
            decayed_connections.append((connection_id, new_weight, usage_count))
        
        return decayed_connections
    
    def normalize_weights(self, 
                         connections: List[Tuple[str, float, int]],
                         preserve_top_n: int = 5) -> List[Tuple[str, float, int]]:
        """
        Normalize connection weights while preserving the strongest connections.
        This prevents weight inflation over time.
        """
        if not connections:
            return connections
        
        # Sort by weight (strongest first)
        sorted_connections = sorted(connections, key=lambda x: x[1], reverse=True)
        
        # Preserve top N connections at full strength
        preserved = sorted_connections[:preserve_top_n]
        to_normalize = sorted_connections[preserve_top_n:]
        
        if not to_normalize:
            return connections
        
        # Calculate normalization factor for remaining connections
        max_remaining_weight = max(conn[1] for conn in to_normalize)
        if max_remaining_weight > 0:
            normalization_factor = 0.8  # Cap remaining weights at 80% of max
            normalized = [
                (conn_id, min(weight * normalization_factor, self.max_weight), usage)
                for conn_id, weight, usage in to_normalize
            ]
        else:
            normalized = to_normalize
        
        # Combine preserved and normalized
        result = preserved + normalized
        
        # Sort back to original order if needed
        return result
    
    def update_access(self, connection_id: str):
        """Update access tracking for a connection"""
        self.connection_access_history[connection_id] = self.total_access_count
    
    def _usage_based_strengthening(self, current_weight: float, usage_count: int) -> float:
        """Simple usage-based strengthening with diminishing returns"""
        # Logarithmic growth - gets harder to strengthen as weight increases
        strength_multiplier = 1.0 / (1.0 + current_weight)
        increment = self.base_increment * strength_multiplier
        return min(self.max_weight, current_weight + increment)
    
    def _frequency_decay_strengthening(self, current_weight: float, usage_count: int, connection_id: str) -> float:
        """Strengthening that considers how recently the connection was used"""
        last_access = self.connection_access_history.get(connection_id, 0)
        recency = max(1, self.total_access_count - last_access + 1)
        
        # More recent usage = stronger increment
        recency_bonus = 1.0 / math.sqrt(recency)
        increment = self.base_increment * recency_bonus
        
        return min(self.max_weight, current_weight + increment)
    
    def _hybrid_strengthening(self, current_weight: float, usage_count: int, connection_id: str) -> float:
        """Combination of usage-based and frequency-based strengthening"""
        # Usage component (diminishing returns)
        usage_component = self._usage_based_strengthening(current_weight, usage_count)
        
        # Frequency component
        last_access = self.connection_access_history.get(connection_id, 0)
        recency = max(1, self.total_access_count - last_access + 1)
        frequency_bonus = 0.02 / math.sqrt(recency)  # Smaller contribution than usage
        
        final_weight = min(self.max_weight, usage_component + frequency_bonus)
        return final_weight
    
    def get_connection_strength_category(self, weight: float) -> str:
        """Categorize connection strength for visualization/analysis"""
        if weight >= 0.8:
            return "very_strong"
        elif weight >= 0.6:
            return "strong" 
        elif weight >= 0.4:
            return "moderate"
        elif weight >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def suggest_pruning_candidates(self, 
                                 connections: List[Tuple[str, float, int]],
                                 max_connections: int) -> List[str]:
        """
        Suggest which connections to prune when approaching max_connections limit.
        Returns list of connection_ids to consider for removal.
        """
        if len(connections) <= max_connections:
            return []
        
        # Sort by weight (weakest first)
        sorted_connections = sorted(connections, key=lambda x: x[1])
        
        # Consider removing the weakest connections beyond the limit
        excess_count = len(connections) - max_connections
        candidates = [conn[0] for conn in sorted_connections[:excess_count]]
        
        return candidates
    
    def get_statistics(self):
        """Get statistics about the current weight calculation state"""
        return {
            "total_access_count": self.total_access_count,
            "tracked_connections": len(self.connection_access_history),
            "base_increment": self.base_increment,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "strategy": self.strategy
        }