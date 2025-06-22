"""
Adaptive Weights - Dynamically adjust connection weights based on usage
Implements the core learning mechanism for the memory system.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any  # Added Any import
# from datetime import datetime, timedelta
from dataclasses import dataclass
import math

from ..core.memory_node import MemoryNode, Connection, ConnectionType
from ..core.edge_manager import EdgeManager
from ..storage.graph_database import MemoryGraphDatabase
from .usage_tracker import UsageTracker, AccessType
from ...config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class WeightUpdate:
    """Record of a weight update."""
    from_id: str
    to_id: str
    old_weight: float
    new_weight: float
    update_reason: str

class AdaptiveWeightManager:
    """
    Manages dynamic weight adjustments based on usage patterns.
    """
    
    def __init__(
        self,
        graph_db: Optional[MemoryGraphDatabase] = None,
        usage_tracker: Optional[UsageTracker] = None,
        edge_manager: Optional[EdgeManager] = None
    ):
        """Initialize adaptive weight manager."""
        self.graph_db = graph_db or MemoryGraphDatabase()
        self.usage_tracker = usage_tracker or UsageTracker(graph_db=self.graph_db)
        self.edge_manager = edge_manager or EdgeManager()
        
        # Weight adjustment parameters
        self.base_increment = settings.connection_strength_increment
        self.max_weight = settings.max_connection_weight
        self.min_weight = settings.min_connection_weight
        
        # Learning rate modifiers
        self.depth_penalty = 0.7  # Reduce increment for deeper traversals
        self.recency_boost = 1.5  # Boost for recently used connections
        self.relevance_multiplier = 2.0  # Multiplier based on result relevance
        
        # Track updates
        self.weight_updates: List[WeightUpdate] = []
    
    def update_weights_from_session(self, session_id: str):
        """Update weights based on a complete session."""
        # Get session data from usage tracker
        session_data = self._get_session_data(session_id)
        if not session_data:
            return
        
        # Update memory importance scores
        self._update_memory_importance(session_data['accessed_memories'])
        
        # Update connection weights
        self._update_connection_weights(session_data['used_connections'])
        
        # Apply learning from query patterns
        self._apply_pattern_learning(session_data['query_patterns'])
        
        logger.info(f"Updated weights for session {session_id}: {len(self.weight_updates)} updates")
    
    def strengthen_connection(
        self,
        from_id: str,
        to_id: str,
        usage_context: Dict[str, Any]  # Fixed: any -> Any
    ) -> float:
        """Strengthen a specific connection based on usage."""
        try:
            # Get current connection
            connections = self.graph_db.get_connections_from_node(from_id, min_weight=0)
            current_connection = None
            
            for target_id, connection in connections:
                if target_id == to_id:
                    current_connection = connection
                    break
            
            if not current_connection:
                logger.warning(f"Connection {from_id} -> {to_id} not found")
                return 0.0
            
            # Calculate weight increment
            increment = self._calculate_weight_increment(
                current_connection,
                usage_context
            )
            
            # Apply weight update
            old_weight = current_connection.weight
            new_weight = min(old_weight + increment, self.max_weight)
            
            if new_weight != old_weight:
                current_connection.weight = new_weight
                # Update strengthened time in a way that works with Connection class
                
                current_connection.usage_count += 1
                
                # Save to database
                self.graph_db.store_connection(from_id, to_id, current_connection)
                
                # Track update
                self.weight_updates.append(WeightUpdate(
                    from_id=from_id,
                    to_id=to_id,
                    old_weight=old_weight,
                    new_weight=new_weight,
                    update_reason=f"Usage: {usage_context.get('reason', 'traversal')}"
                ))
            
            return new_weight
            
        except Exception as e:
            logger.error(f"Failed to strengthen connection: {e}")
            return 0.0
    
    def _calculate_weight_increment(
        self,
        connection: Connection,
        context: Dict[str, Any]  # Fixed: any -> Any
    ) -> float:
        """Calculate weight increment based on usage context."""
        base = self.base_increment
        
        # Depth penalty - deeper traversals get less weight
        depth = context.get('traversal_depth', 1)
        depth_modifier = math.pow(self.depth_penalty, depth - 1)
        
        # Recency boost - recently created/used connections get more weight
        recency_modifier = 1.0
        # Since Connection might not have last_strengthened, use created_at
        # if hasattr(connection, 'created_at') and connection.created_at:
        #     hours_since_creation = (datetime.now() - connection.created_at).total_seconds() / 3600
        #     if hours_since_creation < 24:  # Boost new connections
        #         recency_modifier = self.recency_boost
        
        # Relevance modifier - connections that contributed to good results get more weight
        relevance = context.get('relevance_score', 0.5)
        relevance_modifier = 1 + (relevance - 0.5) * self.relevance_multiplier
        
        # Connection type modifier - some connection types are more important
        type_modifiers = {
            ConnectionType.CAUSE_EFFECT: 1.2,
            ConnectionType.TEMPORAL_BEFORE: 1.1,
            ConnectionType.GENERAL_SPECIFIC: 1.1,
            ConnectionType.SIMILARITY: 1.0,
            ConnectionType.CONTRAST: 0.9,
            ConnectionType.CONTEXT: 0.8
        }
        type_modifier = type_modifiers.get(connection.connection_type, 1.0)
        
        # Calculate final increment
        increment = base * depth_modifier * recency_modifier * relevance_modifier * type_modifier
        
        # Diminishing returns - as weight increases, harder to increase further
        current_weight_modifier = 1 - (connection.weight / self.max_weight) ** 2
        increment *= current_weight_modifier
        
        return increment
    
    def _update_memory_importance(self, accessed_memories: List[Dict[str, Any]]):  # Fixed: any -> Any
        """Update importance scores for accessed memories."""
        for memory_data in accessed_memories:
            memory_id = memory_data['memory_id']
            access_count = memory_data['access_count']
            relevance_scores = memory_data.get('relevance_scores', [])
            
            # Get memory
            memory = self.graph_db.get_memory_node(memory_id)
            if not memory:
                continue
            
            # Calculate importance adjustment
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
            importance_boost = 0.05 * math.log(access_count + 1) * avg_relevance
            
            # Update importance score
            old_importance = memory.importance_score
            new_importance = min(old_importance + importance_boost, 1.0)
            
            if new_importance != old_importance:
                memory.importance_score = new_importance
                self.graph_db.store_memory_node(memory)
                
                logger.debug(f"Updated importance for {memory_id}: {old_importance:.3f} -> {new_importance:.3f}")
    
    def _update_connection_weights(self, used_connections: List[Dict[str, Any]]):  # Fixed: any -> Any
        """Update weights for used connections."""
        # Group by connection
        connection_groups = {}
        for conn_data in used_connections:
            key = (conn_data['from_id'], conn_data['to_id'])
            if key not in connection_groups:
                connection_groups[key] = []
            connection_groups[key].append(conn_data)
        
        # Update each connection
        for (from_id, to_id), usages in connection_groups.items():
            # Aggregate usage data
            total_uses = len(usages)
            avg_depth = sum(u['depth'] for u in usages) / total_uses
            contribution_rate = sum(1 for u in usages if u['contributed']) / total_uses
            
            # Calculate context for weight update
            context = {
                'traversal_depth': avg_depth,
                'relevance_score': contribution_rate,
                'usage_count': total_uses,
                'reason': 'batch_update'
            }
            
            self.strengthen_connection(from_id, to_id, context)
    
    def _apply_pattern_learning(self, patterns: List[Dict[str, Any]]):  # Fixed: any -> Any
        """Apply learning from query patterns."""
        for pattern in patterns:
            if pattern['type'] == 'co_access':
                # Memories accessed together should be connected
                memory_ids = pattern['memory_ids']
                for i in range(len(memory_ids)):
                    for j in range(i + 1, len(memory_ids)):
                        self._suggest_new_connection(memory_ids[i], memory_ids[j], pattern)
            
            elif pattern['type'] == 'traversal_path':
                # Frequently traversed paths should be strengthened
                path = pattern['path']
                for i in range(len(path) - 1):
                    context = {
                        'traversal_depth': 1,
                        'relevance_score': pattern.get('success_rate', 0.5),
                        'reason': 'frequent_path'
                    }
                    self.strengthen_connection(path[i], path[i + 1], context)
    
    def _suggest_new_connection(
        self,
        from_id: str,
        to_id: str,
        pattern_data: Dict[str, Any]  # Fixed: any -> Any
    ):
        """Suggest creating a new connection based on patterns."""
        # Check if connection already exists
        existing = self.graph_db.get_connections_from_node(from_id, min_weight=0)
        if any(target_id == to_id for target_id, _ in existing):
            return
        
        # Log suggestion (actual creation would be done by connection_builder)
        logger.info(f"Suggested new connection: {from_id} -> {to_id} based on {pattern_data['type']}")
    
    def _get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:  # Fixed: any -> Any
        """Get aggregated session data from usage tracker."""
        # This would be implemented to query the usage tracker
        # For now, return mock data structure
        return {
            'accessed_memories': [],
            'used_connections': [],
            'query_patterns': []
        }
    
    def get_learning_metrics(self) -> Dict[str, Any]:  # Fixed: any -> Any
        """Get metrics about the learning process."""
        if not self.weight_updates:
            return {
                'total_updates': 0,
                'avg_weight_change': 0,
                'strengthened_connections': 0,
                'max_weight_reached': 0
            }
        
        weight_changes = [u.new_weight - u.old_weight for u in self.weight_updates]
        
        return {
            'total_updates': len(self.weight_updates),
            'avg_weight_change': sum(weight_changes) / len(weight_changes),
            'strengthened_connections': sum(1 for c in weight_changes if c > 0),
            'max_weight_reached': sum(1 for u in self.weight_updates if u.new_weight >= self.max_weight),
            'recent_updates': [
                {
                    'connection': f"{u.from_id} -> {u.to_id}",
                    'change': u.new_weight - u.old_weight,
                    'reason': u.update_reason
                }
                for u in self.weight_updates[-10:]
            ]
        }