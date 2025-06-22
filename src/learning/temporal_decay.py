"""
Temporal Decay - Weaken connections as new memories are formed
Implements organic forgetting where older connections naturally weaken as the system grows.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import math

from ..core.memory_node import MemoryNode, Connection, ConnectionType
from ..storage.graph_database import MemoryGraphDatabase
from ..storage.vector_store import MemoryVectorStore
from ...config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class DecayEvent:
    """Record of a decay operation."""
    affected_connections: int
    total_decay_applied: float
    memories_added_since_last: int
    weakest_connections_removed: int


class TemporalDecayManager:
    """
    Manages connection decay based on memory system growth.
    Connections weaken as new memories are added, simulating natural forgetting.
    """
    
    def __init__(
        self,
        graph_db: Optional[MemoryGraphDatabase] = None,
        vector_store: Optional[MemoryVectorStore] = None
    ):
        """Initialize temporal decay manager."""
        self.graph_db = graph_db or MemoryGraphDatabase()
        self.vector_store = vector_store or MemoryVectorStore()
        
        # Decay parameters
        self.decay_rate = settings.decay_rate
        self.min_weight = settings.min_connection_weight
        self.removal_threshold = 0.01  # Remove connections below this weight
        
        # Track system state
        self.last_memory_count = 0
        self.total_decay_events = 0
        self.memories_per_decay = 10  # Apply decay every N new memories
        
        # Initialize state
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize decay manager state."""
        try:
            stats = self.graph_db.get_graph_statistics()
            self.last_memory_count = stats.get('total_memories', 0)
            logger.info(f"Initialized decay manager with {self.last_memory_count} existing memories")
        except Exception as e:
            logger.error(f"Failed to initialize decay state: {e}")
    
    def check_and_apply_decay(self) -> Optional[DecayEvent]:
        """
        Check if decay should be applied based on new memories.
        Returns DecayEvent if decay was applied, None otherwise.
        """
        try:
            # Get current memory count
            stats = self.graph_db.get_graph_statistics()
            current_count = stats.get('total_memories', 0)
            
            # Calculate new memories since last check
            new_memories = current_count - self.last_memory_count
            
            # Apply decay if enough new memories
            if new_memories >= self.memories_per_decay:
                event = self._apply_decay(new_memories)
                self.last_memory_count = current_count
                self.total_decay_events += 1
                return event
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check/apply decay: {e}")
            return None
    
    def _apply_decay(self, new_memory_count: int) -> DecayEvent:
        """Apply decay to all connections based on number of new memories."""
        logger.info(f"Applying decay after {new_memory_count} new memories")
        
        affected_connections = 0
        total_decay = 0.0
        removed_connections = 0
        
        try:
            # Calculate decay amount based on new memories
            # More new memories = more decay
            decay_multiplier = 1 + math.log(new_memory_count) / 10
            decay_amount = self.decay_rate * decay_multiplier
            
            # Get all connections
            with self.graph_db.session() as session:
                # Fetch all connections
                result = session.run("""
                    MATCH (m1:Memory)-[r:CONNECTED]->(m2:Memory)
                    RETURN 
                        m1.memory_id as from_id,
                        m2.memory_id as to_id,
                        r.connection_type as conn_type,
                        r.weight as weight,
                        r.usage_count as usage_count
                """)
                
                connections_to_update = []
                connections_to_remove = []
                
                for record in result:
                    from_id = record['from_id']
                    to_id = record['to_id']
                    current_weight = record['weight'] or 0.0
                    usage_count = record['usage_count'] or 0
                    
                    # Calculate decay with protection for frequently used connections
                    usage_protection = math.tanh(usage_count / 10)  # 0 to 1 protection factor
                    protected_decay = decay_amount * (1 - usage_protection * 0.5)
                    
                    # Apply decay
                    new_weight = current_weight - protected_decay
                    
                    if new_weight <= self.removal_threshold:
                        connections_to_remove.append((from_id, to_id))
                        removed_connections += 1
                    elif new_weight < current_weight:
                        connections_to_update.append({
                            'from_id': from_id,
                            'to_id': to_id,
                            'new_weight': max(new_weight, self.min_weight),
                            'decay_applied': current_weight - new_weight
                        })
                        affected_connections += 1
                        total_decay += (current_weight - new_weight)
                
                # Batch update connections
                for conn in connections_to_update:
                    session.run("""
                        MATCH (m1:Memory {memory_id: $from_id})-[r:CONNECTED]->(m2:Memory {memory_id: $to_id})
                        SET r.weight = $new_weight
                    """, conn)
                
                # Remove very weak connections
                for from_id, to_id in connections_to_remove:
                    session.run("""
                        MATCH (m1:Memory {memory_id: $from_id})-[r:CONNECTED]->(m2:Memory {memory_id: $to_id})
                        DELETE r
                    """, {'from_id': from_id, 'to_id': to_id})
            
            logger.info(f"Decay complete: {affected_connections} connections weakened, {removed_connections} removed")
            
            return DecayEvent(
                affected_connections=affected_connections,
                total_decay_applied=total_decay,
                memories_added_since_last=new_memory_count,
                weakest_connections_removed=removed_connections
            )
            
        except Exception as e:
            logger.error(f"Failed to apply decay: {e}")
            return DecayEvent(0, 0.0, new_memory_count, 0)
    
    def apply_focused_decay(self, memory_ids: List[str], intensity: float = 1.0):
        """
        Apply decay focused on specific memories and their connections.
        Used when certain areas of the graph become less relevant.
        """
        logger.info(f"Applying focused decay to {len(memory_ids)} memories")
        
        affected = 0
        
        try:
            with self.graph_db.session() as session:
                for memory_id in memory_ids:
                    # Decay outgoing connections
                    result = session.run("""
                        MATCH (m:Memory {memory_id: $memory_id})-[r:CONNECTED]->()
                        WHERE r.weight > $min_weight
                        SET r.weight = r.weight * $decay_factor
                        RETURN count(r) as count
                    """, {
                        'memory_id': memory_id,
                        'min_weight': self.min_weight,
                        'decay_factor': 1 - (self.decay_rate * intensity)
                    })
                    
                    record = result.single()
                    if record:
                        affected += record['count'] or 0
                    
                    # Decay incoming connections
                    result = session.run("""
                        MATCH ()-[r:CONNECTED]->(m:Memory {memory_id: $memory_id})
                        WHERE r.weight > $min_weight
                        SET r.weight = r.weight * $decay_factor
                        RETURN count(r) as count
                    """, {
                        'memory_id': memory_id,
                        'min_weight': self.min_weight,
                        'decay_factor': 1 - (self.decay_rate * intensity * 0.5)  # Less decay for incoming
                    })
                    
                    record = result.single()
                    if record:
                        affected += record['count'] or 0
            
            logger.info(f"Focused decay affected {affected} connections")
            
        except Exception as e:
            logger.error(f"Failed to apply focused decay: {e}")
    
    def get_decay_candidates(self, limit: int = 100):
        """
        Get connections that are candidates for decay.
        Returns weakest connections that haven't been used recently.
        """
        candidates = []
        
        try:
            with self.graph_db.session() as session:
                result = session.run("""
                    MATCH (m1:Memory)-[r:CONNECTED]->(m2:Memory)
                    WHERE r.weight < $threshold
                    RETURN 
                        m1.memory_id as from_id,
                        m2.memory_id as to_id,
                        m1.concept as from_concept,
                        m2.concept as to_concept,
                        r.weight as weight,
                        r.usage_count as usage_count,
                        r.connection_type as conn_type
                    ORDER BY r.weight ASC
                    LIMIT $limit
                """, {
                    'threshold': self.min_weight * 2,
                    'limit': limit
                })
                
                for record in result:
                    candidates.append({
                        'from_id': record['from_id'],
                        'to_id': record['to_id'],
                        'from_concept': record['from_concept'],
                        'to_concept': record['to_concept'],
                        'weight': record['weight'],
                        'usage_count': record['usage_count'] or 0,
                        'connection_type': record['conn_type']
                    })
            
        except Exception as e:
            logger.error(f"Failed to get decay candidates: {e}")
        
        return candidates
    
    def protect_connection(self, from_id: str, to_id: str):
        """
        Boost a connection to protect it from decay.
        Used for connections that are semantically important.
        """
        try:
            with self.graph_db.session() as session:
                # Boost the connection weight
                session.run("""
                    MATCH (m1:Memory {memory_id: $from_id})-[r:CONNECTED]->(m2:Memory {memory_id: $to_id})
                    SET r.weight = CASE 
                        WHEN r.weight < 0.5 THEN r.weight + 0.1
                        ELSE r.weight
                    END,
                    r.usage_count = r.usage_count + 1
                """, {'from_id': from_id, 'to_id': to_id})
                
                logger.debug(f"Protected connection {from_id} -> {to_id}")
                
        except Exception as e:
            logger.error(f"Failed to protect connection: {e}")
    
    def get_decay_metrics(self):
        """Get metrics about the decay process."""
        try:
            with self.graph_db.session() as session:
                # Get connection weight distribution
                result = session.run("""
                    MATCH ()-[r:CONNECTED]->()
                    RETURN 
                        count(r) as total_connections,
                        avg(r.weight) as avg_weight,
                        min(r.weight) as min_weight,
                        max(r.weight) as max_weight,
                        stdev(r.weight) as weight_stdev
                """)
                
                record = result.single()
                if not record:
                    return {}
                
                # Get weak connection count
                weak_result = session.run("""
                    MATCH ()-[r:CONNECTED]->()
                    WHERE r.weight < $threshold
                    RETURN count(r) as weak_count
                """, {'threshold': self.min_weight * 2})
                
                weak_record = weak_result.single()
                weak_count = weak_record['weak_count'] if weak_record else 0
                
                return {
                    'total_connections': record['total_connections'] or 0,
                    'avg_weight': record['avg_weight'] or 0.0,
                    'min_weight': record['min_weight'] or 0.0,
                    'max_weight': record['max_weight'] or 0.0,
                    'weight_stdev': record['weight_stdev'] or 0.0,
                    'weak_connections': weak_count,
                    'decay_events': self.total_decay_events,
                    'memories_per_decay': self.memories_per_decay,
                    'current_memory_count': self.last_memory_count
                }
                
        except Exception as e:
            logger.error(f"Failed to get decay metrics: {e}")
            return {}