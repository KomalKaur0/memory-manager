"""
Usage Tracker - Track memory and connection usage patterns
Monitors access patterns to inform adaptive learning and connection strengthening.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from pathlib import Path

from ..core.memory_node import MemoryNode, ConnectionType
from ..storage.graph_database import MemoryGraphDatabase
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AccessType(str, Enum):
    """Types of memory access."""
    DIRECT_QUERY = "direct_query"
    TRAVERSAL = "traversal"
    SEMANTIC_SEARCH = "semantic_search"
    RELATED_SEARCH = "related_search"
    IMPORT = "import"
    UPDATE = "update"


@dataclass
class AccessRecord:
    """Record of a single memory access."""
    memory_id: str
    access_type: AccessType
    timestamp: datetime
    query_context: Optional[str] = None
    source_memory_id: Optional[str] = None  # For traversal
    relevance_score: Optional[float] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionUsage:
    """Record of connection usage."""
    from_id: str
    to_id: str
    connection_type: ConnectionType
    timestamp: datetime
    traversal_depth: int
    weight_at_time: float
    contributed_to_result: bool
    session_id: Optional[str] = None


@dataclass
class QueryPattern:
    """Pattern extracted from queries."""
    pattern_type: str  # e.g., "concept_pair", "keyword_cluster"
    pattern_value: Any  # The actual pattern
    frequency: int
    last_seen: datetime
    avg_result_quality: float
    successful_queries: int
    total_queries: int


@dataclass
class UsageStats:
    """Aggregated usage statistics."""
    total_accesses: int
    unique_memories_accessed: int
    avg_accesses_per_memory: float
    most_accessed_memories: List[Tuple[str, int]]
    access_by_type: Dict[AccessType, int]
    peak_usage_hour: int
    avg_session_length: float
    common_query_patterns: List[QueryPattern]
    connection_usage_stats: Dict[str, Any]


class UsageTracker:
    """
    Tracks and analyzes memory system usage patterns.
    """
    
    def __init__(
        self,
        graph_db: Optional[MemoryGraphDatabase] = None,
        persist_interval: int = 300,  # 5 minutes
        buffer_size: int = 1000
    ):
        """Initialize usage tracker."""
        self.graph_db = graph_db or MemoryGraphDatabase()
        self.persist_interval = persist_interval
        self.buffer_size = buffer_size
        
        # In-memory buffers for performance
        self.access_buffer: List[AccessRecord] = []
        self.connection_buffer: List[ConnectionUsage] = []
        self.query_patterns: Dict[str, QueryPattern] = {}
        
        # Tracking data structures
        self.memory_access_counts = Counter()
        self.connection_usage_counts = Counter()
        self.session_data: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self.buffer_lock = threading.Lock()
        
        # Start background persistence thread
        self._start_persistence_thread()
        
        # Load existing usage data
        self._load_usage_data()
    
    def track_memory_access(
        self,
        memory_id: str,
        access_type: AccessType,
        query_context: Optional[str] = None,
        source_memory_id: Optional[str] = None,
        relevance_score: Optional[float] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track a single memory access."""
        access_record = AccessRecord(
            memory_id=memory_id,
            access_type=access_type,
            timestamp=datetime.now(),
            query_context=query_context,
            source_memory_id=source_memory_id,
            relevance_score=relevance_score,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        with self.buffer_lock:
            self.access_buffer.append(access_record)
            self.memory_access_counts[memory_id] += 1
            
            # Track session data
            if session_id:
                if session_id not in self.session_data:
                    self.session_data[session_id] = {
                        'start_time': datetime.now(),
                        'access_count': 0,
                        'unique_memories': set()
                    }
                self.session_data[session_id]['access_count'] += 1
                self.session_data[session_id]['unique_memories'].add(memory_id)
                self.session_data[session_id]['last_access'] = datetime.now()
            
            # Check if buffer should be flushed
            if len(self.access_buffer) >= self.buffer_size:
                self._flush_buffers()
    
    def track_connection_usage(
        self,
        from_id: str,
        to_id: str,
        connection_type: ConnectionType,
        traversal_depth: int,
        weight_at_time: float,
        contributed_to_result: bool,
        session_id: Optional[str] = None
    ):
        """Track usage of a connection during traversal."""
        usage = ConnectionUsage(
            from_id=from_id,
            to_id=to_id,
            connection_type=connection_type,
            timestamp=datetime.now(),
            traversal_depth=traversal_depth,
            weight_at_time=weight_at_time,
            contributed_to_result=contributed_to_result,
            session_id=session_id
        )
        
        with self.buffer_lock:
            self.connection_buffer.append(usage)
            connection_key = f"{from_id}->{to_id}"
            self.connection_usage_counts[connection_key] += 1
            
            # Check if buffer should be flushed
            if len(self.connection_buffer) >= self.buffer_size:
                self._flush_buffers()
    
    def track_query_pattern(
        self,
        query: str,
        results: List[str],
        result_quality: float
    ):
        """Extract and track patterns from queries."""
        # Extract patterns from query
        patterns = self._extract_query_patterns(query, results)
        
        with self.buffer_lock:
            for pattern_type, pattern_value in patterns:
                pattern_key = f"{pattern_type}:{pattern_value}"
                
                if pattern_key not in self.query_patterns:
                    self.query_patterns[pattern_key] = QueryPattern(
                        pattern_type=pattern_type,
                        pattern_value=pattern_value,
                        frequency=0,
                        last_seen=datetime.now(),
                        avg_result_quality=0.0,
                        successful_queries=0,
                        total_queries=0
                    )
                
                pattern = self.query_patterns[pattern_key]
                pattern.frequency += 1
                pattern.last_seen = datetime.now()
                pattern.total_queries += 1
                
                if result_quality > 0.5:  # Threshold for "successful"
                    pattern.successful_queries += 1
                
                # Update average quality
                pattern.avg_result_quality = (
                    (pattern.avg_result_quality * (pattern.total_queries - 1) + result_quality) /
                    pattern.total_queries
                )
    
    def get_memory_usage_stats(self, memory_id: str) -> Dict[str, Any]:
        """Get usage statistics for a specific memory."""
        with self.buffer_lock:
            # Get recent accesses from buffer
            recent_accesses = [
                access for access in self.access_buffer
                if access.memory_id == memory_id
            ]
        
        # Query persisted data
        try:
            with self.graph_db.session() as session:
                result = session.run("""
                    MATCH (m:Memory {memory_id: $memory_id})
                    OPTIONAL MATCH (m)-[r:ACCESSED]->()
                    RETURN 
                        m.access_count as total_count,
                        m.last_accessed as last_accessed,
                        count(r) as access_records
                """, {'memory_id': memory_id})
                
                record = result.single()
                if record:
                    return {
                        'memory_id': memory_id,
                        'total_accesses': record['total_count'] or 0,
                        'last_accessed': record['last_accessed'],
                        'recent_accesses': len(recent_accesses),
                        'access_types': Counter(a.access_type.value for a in recent_accesses),
                        'avg_relevance': sum(a.relevance_score or 0 for a in recent_accesses) / len(recent_accesses) if recent_accesses else 0
                    }
        except Exception as e:
            logger.error(f"Failed to get memory usage stats: {e}")
        
        return {
            'memory_id': memory_id,
            'total_accesses': self.memory_access_counts.get(memory_id, 0),
            'recent_accesses': len(recent_accesses)
        }
    
    def get_connection_usage_stats(
        self,
        from_id: Optional[str] = None,
        to_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get usage statistics for connections."""
        with self.buffer_lock:
            if from_id and to_id:
                # Specific connection
                connection_key = f"{from_id}->{to_id}"
                usages = [
                    usage for usage in self.connection_buffer
                    if usage.from_id == from_id and usage.to_id == to_id
                ]
                
                return {
                    'connection': connection_key,
                    'total_uses': self.connection_usage_counts.get(connection_key, 0),
                    'recent_uses': len(usages),
                    'avg_depth': sum(u.traversal_depth for u in usages) / len(usages) if usages else 0,
                    'contribution_rate': sum(1 for u in usages if u.contributed_to_result) / len(usages) if usages else 0
                }
            else:
                # All connections
                return {
                    'total_connections_used': len(self.connection_usage_counts),
                    'total_uses': sum(self.connection_usage_counts.values()),
                    'most_used': self.connection_usage_counts.most_common(10)
                }
    
    def get_query_patterns(self, pattern_type: Optional[str] = None) -> List[QueryPattern]:
        """Get tracked query patterns."""
        with self.buffer_lock:
            if pattern_type:
                return [
                    p for p in self.query_patterns.values()
                    if p.pattern_type == pattern_type
                ]
            else:
                return sorted(
                    self.query_patterns.values(),
                    key=lambda p: p.frequency,
                    reverse=True
                )
    
    def get_usage_summary(
        self,
        time_window: Optional[timedelta] = None
    ) -> UsageStats:
        """Get comprehensive usage statistics."""
        cutoff_time = datetime.now() - time_window if time_window else None
        
        with self.buffer_lock:
            # Filter accesses by time window
            if cutoff_time:
                relevant_accesses = [
                    a for a in self.access_buffer
                    if a.timestamp >= cutoff_time
                ]
            else:
                relevant_accesses = self.access_buffer.copy()
        
        # Calculate statistics
        unique_memories = set(a.memory_id for a in relevant_accesses)
        access_by_type = Counter(a.access_type for a in relevant_accesses)
        
        # Hour histogram
        hour_counts = Counter(a.timestamp.hour for a in relevant_accesses)
        peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else 0
        
        # Session statistics
        session_lengths = []
        for session_id, data in self.session_data.items():
            if 'last_access' in data:
                length = (data['last_access'] - data['start_time']).total_seconds() / 60
                session_lengths.append(length)
        
        avg_session_length = sum(session_lengths) / len(session_lengths) if session_lengths else 0
        
        # Connection usage stats
        connection_stats = self.get_connection_usage_stats()
        
        return UsageStats(
            total_accesses=len(relevant_accesses),
            unique_memories_accessed=len(unique_memories),
            avg_accesses_per_memory=len(relevant_accesses) / len(unique_memories) if unique_memories else 0,
            most_accessed_memories=self.memory_access_counts.most_common(10),
            access_by_type=dict(access_by_type),
            peak_usage_hour=peak_hour,
            avg_session_length=avg_session_length,
            common_query_patterns=self.get_query_patterns()[:10],
            connection_usage_stats=connection_stats
        )
    
    def get_recommendations(self) -> Dict[str, List[Any]]:
        """Get recommendations based on usage patterns."""
        recommendations = {
            'memories_to_strengthen': [],
            'connections_to_create': [],
            'memories_to_decay': [],
            'patterns_to_optimize': []
        }
        
        # Find frequently accessed memories that should be strengthened
        for memory_id, count in self.memory_access_counts.most_common(20):
            if count > 10:  # Threshold
                recommendations['memories_to_strengthen'].append({
                    'memory_id': memory_id,
                    'access_count': count,
                    'recommendation': 'Increase importance score'
                })
        
        # Find co-accessed memories that should be connected
        co_access_patterns = self._find_co_access_patterns()
        for (memory1, memory2), count in co_access_patterns.most_common(10):
            if count > 5:  # Threshold
                recommendations['connections_to_create'].append({
                    'from_id': memory1,
                    'to_id': memory2,
                    'co_access_count': count,
                    'recommendation': 'Create similarity connection'
                })
        
        # Find rarely accessed memories for decay
        accessed_memories = set(self.memory_access_counts.keys())
        # Would need to query all memories and find ones not in accessed set
        
        # Find query patterns that could be optimized
        for pattern in self.get_query_patterns():
            if pattern.frequency > 10 and pattern.avg_result_quality < 0.5:
                recommendations['patterns_to_optimize'].append({
                    'pattern': pattern,
                    'recommendation': 'Low quality results for common pattern'
                })
        
        return recommendations
    
    def _extract_query_patterns(
        self,
        query: str,
        results: List[str]
    ) -> List[Tuple[str, str]]:
        """Extract patterns from a query."""
        patterns = []
        
        # Extract keywords (simple tokenization)
        words = query.lower().split()
        keywords = [w for w in words if len(w) > 3]
        
        # Keyword patterns
        if len(keywords) >= 2:
            patterns.append(('keyword_pair', f"{keywords[0]}_{keywords[1]}"))
        
        # Query length pattern
        if len(query) < 20:
            patterns.append(('query_type', 'short'))
        elif len(query) < 100:
            patterns.append(('query_type', 'medium'))
        else:
            patterns.append(('query_type', 'long'))
        
        # Result count pattern
        patterns.append(('result_count', f"{len(results)//5*5}-{len(results)//5*5+4}"))
        
        return patterns
    
    def _find_co_access_patterns(self) -> Counter:
        """Find memories that are frequently accessed together."""
        co_access = Counter()
        
        # Group accesses by session
        session_accesses = defaultdict(list)
        for access in self.access_buffer:
            if access.session_id:
                session_accesses[access.session_id].append(access.memory_id)
        
        # Count co-occurrences
        for session_memories in session_accesses.values():
            unique_memories = list(set(session_memories))
            for i in range(len(unique_memories)):
                for j in range(i + 1, len(unique_memories)):
                    pair = tuple(sorted([unique_memories[i], unique_memories[j]]))
                    co_access[pair] += 1
        
        return co_access
    
    def _flush_buffers(self):
        """Persist buffered data to database."""
        access_batch = []
        connection_batch = []
        
        try:
            # Prepare batch data
            access_batch = self.access_buffer.copy()
            connection_batch = self.connection_buffer.copy()
            
            # Clear buffers
            self.access_buffer.clear()
            self.connection_buffer.clear()
            
            # Persist to database
            if access_batch:
                self._persist_access_records(access_batch)
            
            if connection_batch:
                self._persist_connection_usage(connection_batch)
            
            # Persist query patterns
            self._persist_query_patterns()
            
            logger.debug(f"Flushed {len(access_batch)} access records and {len(connection_batch)} connection records")
            
        except Exception as e:
            logger.error(f"Failed to flush usage buffers: {e}")
            # Re-add to buffers on failure
            if access_batch:
                self.access_buffer.extend(access_batch)
            if connection_batch:
                self.connection_buffer.extend(connection_batch)
    
    def _persist_access_records(self, records: List[AccessRecord]):
        """Persist access records to database."""
        with self.graph_db.session() as session:
            # Batch create access nodes
            for record in records:
                session.run("""
                    MATCH (m:Memory {memory_id: $memory_id})
                    CREATE (a:AccessRecord {
                        timestamp: $timestamp,
                        access_type: $access_type,
                        query_context: $query_context,
                        source_memory_id: $source_memory_id,
                        relevance_score: $relevance_score,
                        session_id: $session_id,
                        metadata: $metadata
                    })
                    CREATE (m)-[:ACCESSED]->(a)
                    SET m.access_count = COALESCE(m.access_count, 0) + 1,
                        m.last_accessed = $timestamp
                """, {
                    'memory_id': record.memory_id,
                    'timestamp': record.timestamp.isoformat(),
                    'access_type': record.access_type.value,
                    'query_context': record.query_context,
                    'source_memory_id': record.source_memory_id,
                    'relevance_score': record.relevance_score,
                    'session_id': record.session_id,
                    'metadata': json.dumps(record.metadata)
                })
    
    def _persist_connection_usage(self, usages: List[ConnectionUsage]):
        """Persist connection usage records."""
        with self.graph_db.session() as session:
            for usage in usages:
                session.run("""
                    MATCH (m1:Memory {memory_id: $from_id})-[r:CONNECTED]->(m2:Memory {memory_id: $to_id})
                    WHERE r.connection_type = $connection_type
                    SET r.usage_count = COALESCE(r.usage_count, 0) + 1,
                        r.last_used = $timestamp
                    CREATE (u:ConnectionUsage {
                        timestamp: $timestamp,
                        traversal_depth: $depth,
                        weight_at_time: $weight,
                        contributed: $contributed,
                        session_id: $session_id
                    })
                    CREATE (r)-[:USAGE]->(u)
                """, {
                    'from_id': usage.from_id,
                    'to_id': usage.to_id,
                    'connection_type': usage.connection_type.value,
                    'timestamp': usage.timestamp.isoformat(),
                    'depth': usage.traversal_depth,
                    'weight': usage.weight_at_time,
                    'contributed': usage.contributed_to_result,
                    'session_id': usage.session_id
                })
    
    def _persist_query_patterns(self):
        """Persist query patterns to database."""
        if not self.query_patterns:
            return
        
        with self.graph_db.session() as session:
            for pattern in self.query_patterns.values():
                session.run("""
                    MERGE (p:QueryPattern {
                        pattern_type: $type,
                        pattern_value: $value
                    })
                    SET p.frequency = $frequency,
                        p.last_seen = $last_seen,
                        p.avg_quality = $avg_quality,
                        p.successful = $successful,
                        p.total = $total
                """, {
                    'type': pattern.pattern_type,
                    'value': str(pattern.pattern_value),
                    'frequency': pattern.frequency,
                    'last_seen': pattern.last_seen.isoformat(),
                    'avg_quality': pattern.avg_result_quality,
                    'successful': pattern.successful_queries,
                    'total': pattern.total_queries
                })
    
    def _load_usage_data(self):
        """Load existing usage data from database."""
        try:
            with self.graph_db.session() as session:
                # Load memory access counts
                result = session.run("""
                    MATCH (m:Memory)
                    WHERE m.access_count > 0
                    RETURN m.memory_id as memory_id, m.access_count as count
                """)
                
                for record in result:
                    self.memory_access_counts[record['memory_id']] = record['count']
                
                # Load query patterns
                result = session.run("""
                    MATCH (p:QueryPattern)
                    RETURN p
                """)
                
                for record in result:
                    pattern = record['p']
                    pattern_key = f"{pattern['pattern_type']}:{pattern['pattern_value']}"
                    self.query_patterns[pattern_key] = QueryPattern(
                        pattern_type=pattern['pattern_type'],
                        pattern_value=pattern['pattern_value'],
                        frequency=pattern.get('frequency', 0),
                        last_seen=datetime.fromisoformat(pattern.get('last_seen', datetime.now().isoformat())),
                        avg_result_quality=pattern.get('avg_quality', 0.0),
                        successful_queries=pattern.get('successful', 0),
                        total_queries=pattern.get('total', 0)
                    )
                
                logger.info(f"Loaded usage data: {len(self.memory_access_counts)} memories, {len(self.query_patterns)} patterns")
                
        except Exception as e:
            logger.error(f"Failed to load usage data: {e}")
    
    def _start_persistence_thread(self):
        """Start background thread for periodic persistence."""
        import threading
        
        def persist_periodically():
            import time
            while True:
                time.sleep(self.persist_interval)
                with self.buffer_lock:
                    if self.access_buffer or self.connection_buffer:
                        self._flush_buffers()
        
        thread = threading.Thread(target=persist_periodically, daemon=True)
        thread.start()
        logger.info(f"Started usage persistence thread (interval: {self.persist_interval}s)")
    
    def export_usage_data(self, output_path: str):
        """Export usage data for analysis."""
        data = {
            'export_time': datetime.now().isoformat(),
            'memory_access_counts': dict(self.memory_access_counts),
            'connection_usage_counts': dict(self.connection_usage_counts),
            'query_patterns': [
                {
                    'type': p.pattern_type,
                    'value': str(p.pattern_value),
                    'frequency': p.frequency,
                    'avg_quality': p.avg_result_quality,
                    'success_rate': p.successful_queries / p.total_queries if p.total_queries > 0 else 0
                }
                for p in self.query_patterns.values()
            ],
            'summary': self.get_usage_summary().__dict__
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported usage data to {output_path}")