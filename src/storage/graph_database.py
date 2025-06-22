"""
Graph Database Storage - Neo4j Integration
Handles persistence of memory nodes and their connections in Neo4j.
"""

from typing import List, Dict, Optional, Set, Tuple, Any
# from datetime import datetime
import logging
from contextlib import contextmanager

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import Neo4jError

from ..core.memory_node import MemoryNode, Connection, ConnectionType
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MemoryGraphDatabase:
    """
    Neo4j database interface for storing and retrieving memory nodes and connections.
    """
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection."""
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self.driver: Optional[Driver] = None
        
    def connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(  # Using Neo4j's GraphDatabase here
                self.uri, 
                auth=(self.username, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Neo4jError as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def disconnect(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Disconnected from Neo4j")
    
    @contextmanager
    def session(self):
        """Context manager for Neo4j sessions."""
        if not self.driver:
            self.connect()
        if self.driver is not None:
            session = self.driver.session()
            try:
                yield session
            finally:
                session.close()
        else:
            print('session returned None')
    
    def initialize_schema(self):
        """Create indexes and constraints for optimal performance."""
        with self.session() as session:
            # Execute each query directly as a string literal
            try:
                session.run("CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Schema constraint failed: {e}")
            
            try:
                session.run("CREATE CONSTRAINT concept_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.concept IS UNIQUE")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Schema constraint failed: {e}")
            
            try:
                session.run("CREATE INDEX memory_keywords IF NOT EXISTS FOR (m:Memory) ON (m.keywords)")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Schema index failed: {e}")
            
            try:
                session.run("CREATE INDEX memory_tags IF NOT EXISTS FOR (m:Memory) ON (m.tags)")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Schema index failed: {e}")
            
            try:
                session.run("CREATE INDEX memory_created IF NOT EXISTS FOR (m:Memory) ON (m.created_at)")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Schema index failed: {e}")
            
            try:
                session.run("CREATE INDEX memory_accessed IF NOT EXISTS FOR (m:Memory) ON (m.last_accessed)")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Schema index failed: {e}")
            
            try:
                session.run("CREATE INDEX connection_weight IF NOT EXISTS FOR ()-[r:CONNECTED]-() ON (r.weight)")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Schema index failed: {e}")
            
            try:
                session.run("CREATE INDEX connection_type IF NOT EXISTS FOR ()-[r:CONNECTED]-() ON (r.connection_type)")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Schema index failed: {e}")
        
        logger.info("Database schema initialized")
    
    def store_memory_node(self, node: MemoryNode) -> bool:
        """
        Store or update a memory node in Neo4j.
        
        Args:
            node: MemoryNode to store
            
        Returns:
            bool: True if successful
        """
        try:
            with self.session() as session:
                result = session.run("""
                    MERGE (m:Memory {memory_id: $memory_id})
                    SET m.concept = $concept,
                        m.keywords = $keywords,
                        m.tags = $tags,
                        m.summary = $summary,
                        m.full_content = $full_content,
                        m.importance_score = $importance_score,
                        m.access_count = $access_count,
                        m.created_at = $created_at,
                        m.last_accessed = $last_accessed,
                        m.last_modified = $last_modified
                    RETURN m.memory_id as id
                """, {
                    'memory_id': node.id,  # Changed from node.memory_id to node.id
                    'concept': node.concept,
                    'keywords': node.keywords,
                    'tags': list(node.tags),
                    'summary': node.summary,
                    'full_content': node.full_content,
                    'importance_score': node.importance_score,
                    'access_count': node.access_count,
                })
                
                record = result.single()
                if record:
                    logger.debug(f"Stored memory node: {node.id}")  # Changed here too
                    return True
                    
        except Neo4jError as e:
            logger.error(f"Failed to store memory node {node.id}: {e}")  # And here
            
        return False
    
    def get_memory_node(self, memory_id: str) -> Optional[MemoryNode]:
        """
        Retrieve a memory node by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            MemoryNode or None if not found
        """
        query = """
        MATCH (m:Memory {memory_id: $memory_id})
        RETURN m
        """
        
        try:
            with self.session() as session:
                result = session.run(query, {'memory_id': memory_id})
                record = result.single()
                
                if record:
                    return self._record_to_memory_node(record['m'])
                    
        except Neo4jError as e:
            logger.error(f"Failed to retrieve memory node {memory_id}: {e}")
            
        return None
    
    def store_connection(self, from_id: str, to_id: str, connection: Connection) -> bool:
        """
        Store or update a connection between two memory nodes.
        
        Args:
            from_id: Source memory node ID
            to_id: Target memory node ID
            connection: Connection object with metadata
            
        Returns:
            bool: True if successful
        """
        query = """
        MATCH (from:Memory {memory_id: $from_id})
        MATCH (to:Memory {memory_id: $to_id})
        MERGE (from)-[r:CONNECTED {connection_type: $connection_type}]->(to)
        SET r.weight = $weight,
            r.created_at = $created_at,
            r.last_strengthened = $last_strengthened,
            r.usage_count = $usage_count
        RETURN r
        """
        
        try:
            with self.session() as session:
                result = session.run(query, {
                    'from_id': from_id,
                    'to_id': to_id,
                    'connection_type': connection.connection_type.value,
                    'weight': connection.weight,
                    'usage_count': connection.usage_count
                })
                
                if result.single():
                    logger.debug(f"Stored connection: {from_id} -> {to_id} ({connection.connection_type.value})")
                    return True
                    
        except Neo4jError as e:
            logger.error(f"Failed to store connection {from_id} -> {to_id}: {e}")
            
        return False
    
    def get_connections_from_node(self, memory_id: str, min_weight: float = 0.0) -> List[Tuple[str, Connection]]:
        """
        Get all outgoing connections from a memory node.
        
        Args:
            memory_id: Source memory node ID
            min_weight: Minimum connection weight threshold
            
        Returns:
            List of (target_id, Connection) tuples
        """
        query = """
        MATCH (from:Memory {memory_id: $memory_id})-[r:CONNECTED]->(to:Memory)
        WHERE r.weight >= $min_weight
        RETURN to.memory_id as target_id, r
        ORDER BY r.weight DESC
        """
        
        connections = []
        try:
            with self.session() as session:
                result = session.run(query, {
                    'memory_id': memory_id,
                    'min_weight': min_weight
                })
                
                for record in result:
                    connection = self._record_to_connection(record['r'])
                    connections.append((record['target_id'], connection))
                    
        except Neo4jError as e:
            logger.error(f"Failed to get connections from {memory_id}: {e}")
            
        return connections
    
    def find_memories_by_concept(self, concept: str) -> List[MemoryNode]:
        """Find memories with matching or similar concepts."""
        query = """
        MATCH (m:Memory)
        WHERE m.concept CONTAINS $concept
        RETURN m
        ORDER BY m.importance_score DESC, m.last_accessed DESC
        """
        
        memories = []
        try:
            with self.session() as session:
                result = session.run(query, {'concept': concept})
                
                for record in result:
                    memory = self._record_to_memory_node(record['m'])
                    memories.append(memory)
                    
        except Neo4jError as e:
            logger.error(f"Failed to find memories by concept '{concept}': {e}")
            
        return memories
    
    def find_memories_by_keywords(self, keywords: List[str]) -> List[MemoryNode]:
        """Find memories containing any of the specified keywords."""
        query = """
        MATCH (m:Memory)
        WHERE any(keyword IN $keywords WHERE any(mk IN m.keywords WHERE mk CONTAINS keyword))
        RETURN m
        ORDER BY m.importance_score DESC, m.last_accessed DESC
        """
        
        memories = []
        try:
            with self.session() as session:
                result = session.run(query, {'keywords': keywords})
                
                for record in result:
                    memory = self._record_to_memory_node(record['m'])
                    memories.append(memory)
                    
        except Neo4jError as e:
            logger.error(f"Failed to find memories by keywords {keywords}: {e}")
            
        return memories
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the memory graph."""
        query = """
        MATCH (m:Memory)
        OPTIONAL MATCH (m)-[r:CONNECTED]->()
        RETURN 
            count(DISTINCT m) as total_memories,
            count(r) as total_connections,
            avg(r.weight) as avg_connection_weight,
            max(r.weight) as max_connection_weight,
            min(r.weight) as min_connection_weight
        """
        
        try:
            with self.session() as session:
                result = session.run(query)
                record = result.single()
                
                if record:
                    return {
                        'total_memories': record['total_memories'] or 0,
                        'total_connections': record['total_connections'] or 0,
                        'avg_connection_weight': record['avg_connection_weight'] or 0.0,
                        'max_connection_weight': record['max_connection_weight'] or 0.0,
                        'min_connection_weight': record['min_connection_weight'] or 0.0
                    }
                    
        except Neo4jError as e:
            logger.error(f"Failed to get graph statistics: {e}")
            
        return {}
    
    def delete_memory_node(self, memory_id: str) -> bool:
        """Delete a memory node and all its connections."""
        query = """
        MATCH (m:Memory {memory_id: $memory_id})
        DETACH DELETE m
        RETURN count(m) as deleted_count
        """
        
        try:
            with self.session() as session:
                result = session.run(query, {'memory_id': memory_id})
                record = result.single()
                
                if record and record['deleted_count'] > 0:
                    logger.info(f"Deleted memory node: {memory_id}")
                    return True
                    
        except Neo4jError as e:
            logger.error(f"Failed to delete memory node {memory_id}: {e}")
            
        return False
    
    def _record_to_memory_node(self, record) -> MemoryNode:
        """Convert Neo4j record to MemoryNode object."""
        return MemoryNode(
            id=record['memory_id'],  # Changed from memory_id to id
            concept=record['concept'],
            keywords=record['keywords'],
            tags=list(record['tags']) if record['tags'] else [],  # Convert to list and handle None
            summary=record['summary'],
            full_content=record['full_content'],
            importance_score=record['importance_score'],
            access_count=record['access_count'],
        )
    
    def _record_to_connection(self, record) -> Connection:
        """Convert Neo4j relationship record to Connection object."""
        return Connection(
            target_node_id="",  # Changed from target_id to target_node_id
            connection_type=ConnectionType(record['connection_type']),
            weight=record['weight'],
            usage_count=record['usage_count']
        )
    