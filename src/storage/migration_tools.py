"""
Migration Tools - Database schema management and data migrations
Handles schema versioning, upgrades, and data transformations.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .graph_database import MemoryGraphDatabase
from .vector_store import MemoryVectorStore
from ..core.memory_node import MemoryNode, Connection, ConnectionType
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MigrationStatus(str, Enum):
    """Status of a migration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Migration:
    """Represents a database migration."""
    version: str
    name: str
    description: str
    up_function: Callable
    down_function: Optional[Callable]
    dependencies: List[str]
    created_at: datetime
    
    def __str__(self):
        return f"{self.version}: {self.name}"


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    migration_version: str
    status: MigrationStatus
    started_at: datetime
    completed_at: Optional[datetime]
    records_affected: int
    errors: List[str]
    rollback_available: bool


class MigrationManager:
    """
    Manages database migrations for the memory system.
    """
    
    def __init__(
        self,
        graph_db: Optional[MemoryGraphDatabase] = None,
        vector_store: Optional[MemoryVectorStore] = None,
        migrations_dir: Optional[str] = None
    ):
        """Initialize migration manager."""
        self.graph_db = graph_db or MemoryGraphDatabase()
        self.vector_store = vector_store or MemoryVectorStore()
        self.migrations_dir = Path(migrations_dir or settings.migrations_dir or "migrations")
        
        # Migration tracking
        self.migrations: Dict[str, Migration] = {}
        self.applied_migrations: Dict[str, MigrationResult] = {}
        
        # Ensure migrations directory exists
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize migration tracking in database
        self._initialize_migration_tracking()
        
        # Register built-in migrations
        self._register_builtin_migrations()
    
    def _initialize_migration_tracking(self):
        """Create migration tracking table/collection."""
        try:
            # Initialize Neo4j migration tracking
            self.graph_db.connect()
            with self.graph_db.session() as session:
                # Create migration node constraint
                session.run("""
                    CREATE CONSTRAINT migration_version_unique IF NOT EXISTS 
                    FOR (m:Migration) REQUIRE m.version IS UNIQUE
                """)
                
                # Load applied migrations
                result = session.run("""
                    MATCH (m:Migration)
                    RETURN m.version as version, 
                           m.status as status,
                           m.started_at as started_at,
                           m.completed_at as completed_at,
                           m.records_affected as records_affected,
                           m.errors as errors,
                           m.rollback_available as rollback_available
                    ORDER BY m.version
                """)
                
                for record in result:
                    self.applied_migrations[record['version']] = MigrationResult(
                        migration_version=record['version'],
                        status=MigrationStatus(record['status']),
                        started_at=datetime.fromisoformat(record['started_at']),
                        completed_at=datetime.fromisoformat(record['completed_at']) if record['completed_at'] else None,
                        records_affected=record['records_affected'] or 0,
                        errors=record['errors'] or [],
                        rollback_available=record['rollback_available'] or False
                    )
            
            logger.info("Migration tracking initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize migration tracking: {e}")
    
    def _register_builtin_migrations(self):
        """Register built-in migrations."""
        # V1.0.0 - Initial schema
        self.register_migration(
            version="1.0.0",
            name="initial_schema",
            description="Create initial database schema",
            up_function=self._migration_initial_schema_up,
            down_function=self._migration_initial_schema_down
        )
        
        # V1.1.0 - Add importance score index
        self.register_migration(
            version="1.1.0",
            name="add_importance_index",
            description="Add index on importance_score for better query performance",
            up_function=self._migration_add_importance_index_up,
            down_function=self._migration_add_importance_index_down,
            dependencies=["1.0.0"]
        )
        
        # V1.2.0 - Add connection metadata
        self.register_migration(
            version="1.2.0",
            name="add_connection_metadata",
            description="Add metadata fields to connections",
            up_function=self._migration_add_connection_metadata_up,
            down_function=None,  # Not reversible
            dependencies=["1.0.0"]
        )
    
    def register_migration(
        self,
        version: str,
        name: str,
        description: str,
        up_function: Callable,
        down_function: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None
    ):
        """Register a new migration."""
        migration = Migration(
            version=version,
            name=name,
            description=description,
            up_function=up_function,
            down_function=down_function,
            dependencies=dependencies or [],
            created_at=datetime.now()
        )
        
        self.migrations[version] = migration
        logger.info(f"Registered migration: {migration}")
    
    def get_pending_migrations(self) -> List[Migration]:
        """Get list of migrations that haven't been applied."""
        pending = []
        
        for version, migration in sorted(self.migrations.items()):
            if version not in self.applied_migrations or \
               self.applied_migrations[version].status != MigrationStatus.COMPLETED:
                # Check dependencies
                deps_satisfied = all(
                    dep in self.applied_migrations and 
                    self.applied_migrations[dep].status == MigrationStatus.COMPLETED
                    for dep in migration.dependencies
                )
                
                if deps_satisfied:
                    pending.append(migration)
        
        return pending
    
    def apply_migration(self, version: str) -> MigrationResult:
        """Apply a specific migration."""
        if version not in self.migrations:
            raise ValueError(f"Migration {version} not found")
        
        migration = self.migrations[version]
        logger.info(f"Applying migration: {migration}")
        
        # Check if already applied
        if version in self.applied_migrations and \
           self.applied_migrations[version].status == MigrationStatus.COMPLETED:
            logger.info(f"Migration {version} already applied")
            return self.applied_migrations[version]
        
        # Check dependencies
        for dep in migration.dependencies:
            if dep not in self.applied_migrations or \
               self.applied_migrations[dep].status != MigrationStatus.COMPLETED:
                raise ValueError(f"Dependency {dep} not satisfied for migration {version}")
        
        # Start migration
        result = MigrationResult(
            migration_version=version,
            status=MigrationStatus.IN_PROGRESS,
            started_at=datetime.now(),
            completed_at=None,
            records_affected=0,
            errors=[],
            rollback_available=migration.down_function is not None
        )
        
        # Save migration start
        self._save_migration_status(result)
        
        try:
            # Execute migration
            records_affected = migration.up_function()
            
            # Update result
            result.status = MigrationStatus.COMPLETED
            result.completed_at = datetime.now()
            result.records_affected = records_affected or 0
            
            logger.info(f"Migration {version} completed successfully. Records affected: {result.records_affected}")
            
        except Exception as e:
            # Migration failed
            result.status = MigrationStatus.FAILED
            result.errors.append(str(e))
            logger.error(f"Migration {version} failed: {e}")
        
        # Save final status
        self._save_migration_status(result)
        self.applied_migrations[version] = result
        
        return result
    
    def rollback_migration(self, version: str) -> MigrationResult:
        """Rollback a specific migration."""
        if version not in self.migrations:
            raise ValueError(f"Migration {version} not found")
        
        migration = self.migrations[version]
        
        if not migration.down_function:
            raise ValueError(f"Migration {version} does not support rollback")
        
        if version not in self.applied_migrations or \
           self.applied_migrations[version].status != MigrationStatus.COMPLETED:
            raise ValueError(f"Migration {version} is not in a completed state")
        
        logger.info(f"Rolling back migration: {migration}")
        
        # Start rollback
        result = MigrationResult(
            migration_version=version,
            status=MigrationStatus.IN_PROGRESS,
            started_at=datetime.now(),
            completed_at=None,
            records_affected=0,
            errors=[],
            rollback_available=False
        )
        
        try:
            # Execute rollback
            records_affected = migration.down_function()
            
            # Update result
            result.status = MigrationStatus.ROLLED_BACK
            result.completed_at = datetime.now()
            result.records_affected = records_affected or 0
            
            logger.info(f"Migration {version} rolled back successfully. Records affected: {result.records_affected}")
            
        except Exception as e:
            # Rollback failed
            result.status = MigrationStatus.FAILED
            result.errors.append(str(e))
            logger.error(f"Rollback of migration {version} failed: {e}")
        
        # Save final status
        self._save_migration_status(result)
        self.applied_migrations[version] = result
        
        return result
    
    def migrate_all(self) -> List[MigrationResult]:
        """Apply all pending migrations."""
        results = []
        pending = self.get_pending_migrations()
        
        logger.info(f"Found {len(pending)} pending migrations")
        
        for migration in pending:
            result = self.apply_migration(migration.version)
            results.append(result)
            
            # Stop if migration failed
            if result.status == MigrationStatus.FAILED:
                logger.error(f"Migration failed, stopping migration process")
                break
        
        return results
    
    def _save_migration_status(self, result: MigrationResult):
        """Save migration status to database."""
        try:
            with self.graph_db.session() as session:
                session.run("""
                    MERGE (m:Migration {version: $version})
                    SET m.status = $status,
                        m.started_at = $started_at,
                        m.completed_at = $completed_at,
                        m.records_affected = $records_affected,
                        m.errors = $errors,
                        m.rollback_available = $rollback_available
                """, {
                    'version': result.migration_version,
                    'status': result.status.value,
                    'started_at': result.started_at.isoformat(),
                    'completed_at': result.completed_at.isoformat() if result.completed_at else None,
                    'records_affected': result.records_affected,
                    'errors': result.errors,
                    'rollback_available': result.rollback_available
                })
        except Exception as e:
            logger.error(f"Failed to save migration status: {e}")
    
    # Built-in migration implementations
    
    def _migration_initial_schema_up(self) -> int:
        """Create initial database schema."""
        records_affected = 0
        
        # Neo4j schema
        self.graph_db.initialize_schema()
        
        # Weaviate schema
        self.vector_store.connect()
        self.vector_store.initialize_schema()
        
        return records_affected
    
    def _migration_initial_schema_down(self) -> int:
        """Remove initial database schema (dangerous!)."""
        logger.warning("Rolling back initial schema - this will delete all data!")
        
        count = 0
        
        try:
            # This is a destructive operation
            with self.graph_db.session() as session:
                result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) as count")
                record = result.single()
                if record and 'count' in record:
                    count = record['count']
        except Exception as e:
            logger.error(f"Failed to rollback initial schema: {e}")
            raise
        
        return count    
    
    def _migration_add_importance_index_up(self) -> int:
        """Add index on importance score."""
        with self.graph_db.session() as session:
            session.run("""
                CREATE INDEX memory_importance IF NOT EXISTS 
                FOR (m:Memory) ON (m.importance_score)
            """)
        return 0
    
    def _migration_add_importance_index_down(self) -> int:
        """Remove importance score index."""
        with self.graph_db.session() as session:
            session.run("DROP INDEX memory_importance IF EXISTS")
        return 0
    
    def _migration_add_connection_metadata_up(self) -> int:
        """Add metadata to existing connections."""
        records_affected = 0
        
        try:
            with self.graph_db.session() as session:
                # Add default metadata to connections missing it
                result = session.run("""
                    MATCH ()-[r:CONNECTED]->()
                    WHERE r.created_at IS NULL
                    SET r.created_at = datetime().isoformat(),
                        r.usage_count = COALESCE(r.usage_count, 0)
                    RETURN count(r) as count
                """)
                
                record = result.single()
                records_affected = record['count'] if record else 0
                
        except Exception as e:
            logger.error(f"Failed to add connection metadata: {e}")
            # Don't raise - allow migration to continue with 0 records affected
        
        return records_affected
    
    # Data transformation utilities
    
    def transform_memory_format(
        self,
        from_version: str,
        to_version: str,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Transform memory data between format versions."""
        logger.info(f"Transforming memory format from {from_version} to {to_version}")
        
        stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Get transformation function
            transform_func = self._get_transform_function(from_version, to_version)
            
            # Process memories in batches
            with self.graph_db.session() as session:
                skip = 0
                
                while True:
                    # Get batch of memories
                    result = session.run("""
                        MATCH (m:Memory)
                        RETURN m
                        SKIP $skip
                        LIMIT $limit
                    """, {'skip': skip, 'limit': batch_size})
                    
                    memories = list(result)
                    if not memories:
                        break
                    
                    # Transform each memory
                    for record in memories:
                        stats['total_processed'] += 1
                        
                        try:
                            # Apply transformation
                            transformed = transform_func(record['m'])
                            
                            # Update memory
                            session.run("""
                                MATCH (m:Memory {memory_id: $memory_id})
                                SET m = $properties
                            """, {
                                'memory_id': transformed['memory_id'],
                                'properties': transformed
                            })
                            
                            stats['successful'] += 1
                            
                        except Exception as e:
                            stats['failed'] += 1
                            stats['errors'].append(f"Failed to transform memory: {e}")
                    
                    skip += batch_size
                    
        except Exception as e:
            logger.error(f"Format transformation failed: {e}")
            stats['errors'].append(str(e))
        
        return stats
    
    def _get_transform_function(self, from_version: str, to_version: str) -> Callable:
        """Get transformation function for version upgrade."""
        # Define transformation functions between versions
        transformations = {
            ('1.0.0', '1.1.0'): self._transform_1_0_0_to_1_1_0,
            # Add more transformations as needed
        }
        
        key = (from_version, to_version)
        if key not in transformations:
            raise ValueError(f"No transformation available from {from_version} to {to_version}")
        
        return transformations[key]
    
    def _transform_1_0_0_to_1_1_0(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Example transformation function."""
        # Add any new fields or transform existing ones
        if 'importance_score' not in memory_data:
            memory_data['importance_score'] = 0.5
        
        return memory_data
    
    def export_schema(self, output_path: str):
        """Export current schema definition."""
        schema = {
            'version': '1.0',
            'exported_at': datetime.now().isoformat(),
            'neo4j': self._export_neo4j_schema(),
            'weaviate': self._export_weaviate_schema()
        }
        
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)
        
        logger.info(f"Schema exported to {output_path}")
    
    def _export_neo4j_schema(self) -> Dict[str, Any]:
        """Export Neo4j schema information."""
        schema = {
            'nodes': {},
            'relationships': {},
            'indexes': [],
            'constraints': []
        }
        
        try:
            with self.graph_db.session() as session:
                # Get constraints
                result = session.run("SHOW CONSTRAINTS")
                for record in result:
                    schema['constraints'].append({
                        'name': record.get('name'),
                        'type': record.get('type'),
                        'entity_type': record.get('entityType'),
                        'properties': record.get('properties')
                    })
                
                # Get indexes
                result = session.run("SHOW INDEXES")
                for record in result:
                    schema['indexes'].append({
                        'name': record.get('name'),
                        'type': record.get('type'),
                        'entity_type': record.get('entityType'),
                        'properties': record.get('properties')
                    })
        
        except Exception as e:
            logger.error(f"Failed to export Neo4j schema: {e}")
        
        return schema
    
    def _export_weaviate_schema(self) -> Dict[str, Any]:
        """Export Weaviate schema information."""
        try:
            self.vector_store.connect()
            c = self.vector_store.client
            # Get collection info
            if c is not None:
                collection = c.collections.get(self.vector_store.class_name)
            else:
                print('export weaviate schema returned None for client')
            
            return {
                'class_name': self.vector_store.class_name,
                'vectorizer': 'none',
                'properties': []  # Would need to query schema details
            }
            
        except Exception as e:
            logger.error(f"Failed to export Weaviate schema: {e}")
            return {}
    
    def verify_data_integrity(self) -> Dict[str, Any]:
        """Verify data integrity across databases."""
        logger.info("Starting data integrity verification")
        
        results = {
            'total_memories_neo4j': 0,
            'total_memories_weaviate': 0,
            'missing_in_weaviate': [],
            'missing_in_neo4j': [],
            'orphaned_connections': [],
            'integrity_issues': []
        }
        
        try:
            # Get all memory IDs from Neo4j
            neo4j_ids = set()
            with self.graph_db.session() as session:
                result = session.run("MATCH (m:Memory) RETURN m.memory_id as id")
                for record in result:
                    neo4j_ids.add(record['id'])
            
            results['total_memories_neo4j'] = len(neo4j_ids)
            
            # Get all memory IDs from Weaviate
            weaviate_ids = set()
            try:
                self.vector_store.connect()
                c = self.vector_store.client
                if c is not None:
                    collection = c.collections.get(self.vector_store.class_name)
                else:
                    collection = None
                    print('verify data integrity returned None for client so collection is None')
                # Query all objects (this is simplified - in production you'd paginate)
                if collection is not None:
                    response = collection.query.fetch_objects(limit=10000)


                    for obj in response.objects:
                        if 'memory_id' in obj.properties:
                            weaviate_ids.add(obj.properties['memory_id'])
                
                results['total_memories_weaviate'] = len(weaviate_ids)
                
            except Exception as e:
                logger.error(f"Failed to query Weaviate: {e}")
                results['integrity_issues'].append(f"Could not verify Weaviate data: {e}")
            
            # Find discrepancies only if we have Weaviate data
            if weaviate_ids or results['total_memories_weaviate'] == 0:
                results['missing_in_weaviate'] = list(neo4j_ids - weaviate_ids)
                results['missing_in_neo4j'] = list(weaviate_ids - neo4j_ids)
            
            # Check for orphaned connections
            with self.graph_db.session() as session:
                result = session.run("""
                    MATCH (m1:Memory)-[r:CONNECTED]->(m2:Memory)
                    WHERE NOT EXISTS(m1.memory_id) OR NOT EXISTS(m2.memory_id)
                    RETURN id(r) as connection_id
                """)
                
                for record in result:
                    results['orphaned_connections'].append(record['connection_id'])
            
            # Summary
            if results['missing_in_weaviate'] or results['missing_in_neo4j'] or results['orphaned_connections']:
                results['integrity_issues'].append("Data inconsistencies found between databases")
            
        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            results['integrity_issues'].append(str(e))
        
        return results