"""
Storage Module - Database persistence for the AI Memory System

This module provides interfaces for storing and retrieving memories
across Neo4j (graph database) and Weaviate (vector database).
"""

from .graph_database import MemoryGraphDatabase
from .vector_store import MemoryVectorStore
from .memory_importer import (
    MemoryImporter,
    ImportFormat,
    ImportResult
)
from .migration_tools import (
    MigrationManager,
    Migration,
    MigrationResult,
    MigrationStatus
)

# Version info
__version__ = "1.0.0"

# Module exports
__all__ = [
    # Core storage classes
    "MemoryGraphDatabase",
    "MemoryVectorStore",
    
    # Import functionality
    "MemoryImporter",
    "ImportFormat",
    "ImportResult",
    
    # Migration tools
    "MigrationManager",
    "Migration",
    "MigrationResult",
    "MigrationStatus",
    
    # Convenience functions
    "create_storage_backends",
    "initialize_storage",
    "verify_storage_health"
]


def create_storage_backends(
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    weaviate_url: str,
    weaviate_api_key: str,
    embedding_model: str
) -> tuple[MemoryGraphDatabase, MemoryVectorStore]:
    """
    Create and initialize both storage backends.
    
    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        weaviate_url: Weaviate connection URL
        weaviate_api_key: Weaviate API key (optional)
        embedding_model: Name of the embedding model to use
        
    Returns:
        Tuple of (MemoryGraphDatabase, MemoryVectorStore)
    """
    # Create graph database
    graph_db = MemoryGraphDatabase(
        uri=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password
    )
    
    # Create vector store
    vector_store = MemoryVectorStore(
        weaviate_url=weaviate_url,
        weaviate_api_key=weaviate_api_key,
        embedding_model=embedding_model
    )
    
    return graph_db, vector_store


def initialize_storage(
    graph_db: MemoryGraphDatabase,
    vector_store: MemoryVectorStore,
    run_migrations: bool = True
) -> dict:
    """
    Initialize storage backends and optionally run migrations.
    
    Args:
        graph_db: Graph database instance (creates new if None)
        vector_store: Vector store instance (creates new if None)
        run_migrations: Whether to run pending migrations
        
    Returns:
        Dict with initialization results
    """
    results = {
        'graph_db_connected': False,
        'vector_store_connected': False,
        'schemas_initialized': False,
        'migrations_applied': [],
        'errors': []
    }
    
    # Initialize graph database
    if not graph_db:
        graph_db = MemoryGraphDatabase()
    
    try:
        graph_db.connect()
        graph_db.initialize_schema()
        results['graph_db_connected'] = True
    except Exception as e:
        results['errors'].append(f"Graph DB initialization failed: {e}")
    
    # Initialize vector store
    if not vector_store:
        vector_store = MemoryVectorStore()
    
    try:
        vector_store.connect()
        vector_store.initialize_schema()
        results['vector_store_connected'] = True
    except Exception as e:
        results['errors'].append(f"Vector store initialization failed: {e}")
    
    # Mark schemas as initialized if both connected
    if results['graph_db_connected'] and results['vector_store_connected']:
        results['schemas_initialized'] = True
    
    # Run migrations if requested
    if run_migrations and results['schemas_initialized']:
        try:
            migration_manager = MigrationManager(
                graph_db=graph_db,
                vector_store=vector_store
            )
            
            migration_results = migration_manager.migrate_all()
            results['migrations_applied'] = [
                {
                    'version': r.migration_version,
                    'status': r.status.value,
                    'records_affected': r.records_affected
                }
                for r in migration_results
            ]
        except Exception as e:
            results['errors'].append(f"Migration failed: {e}")
    
    return results


def verify_storage_health(
    graph_db: MemoryGraphDatabase,
    vector_store: MemoryVectorStore
) -> dict:
    """
    Verify health and integrity of storage backends.
    
    Args:
        graph_db: Graph database instance
        vector_store: Vector store instance
        
    Returns:
        Dict with health check results
    """
    health = {
        'graph_db_healthy': False,
        'vector_store_healthy': False,
        'graph_stats': {},
        'vector_stats': {},
        'integrity_check': {},
        'errors': []
    }
    
    # Check graph database
    if not graph_db:
        graph_db = MemoryGraphDatabase()
    
    try:
        graph_db.connect()
        health['graph_stats'] = graph_db.get_graph_statistics()
        health['graph_db_healthy'] = True
    except Exception as e:
        health['errors'].append(f"Graph DB health check failed: {e}")
    
    # Check vector store
    if not vector_store:
        vector_store = MemoryVectorStore()
    
    try:
        vector_store.connect()
        health['vector_stats'] = vector_store.get_collection_stats()
        health['vector_store_healthy'] = True
    except Exception as e:
        health['errors'].append(f"Vector store health check failed: {e}")
    
    # Run integrity check if both are healthy
    if health['graph_db_healthy'] and health['vector_store_healthy']:
        try:
            migration_manager = MigrationManager(
                graph_db=graph_db,
                vector_store=vector_store
            )
            health['integrity_check'] = migration_manager.verify_data_integrity()
        except Exception as e:
            health['errors'].append(f"Integrity check failed: {e}")
    
    return health


# Example usage in docstring
"""
Example Usage:

    # Quick setup
    from src.storage import initialize_storage
    
    results = initialize_storage()
    if results['schemas_initialized']:
        print("Storage is ready!")
    
    # Manual setup with custom configuration
    from src.storage import MemoryGraphDatabase, MemoryVectorStore
    
    graph_db = MemoryGraphDatabase(uri="bolt://localhost:7687")
    vector_store = MemoryVectorStore(weaviate_url="http://localhost:8080")
    
    graph_db.connect()
    vector_store.connect()
    
    # Import data
    from src.storage import MemoryImporter, ImportFormat
    
    importer = MemoryImporter(graph_db, vector_store)
    result = importer.import_file("memories.json", ImportFormat.JSON)
    print(f"Imported {result.successful_imports} memories")
    
    # Check health
    from src.storage import verify_storage_health
    
    health = verify_storage_health(graph_db, vector_store)
    print(f"Graph DB: {health['graph_stats']}")
    print(f"Vector Store: {health['vector_stats']}")
"""