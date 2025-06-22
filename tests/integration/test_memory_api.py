"""
Integration tests for memory API endpoints
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import os

# Set environment variables before importing app
os.environ['VOYAGER_LITE_API_KEY'] = 'test-api-key'
os.environ['CLAUDE_API_KEY'] = 'test-claude-key'

from src.core.memory_graph import MemoryGraph
from src.core.memory_node import MemoryNode, ConnectionType
from src.retrieval.embedding_search import EmbeddingSearch
from src.retrieval.hybrid_retriever import HybridRetriever

# Import app components separately to avoid startup
from fastapi import FastAPI
from src.api.memory_api import memory_router
from src.api.search_api import search_router
from src.api.admin_api import admin_router
from src.api.chat_api import chat_router

# Test fixtures
@pytest.fixture
def mock_memory_graph():
    """Create a mock memory graph with test data"""
    graph = MemoryGraph()
    
    # Create test nodes
    node1 = MemoryNode(
        concept="Test Concept 1",
        summary="Test summary 1",
        full_content="Test content 1",
        tags=["test", "memory"],
        keywords=["test", "concept"]
    )
    node1.id = "test-node-1"
    node1.embedding_id = "test-embedding-1"  # Mock embedding ID
    
    node2 = MemoryNode(
        concept="Test Concept 2", 
        summary="Test summary 2",
        full_content="Test content 2",
        tags=["test", "graph"],
        keywords=["test", "graph"]
    )
    node2.id = "test-node-2"
    node2.embedding_id = "test-embedding-2"  # Mock embedding ID
    
    graph.nodes = {
        "test-node-1": node1,
        "test-node-2": node2
    }
    
    # Add connection
    graph.create_connection("test-node-1", "test-node-2", ConnectionType.SIMILARITY, 0.8)
    
    return graph

@pytest.fixture
def mock_embedding_search():
    """Create a mock embedding search service"""
    mock = MagicMock(spec=EmbeddingSearch)
    # Set up async methods
    mock.get_embedding = AsyncMock(return_value=[0.5] * 384)
    mock.find_similar = AsyncMock(return_value=[("test-node-2", 0.85)])
    mock.search_by_text = AsyncMock(return_value=[("test-node-2", 0.85)])
    mock.is_ready = MagicMock(return_value=True)
    mock.initialize = AsyncMock()
    mock.store_embedding = MagicMock()
    return mock

@pytest.fixture
def mock_hybrid_retriever(mock_memory_graph, mock_embedding_search):
    """Create a mock hybrid retriever"""
    mock = MagicMock(spec=HybridRetriever)
    mock.memory_graph = mock_memory_graph
    mock.embedding_search = mock_embedding_search
    
    # Mock the search_memories method
    async def mock_search_memories(query, max_results=10, use_graph_expansion=True):
        return [
            {
                'node': mock_memory_graph.get_node("test-node-2").dict(),
                'score': 0.85,
                'source': 'embedding'
            }
        ]
    
    mock.search_memories = AsyncMock(side_effect=mock_search_memories)
    return mock

@pytest.fixture
def test_app(mock_memory_graph, mock_embedding_search, mock_hybrid_retriever):
    """Create test app with mocked dependencies"""
    # Create a test app without lifespan
    test_app = FastAPI()
    
    # Include routers
    test_app.include_router(memory_router, prefix="/api/memory", tags=["memory"])
    test_app.include_router(search_router, prefix="/api/search", tags=["search"])
    test_app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
    test_app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
    
    # Add health check endpoint
    @test_app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "components": {
                "memory_graph": True,
                "embedding_search": True,
                "hybrid_retriever": True
            }
        }
    
    # Set up app state
    test_app.state.memory_graph = mock_memory_graph
    test_app.state.embedding_search = mock_embedding_search
    test_app.state.hybrid_retriever = mock_hybrid_retriever
    test_app.state.claude_client = MagicMock()
    
    return test_app

@pytest.fixture
def test_client(test_app):
    """Create test client from test app"""
    with TestClient(test_app) as client:
        yield client

class TestMemoryAPI:
    """Test suite for memory API endpoints"""
    
    def test_health_check(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
    
    def test_get_all_memory_nodes(self, test_client):
        """Test getting all memory nodes"""
        response = test_client.get("/api/memory/nodes")
        assert response.status_code == 200
        data = response.json()
        
        # Should return both test nodes
        assert len(data) == 2
        assert "test-node-1" in data
        assert "test-node-2" in data
        
        # Check node structure
        node1 = data["test-node-1"]
        assert node1["concept"] == "Test Concept 1"
        assert node1["summary"] == "Test summary 1"
        assert node1["tags"] == ["test", "memory"]
    
    def test_get_memory_node_by_id(self, test_client):
        """Test getting a specific memory node"""
        response = test_client.get("/api/memory/nodes/test-node-1")
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == "test-node-1"
        assert data["concept"] == "Test Concept 1"
        assert data["content"] == "Test content 1"
    
    def test_get_nonexistent_memory_node(self, test_client):
        """Test getting a non-existent memory node"""
        response = test_client.get("/api/memory/nodes/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_create_memory_node(self, test_client):
        """Test creating a new memory node"""
        new_node_data = {
            "concept": "New Test Concept",
            "summary": "New test summary", 
            "content": "New test content",
            "tags": ["new", "test"],
            "keywords": ["new", "concept"]
        }
        
        response = test_client.post("/api/memory/nodes", json=new_node_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["concept"] == "New Test Concept"
        assert data["summary"] == "New test summary"
        assert data["tags"] == ["new", "test"]
        assert data["id"]  # Should have generated ID
    
    def test_create_memory_node_invalid_data(self, test_client):
        """Test creating memory node with invalid data"""
        invalid_data = {
            "summary": "Missing concept field"
        }
        
        response = test_client.post("/api/memory/nodes", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_update_memory_node(self, test_client):
        """Test updating an existing memory node"""
        update_data = {
            "concept": "Updated Concept",
            "summary": "Updated summary",
            "content": "Updated content", 
            "tags": ["updated", "test"],
            "keywords": ["updated"]
        }
        
        response = test_client.put("/api/memory/nodes/test-node-1", json=update_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["concept"] == "Updated Concept"
        assert data["summary"] == "Updated summary"
        assert data["tags"] == ["updated", "test"]
    
    def test_update_nonexistent_memory_node(self, test_client):
        """Test updating a non-existent memory node"""
        update_data = {
            "concept": "Updated Concept",
            "summary": "Updated summary",
            "content": "Updated content",
            "tags": [],
            "keywords": []
        }
        
        response = test_client.put("/api/memory/nodes/nonexistent", json=update_data)
        assert response.status_code == 404
    
    def test_delete_memory_node(self, test_client):
        """Test deleting a memory node"""
        response = test_client.delete("/api/memory/nodes/test-node-1")
        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data["message"]
        
        # Verify node is gone
        response = test_client.get("/api/memory/nodes/test-node-1")
        assert response.status_code == 404
    
    def test_delete_nonexistent_memory_node(self, test_client):
        """Test deleting a non-existent memory node"""
        response = test_client.delete("/api/memory/nodes/nonexistent")
        assert response.status_code == 404
    
    def test_create_connection(self, test_client):
        """Test creating a connection between nodes"""
        connection_data = {
            "source_id": "test-node-1",
            "target_id": "test-node-2", 
            "connection_type": "similarity",
            "initial_weight": 0.7
        }
        
        response = test_client.post("/api/memory/connections", json=connection_data)
        assert response.status_code == 200
        data = response.json()
        assert "created successfully" in data["message"]
    
    def test_create_connection_invalid_nodes(self, test_client):
        """Test creating connection with invalid node IDs"""
        connection_data = {
            "source_id": "nonexistent-1",
            "target_id": "nonexistent-2",
            "connection_type": "similarity",
            "initial_weight": 0.5
        }
        
        response = test_client.post("/api/memory/connections", json=connection_data)
        assert response.status_code == 400
    
    def test_record_memory_access(self, test_client):
        """Test recording a memory access event"""
        access_data = {
            "node_id": "test-node-1",
            "access_type": "read",
            "timestamp": 1234567890.0
        }
        
        response = test_client.post("/api/memory/access", json=access_data)
        assert response.status_code == 200
        data = response.json()
        assert "recorded successfully" in data["message"]
    
    def test_record_memory_access_invalid_node(self, test_client):
        """Test recording access for non-existent node"""
        access_data = {
            "node_id": "nonexistent",
            "access_type": "read", 
            "timestamp": 1234567890.0
        }
        
        response = test_client.post("/api/memory/access", json=access_data)
        assert response.status_code == 404
    
    def test_get_connected_nodes(self, test_client):
        """Test getting connected nodes"""
        response = test_client.get("/api/memory/nodes/test-node-1/connected")
        assert response.status_code == 200
        data = response.json()
        
        # Should find connected node
        assert len(data) > 0
        connected_node = data[0]
        assert "node" in connected_node
        assert "weight" in connected_node
        assert "depth" in connected_node
    
    def test_get_connected_nodes_with_params(self, test_client):
        """Test getting connected nodes with custom parameters"""
        response = test_client.get(
            "/api/memory/nodes/test-node-1/connected?max_depth=1&min_weight=0.5"
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

class TestMemoryAPIValidation:
    """Test input validation for memory API"""
    
    def test_invalid_access_type(self, test_client):
        """Test invalid access type in memory access event"""
        access_data = {
            "node_id": "test-node-1",
            "access_type": "invalid_type",  # Invalid access type
            "timestamp": 1234567890.0
        }
        
        response = test_client.post("/api/memory/access", json=access_data)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_weight_range(self, test_client):
        """Test invalid weight range in connection creation"""
        connection_data = {
            "source_id": "test-node-1",
            "target_id": "test-node-2",
            "connection_type": "similarity",
            "initial_weight": 1.5  # Invalid weight > 1.0
        }
        
        response = test_client.post("/api/memory/connections", json=connection_data)
        assert response.status_code == 422  # Validation error

# Run tests with: pytest tests/integration/test_memory_api.py -v