"""
Integration tests for chat API co-access functionality
"""
import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock

# Set environment variables before importing
os.environ['VOYAGER_LITE_API_KEY'] = 'test-api-key'
os.environ['CLAUDE_API_KEY'] = 'test-claude-key'

from src.core.memory_graph import MemoryGraph
from src.core.memory_node import MemoryNode, ConnectionType
from src.agents.connection_agent import ConnectionAgent
from src.agents.mock_claude_client import MockClaudeClient
from src.retrieval.embedding_search import EmbeddingSearch
from src.retrieval.hybrid_retriever import HybridRetriever

# Import app components
from fastapi import FastAPI
from src.api.chat_api import chat_router


class TestChatCoAccess:
    """Test chat API co-access integration"""
    
    @pytest.fixture
    def memory_graph(self):
        """Create memory graph with test data"""
        graph = MemoryGraph()
        
        # Create test memories with strong semantic overlap to the test query
        memory1 = MemoryNode(
            concept="How to handle asynchronous operations in Python",
            summary="Complete guide on handling asynchronous operations in Python programming",
            full_content="How to handle asynchronous operations in Python using async await keywords and asyncio library",
            tags=["python", "async", "asynchronous", "operations", "handle"],
            keywords=["handle", "asynchronous", "operations", "python", "async", "await", "asyncio"]
        )
        
        memory2 = MemoryNode(
            concept="Python async await operations handling", 
            summary="Comprehensive guide to handling async operations with await in Python",
            full_content="Python async operations handling using await keywords for asynchronous programming",
            tags=["python", "async", "await", "operations", "handling"],
            keywords=["python", "async", "await", "operations", "handling", "asynchronous"]
        )
        
        memory3 = MemoryNode(
            concept="Asynchronous operations Python async handling",
            summary="Best practices for handling asynchronous operations in Python async programming",
            full_content="Asynchronous operations handling in Python using async programming techniques",
            tags=["asynchronous", "operations", "python", "async", "handling"],
            keywords=["asynchronous", "operations", "python", "async", "handling", "programming"]
        )
        
        # Add memories to graph
        id1 = graph.add_node(memory1)
        id2 = graph.add_node(memory2)
        id3 = graph.add_node(memory3)
        
        return graph
    
    @pytest.fixture
    def mock_embedding_search(self):
        """Create mock embedding search"""
        mock = MagicMock(spec=EmbeddingSearch)
        mock.get_embedding = AsyncMock(return_value=[0.1] * 384)
        mock.search_by_text = AsyncMock(return_value=[
            ("memory_id_1", 0.85),
            ("memory_id_2", 0.75),
            ("memory_id_3", 0.65)
        ])
        mock.is_ready = MagicMock(return_value=True)
        mock.initialize = AsyncMock()
        mock.store_embedding = MagicMock()
        return mock
    
    @pytest.fixture
    def mock_hybrid_retriever(self, memory_graph, mock_embedding_search):
        """Create mock hybrid retriever"""
        mock = MagicMock(spec=HybridRetriever)
        mock.memory_graph = memory_graph
        mock.embedding_search = mock_embedding_search
        
        # Mock search_memories to return actual memories from the graph
        async def mock_search_memories(query, max_results=10, use_graph_expansion=True):
            # Get all memory nodes for testing
            memories = list(memory_graph.nodes.values())
            return [
                {
                    "memory_id": memory.id,
                    "node": memory,
                    "combined_score": 0.8 - (i * 0.1),  # Decreasing scores
                    "source": "embedding"
                }
                for i, memory in enumerate(memories[:max_results])
            ]
        
        mock.search_memories = AsyncMock(side_effect=mock_search_memories)
        return mock
    
    @pytest.fixture
    def connection_agent(self, memory_graph):
        """Create connection agent"""
        return ConnectionAgent(memory_graph)
    
    @pytest.fixture
    def claude_client(self):
        """Create mock Claude client"""
        return MockClaudeClient()
    
    @pytest.fixture
    def test_app(self, memory_graph, mock_embedding_search, mock_hybrid_retriever, connection_agent, claude_client):
        """Create test app with all dependencies"""
        app = FastAPI()
        app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
        
        # Set up app state
        app.state.memory_graph = memory_graph
        app.state.embedding_search = mock_embedding_search
        app.state.hybrid_retriever = mock_hybrid_retriever
        app.state.connection_agent = connection_agent
        app.state.claude_client = claude_client
        
        return app
    
    @pytest.fixture
    def test_client(self, test_app):
        """Create test client"""
        with TestClient(test_app) as client:
            yield client
    
    def test_chat_creates_co_access_connections(self, test_client, memory_graph):
        """Test that chat API creates co-access connections between memories"""
        # Get initial connection count
        initial_connections = sum(len(node.connections) for node in memory_graph.nodes.values())
        
        # Send a chat message that should retrieve multiple relevant memories
        request_data = {
            "content": "How do I handle asynchronous operations in Python?",
            "conversation_history": []
        }
        
        response = test_client.post("/api/chat/send", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Should have retrieved multiple memories
        assert len(data["retrieved_memories"]) >= 2
        
        # Check that co-access connections were created
        final_connections = sum(len(node.connections) for node in memory_graph.nodes.values())
        assert final_connections > initial_connections
        
        # Verify that CO_ACCESS connections exist
        co_access_found = False
        for node in memory_graph.nodes.values():
            for connection in node.connections.values():
                if connection.connection_type == ConnectionType.CO_ACCESS:
                    co_access_found = True
                    break
            if co_access_found:
                break
        
        assert co_access_found, "No CO_ACCESS connections were created"
    
    def test_chat_response_includes_co_access_info(self, test_client):
        """Test that chat response includes co-access information"""
        request_data = {
            "content": "Explain async programming in Python",
            "conversation_history": []
        }
        
        response = test_client.post("/api/chat/send", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check response structure
        assert "message" in data
        assert "retrieved_memories" in data
        assert "memory_access_events" in data
        assert "processing_time" in data
        
        # Check that retrieved memories have relevance information
        for memory in data["retrieved_memories"]:
            assert "relevance_score" in memory
            assert "confidence" in memory
            assert "reasoning" in memory
            assert "must_keep" in memory
        
        # Check that memory access events have detailed information
        for event in data["memory_access_events"]:
            assert "relevance_score" in event
            assert "confidence" in event
            assert "reasoning" in event
    
    def test_repeated_chat_strengthens_connections(self, test_client, memory_graph):
        """Test that repeated chat interactions strengthen co-access connections"""
        # First chat interaction
        request_data = {
            "content": "How to use async/await in Python?",
            "conversation_history": []
        }
        
        response1 = test_client.post("/api/chat/send", json=request_data)
        assert response1.status_code == 200
        
        # Get connection weights after first interaction
        first_weights = {}
        for node_id, node in memory_graph.nodes.items():
            for target_id, connection in node.connections.items():
                if connection.connection_type == ConnectionType.CO_ACCESS:
                    first_weights[(node_id, target_id)] = connection.weight
        
        # Second chat interaction with similar content that should retrieve same memories
        request_data2 = {
            "content": "How do I handle asynchronous operations with Python async?",
            "conversation_history": []
        }
        
        response2 = test_client.post("/api/chat/send", json=request_data2)
        assert response2.status_code == 200
        
        # Check that some connections were strengthened
        strengthened_count = 0
        for node_id, node in memory_graph.nodes.items():
            for target_id, connection in node.connections.items():
                if connection.connection_type == ConnectionType.CO_ACCESS:
                    key = (node_id, target_id)
                    if key in first_weights:
                        if connection.weight > first_weights[key]:
                            strengthened_count += 1
        
        # At least some connections should have been strengthened
        assert strengthened_count > 0, "No connections were strengthened on repeated access"
    
    def test_chat_with_conversation_history(self, test_client):
        """Test chat with conversation history affects co-access patterns"""
        # Chat with conversation history that relates to our test memories
        request_data = {
            "content": "Can you give me more details about handling async operations?",
            "conversation_history": [
                {
                    "id": "msg_1",
                    "content": "What is asynchronous programming in Python?",
                    "role": "user",
                    "timestamp": 1234567890.0,
                    "memory_accesses": []
                },
                {
                    "id": "msg_2", 
                    "content": "Async programming allows handling asynchronous operations...",
                    "role": "assistant",
                    "timestamp": 1234567891.0,
                    "memory_accesses": []
                }
            ]
        }
        
        response = test_client.post("/api/chat/send", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Should still retrieve memories and create connections
        assert len(data["retrieved_memories"]) >= 1
        assert len(data["memory_access_events"]) >= 1
        
        # Response should mention the conversation context
        assert "processing_time" in data
        assert data["processing_time"] > 0
    
    def test_chat_api_handles_no_relevant_memories(self, test_client, mock_hybrid_retriever):
        """Test chat API behavior when no relevant memories are found"""
        # Mock hybrid retriever to return no results
        async def mock_search_no_results(query, max_results=10, use_graph_expansion=True):
            return []
        
        mock_hybrid_retriever.search_memories = AsyncMock(side_effect=mock_search_no_results)
        
        request_data = {
            "content": "Tell me about quantum physics",
            "conversation_history": []
        }
        
        response = test_client.post("/api/chat/send", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Should handle gracefully with no memories
        assert len(data["retrieved_memories"]) == 0
        assert len(data["memory_access_events"]) == 0
        assert "don't have specific memories" in data["message"]["content"]