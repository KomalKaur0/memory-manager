"""
Tests for co-access connection functionality
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.agents.connection_agent import ConnectionAgent, AccessEvent
from src.agents.relevance_agent import RelevanceAgent, QueryContext, QueryType
from src.core.memory_node import MemoryNode, ConnectionType
from src.core.memory_graph import MemoryGraph


class TestCoAccessFunctionality:
    """Test suite for co-access connection creation and learning"""
    
    @pytest.fixture
    def memory_graph(self):
        return MemoryGraph()
    
    @pytest.fixture
    def connection_agent(self, memory_graph):
        return ConnectionAgent(memory_graph)
    
    @pytest.fixture
    def relevance_agent(self, memory_graph):
        return RelevanceAgent(memory_graph=memory_graph)
    
    @pytest.fixture
    def test_memories(self, memory_graph):
        """Create test memories with related content"""
        memory1 = MemoryNode(
            concept="Python async programming",
            summary="Complete guide to async/await in Python",
            full_content="How to handle asynchronous operations in Python using async and await",
            tags=["python", "async", "programming"],
            keywords=["async", "await", "python", "asynchronous"]
        )
        
        memory2 = MemoryNode(
            concept="Python asyncio library",
            summary="Comprehensive guide to asyncio for async operations",
            full_content="Python asyncio library provides tools for asynchronous programming",
            tags=["python", "asyncio", "async"],
            keywords=["asyncio", "async", "python", "library"]
        )
        
        memory3 = MemoryNode(
            concept="Asynchronous patterns",
            summary="Best practices for asynchronous programming patterns",
            full_content="Patterns and best practices for effective asynchronous programming",
            tags=["async", "patterns", "programming"],
            keywords=["asynchronous", "patterns", "programming"]
        )
        
        id1 = memory_graph.add_node(memory1)
        id2 = memory_graph.add_node(memory2)
        id3 = memory_graph.add_node(memory3)
        
        return [(id1, memory1), (id2, memory2), (id3, memory3)]
    
    def test_co_access_connection_creation(self, connection_agent, test_memories):
        """Test that co-access connections are created for relevant memories"""
        memory_ids = [item[0] for item in test_memories[:2]]  # Use first 2 memories
        
        # Record co-access with good relevance scores
        connection_agent.record_co_access_with_feedback(
            memory_ids=memory_ids,
            query="How to use async/await in Python?",
            relevance_scores=[0.8, 0.7],  # Both above threshold
            response_quality=0.85,
            session_id="test-session"
        )
        
        # Check that bidirectional connections were created
        memory1 = connection_agent.memory_graph.get_node(memory_ids[0])
        memory2 = connection_agent.memory_graph.get_node(memory_ids[1])
        
        assert memory_ids[1] in memory1.connections
        assert memory_ids[0] in memory2.connections
        
        # Check connection properties
        connection_1_to_2 = memory1.connections[memory_ids[1]]
        connection_2_to_1 = memory2.connections[memory_ids[0]]
        
        assert connection_1_to_2.connection_type == ConnectionType.CO_ACCESS
        assert connection_2_to_1.connection_type == ConnectionType.CO_ACCESS
        assert connection_1_to_2.weight >= 0.3  # Minimum connection strength
        assert connection_2_to_1.weight >= 0.3
        
        # Connections should be strong due to good relevance and response quality
        assert connection_1_to_2.weight >= 0.6
        assert connection_2_to_1.weight >= 0.6
    
    def test_co_access_threshold_filtering(self, connection_agent, test_memories):
        """Test that connections are only created for memories above relevance threshold"""
        memory_ids = [item[0] for item in test_memories]
        
        # Record co-access with mixed relevance scores
        connection_agent.record_co_access_with_feedback(
            memory_ids=memory_ids,
            query="What is async programming?",
            relevance_scores=[0.8, 0.25, 0.2],  # Only first is above 0.3 threshold
            response_quality=0.75,
            session_id="test-session"
        )
        
        # Only the first memory should have no connections since no other memory
        # meets the threshold for pairing
        memory1 = connection_agent.memory_graph.get_node(memory_ids[0])
        memory2 = connection_agent.memory_graph.get_node(memory_ids[1])
        memory3 = connection_agent.memory_graph.get_node(memory_ids[2])
        
        # No connections should be created since only one memory is above threshold
        assert len(memory1.connections) == 0
        assert len(memory2.connections) == 0
        assert len(memory3.connections) == 0
    
    def test_co_access_connection_strengthening(self, connection_agent, test_memories):
        """Test that existing connections get strengthened on repeated co-access"""
        memory_ids = [item[0] for item in test_memories[:2]]
        
        # First co-access
        connection_agent.record_co_access_with_feedback(
            memory_ids=memory_ids,
            query="Python async programming",
            relevance_scores=[0.7, 0.6],
            response_quality=0.8,
            session_id="session-1"
        )
        
        memory1 = connection_agent.memory_graph.get_node(memory_ids[0])
        initial_weight = memory1.connections[memory_ids[1]].weight
        
        # Second co-access - should strengthen the connection
        connection_agent.record_co_access_with_feedback(
            memory_ids=memory_ids,
            query="How to handle async operations in Python?",
            relevance_scores=[0.8, 0.7],
            response_quality=0.9,
            session_id="session-2"
        )
        
        # Connection should be strengthened
        new_weight = memory1.connections[memory_ids[1]].weight
        assert new_weight > initial_weight
        assert new_weight <= 0.95  # Max weight cap
    
    def test_multiple_memory_co_access(self, connection_agent, test_memories):
        """Test co-access with 3+ memories creates multiple connections"""
        memory_ids = [item[0] for item in test_memories]
        
        # All memories have good relevance scores
        connection_agent.record_co_access_with_feedback(
            memory_ids=memory_ids,
            query="Comprehensive guide to async programming",
            relevance_scores=[0.9, 0.8, 0.7],  # All above threshold
            response_quality=0.85,
            session_id="multi-session"
        )
        
        # Check that multiple pairwise connections were created
        memory1 = connection_agent.memory_graph.get_node(memory_ids[0])
        memory2 = connection_agent.memory_graph.get_node(memory_ids[1])
        memory3 = connection_agent.memory_graph.get_node(memory_ids[2])
        
        # Memory 1 should connect to memories 2 and 3
        assert memory_ids[1] in memory1.connections
        assert memory_ids[2] in memory1.connections
        
        # Memory 2 should connect to memories 1 and 3
        assert memory_ids[0] in memory2.connections
        assert memory_ids[2] in memory2.connections
        
        # Memory 3 should connect to memories 1 and 2
        assert memory_ids[0] in memory3.connections
        assert memory_ids[1] in memory3.connections
    
    def test_relevance_agent_connection_strength_learning(self, relevance_agent):
        """Test that RelevanceAgent increases connection strength weight with learning"""
        initial_weight = relevance_agent.connection_strength_weight
        initial_count = relevance_agent.connection_modification_count
        
        # Record several connection modifications
        for _ in range(5):
            relevance_agent.record_connection_modification()
        
        # Connection strength weight should increase
        assert relevance_agent.connection_modification_count == initial_count + 5
        assert relevance_agent.connection_strength_weight > initial_weight
        
        # Weight should be capped at 0.6 (60%)
        for _ in range(20):  # Many more modifications
            relevance_agent.record_connection_modification()
        
        assert relevance_agent.connection_strength_weight <= 0.6
    
    def test_relevance_agent_connection_strength_scoring(self, memory_graph, test_memories):
        """Test that RelevanceAgent uses connection strength in relevance scoring"""
        relevance_agent = RelevanceAgent(memory_graph=memory_graph)
        
        # Create a connection between two memories
        memory_ids = [item[0] for item in test_memories[:2]]
        memory_graph.create_connection(
            memory_ids[0], memory_ids[1], ConnectionType.CO_ACCESS, 0.8
        )
        
        # Record some connection modifications to increase learning
        for _ in range(10):
            relevance_agent.record_connection_modification()
        
        # Evaluate relevance with connection strength
        memory = memory_graph.get_node(memory_ids[0])
        query_context = QueryContext(
            query="Test query",
            conversation_history=[],
            user_intent="test",
            domain="test"
        )
        
        # Score with connection reference should be higher
        score_with_connections = relevance_agent.evaluate_relevance(
            memory=memory,
            query="Test query", 
            context=query_context,
            reference_memory_ids=[memory_ids[1]]  # Reference the connected memory
        )
        
        # Score without connection reference should be lower
        score_without_connections = relevance_agent.evaluate_relevance(
            memory=memory,
            query="Test query",
            context=query_context,
            reference_memory_ids=[]  # No connection reference
        )
        
        # Connection strength should boost the relevance score
        assert score_with_connections.connection_strength_score > 0
        assert score_with_connections.overall >= score_without_connections.overall
    
    def test_co_access_frequency_tracking(self, connection_agent, test_memories):
        """Test that co-access frequency is tracked and affects connection strength"""
        memory_ids = [item[0] for item in test_memories[:2]]
        
        # Multiple co-accesses should increase connection strength
        for i in range(3):
            connection_agent.record_co_access_with_feedback(
                memory_ids=memory_ids,
                query=f"Query {i+1}",
                relevance_scores=[0.7, 0.6],
                response_quality=0.8,
                session_id=f"session-{i+1}"
            )
        
        # Check that co-access pattern was tracked
        pair = tuple(sorted(memory_ids))
        assert pair in connection_agent.co_access_patterns
        assert connection_agent.co_access_patterns[pair] >= 3
        
        # Connection should be very strong due to repeated co-access
        memory1 = connection_agent.memory_graph.get_node(memory_ids[0])
        connection_weight = memory1.connections[memory_ids[1]].weight
        assert connection_weight >= 0.8  # Should be strengthened by frequency
    
    def test_response_quality_impact(self, connection_agent, test_memories):
        """Test that response quality affects connection strength"""
        memory_ids = [item[0] for item in test_memories[:2]]
        
        # Test with high response quality
        connection_agent.record_co_access_with_feedback(
            memory_ids=memory_ids,
            query="High quality response test",
            relevance_scores=[0.7, 0.6],
            response_quality=0.95,  # Very high quality
            session_id="high-quality"
        )
        
        memory1 = connection_agent.memory_graph.get_node(memory_ids[0])
        high_quality_weight = memory1.connections[memory_ids[1]].weight
        
        # Reset for comparison
        del memory1.connections[memory_ids[1]]
        memory2 = connection_agent.memory_graph.get_node(memory_ids[1])
        del memory2.connections[memory_ids[0]]
        
        # Test with low response quality
        connection_agent.record_co_access_with_feedback(
            memory_ids=memory_ids,
            query="Low quality response test",
            relevance_scores=[0.7, 0.6],  # Same relevance scores
            response_quality=0.3,  # Low quality
            session_id="low-quality"
        )
        
        low_quality_weight = memory1.connections[memory_ids[1]].weight
        
        # High quality should result in stronger connections
        assert high_quality_weight > low_quality_weight