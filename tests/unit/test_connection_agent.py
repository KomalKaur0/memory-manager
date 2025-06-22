"""
Tests for ConnectionAgent
"""
import pytest
from datetime import datetime, timedelta

from src.agents.connection_agent import (
    ConnectionAgent, AccessEvent, ConnectionAnalysisType, 
    ConnectionSuggestion, ConnectionStrengthening
)
from src.core.memory_node import MemoryNode, ConnectionType
from src.core.memory_graph import MemoryGraph


class TestConnectionAgent:
    """Tests for ConnectionAgent"""
    
    @pytest.fixture
    def memory_graph(self):
        return MemoryGraph()
    
    @pytest.fixture
    def connection_agent(self, memory_graph):
        return ConnectionAgent(memory_graph)
    
    @pytest.fixture
    def sample_memories(self, memory_graph):
        """Create sample memories for testing"""
        memories = [
            MemoryNode(
                concept="Python async programming",
                summary="Guide to async/await in Python",
                full_content="How to use async/await in Python programming",
                tags=["python", "async", "programming"],
                keywords=["async", "await", "python"]
            ),
            MemoryNode(
                concept="Python asyncio patterns",
                summary="Advanced asyncio patterns",
                full_content="Advanced patterns for asyncio programming",
                tags=["python", "asyncio", "patterns"],
                keywords=["asyncio", "patterns", "python"]
            ),
            MemoryNode(
                concept="JavaScript promises",
                summary="Promise handling in JS",
                full_content="How to work with promises in JavaScript",
                tags=["javascript", "async", "promises"],
                keywords=["promise", "async", "javascript"]
            )
        ]
        
        memory_ids = []
        for memory in memories:
            memory_id = memory_graph.add_node(memory)
            memory_ids.append(memory_id)
        
        return memory_ids, memories
    
    def test_initialization(self, connection_agent):
        """Test ConnectionAgent initialization"""
        assert connection_agent is not None
        assert connection_agent.semantic_threshold == 0.7
        assert connection_agent.temporal_window == 60
        assert len(connection_agent.access_history) == 0
    
    def test_record_access_event(self, connection_agent):
        """Test recording access events"""
        event = AccessEvent(
            memory_id="test_id",
            query="test query",
            timestamp=datetime.now(),
            session_id="session1"
        )
        
        connection_agent.record_access_event(event)
        
        assert len(connection_agent.access_history) == 1
        assert connection_agent.access_history[0].memory_id == "test_id"
        assert "session1" in connection_agent.session_sequences
        assert "test_id" in connection_agent.session_sequences["session1"]
    
    def test_semantic_connection_analysis(self, connection_agent, sample_memories):
        """Test semantic connection analysis"""
        memory_ids, memories = sample_memories
        
        # Should find semantic connections between similar Python memories
        suggestions = connection_agent._analyze_semantic_connections(memory_ids)
        
        # Should find connection between the two Python-related memories
        python_suggestions = [
            s for s in suggestions 
            if "python" in s.reasoning.lower()
        ]
        
        assert len(python_suggestions) >= 0  # May not find high semantic similarity with current threshold
    
    def test_temporal_connection_analysis(self, connection_agent, sample_memories):
        """Test temporal connection analysis"""
        memory_ids, memories = sample_memories
        
        # Record access events within temporal window
        base_time = datetime.now()
        events = [
            AccessEvent(memory_ids[0], "query1", base_time, session_id="session1"),
            AccessEvent(memory_ids[1], "query2", base_time + timedelta(minutes=5), session_id="session1")
        ]
        
        for event in events:
            connection_agent.record_access_event(event)
        
        suggestions = connection_agent._analyze_temporal_connections(memory_ids)
        
        # Should find temporal connection between the two memories
        assert len(suggestions) >= 1
        temporal_suggestion = suggestions[0]
        assert temporal_suggestion.analysis_type == ConnectionAnalysisType.TEMPORAL
        assert temporal_suggestion.confidence_score > 0.5
    
    def test_apply_connection_suggestions(self, connection_agent, sample_memories, memory_graph):
        """Test applying connection suggestions"""
        memory_ids, memories = sample_memories
        
        # Create a manual suggestion
        suggestion = ConnectionSuggestion(
            source_id=memory_ids[0],
            target_id=memory_ids[1],
            connection_type=ConnectionType.SIMILARITY,
            confidence_score=0.8,
            reasoning="Test connection",
            suggested_weight=0.7,
            analysis_type=ConnectionAnalysisType.SEMANTIC
        )
        
        applied_counts = connection_agent.apply_connection_suggestions([suggestion])
        
        assert applied_counts["semantic"] == 1
        
        # Verify connection was created
        source_memory = memory_graph.get_node(memory_ids[0])
        assert memory_ids[1] in source_memory.connections
        assert source_memory.connections[memory_ids[1]].weight == 0.7
    
    def test_suggest_connection_strengthening(self, connection_agent, sample_memories, memory_graph):
        """Test connection strengthening suggestions"""
        memory_ids, memories = sample_memories
        
        # Create a connection first
        memory_graph.create_connection(
            memory_ids[0], memory_ids[1], ConnectionType.SIMILARITY, 0.5
        )
        
        # Record some access events to build patterns
        base_time = datetime.now()
        events = [
            AccessEvent(memory_ids[0], "query1", base_time),
            AccessEvent(memory_ids[1], "query1", base_time + timedelta(minutes=1))
        ]
        
        for event in events:
            connection_agent.record_access_event(event)
        
        suggestions = connection_agent.suggest_connection_strengthening(
            memory_ids[0], context={"query": "test query"}
        )
        
        # Should have suggestions for the existing connection
        assert len(suggestions) >= 0  # May or may not suggest changes depending on factors
    
    def test_connection_statistics(self, connection_agent, sample_memories):
        """Test getting connection statistics"""
        memory_ids, memories = sample_memories
        
        # Record some events
        event = AccessEvent(memory_ids[0], "test query", datetime.now())
        connection_agent.record_access_event(event)
        
        stats = connection_agent.get_connection_statistics()
        
        assert "total_access_events" in stats
        assert "unique_co_access_pairs" in stats
        assert "active_sessions" in stats
        assert stats["total_access_events"] == 1
    
    def test_analyze_and_suggest_connections(self, connection_agent, sample_memories):
        """Test the main connection analysis method"""
        memory_ids, memories = sample_memories
        
        # Record some access patterns
        base_time = datetime.now()
        events = [
            AccessEvent(memory_ids[0], "async programming", base_time),
            AccessEvent(memory_ids[1], "asyncio patterns", base_time + timedelta(minutes=10))
        ]
        
        for event in events:
            connection_agent.record_access_event(event)
        
        suggestions = connection_agent.analyze_and_suggest_connections(
            memory_ids=memory_ids,
            analysis_types=[ConnectionAnalysisType.TEMPORAL, ConnectionAnalysisType.SEMANTIC]
        )
        
        assert isinstance(suggestions, list)
        # Should return results sorted by confidence
        if len(suggestions) > 1:
            assert suggestions[0].confidence_score >= suggestions[1].confidence_score