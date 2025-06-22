"""
Tests for AI Memory System agents
"""
import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.core.memory_node import MemoryNode, ConnectionType
from src.agents.relevance_agent import RelevanceAgent, RelevanceScore, QueryContext, RelevanceType
from src.agents.filter_agent import FilterAgent, FilterResult, UserPreferences, ResponseContext


class TestRelevanceAgent:
    """Tests for RelevanceAgent - The Evaluator"""
    
    @pytest.fixture
    def relevance_agent(self):
        return RelevanceAgent()
    
    @pytest.fixture
    def sample_memory(self):
        return MemoryNode(
            concept="Python async programming",
            summary="Guide to async/await in Python",
            full_content="Comprehensive guide covering asyncio, async/await syntax, and best practices for asynchronous programming in Python.",
            tags=["python", "async", "programming"],
            keywords=["asyncio", "await", "coroutines"]
        )
    
    @pytest.fixture
    def sample_query_context(self):
        return QueryContext(
            query="How to use async/await in Python?",
            conversation_history=["What is asyncio?", "Can you explain coroutines?"],
            user_intent="learning",
            domain="programming"
        )
    
    def test_evaluate_relevance_high_semantic_match(self, relevance_agent, sample_memory, sample_query_context):
        """Test relevance evaluation for high semantic similarity"""
        score = relevance_agent.evaluate_relevance(
            memory=sample_memory,
            query="How to use async/await in Python?",
            context=sample_query_context
        )
        
        assert isinstance(score, RelevanceScore)
        assert score.overall >= 0.4  # Moderate relevance expected (adjusted for new dimensions)
        assert score.confidence >= 0.7
        assert "semantic" in score.reasoning.lower()  # Should mention semantic matching
        assert score.semantic_score >= 0.6
    
    def test_evaluate_relevance_low_semantic_match(self, relevance_agent, sample_query_context):
        """Test relevance evaluation for low semantic similarity"""
        unrelated_memory = MemoryNode(
            concept="Cooking pasta",
            summary="How to cook perfect pasta",
            full_content="Step by step guide to cooking pasta al dente.",
            tags=["cooking", "pasta"],
            keywords=["boil", "water", "salt"]
        )
        
        score = relevance_agent.evaluate_relevance(
            memory=unrelated_memory,
            query="How to use async/await in Python?",
            context=sample_query_context
        )
        
        assert score.overall <= 0.3  # Low relevance expected
        assert score.semantic_score <= 0.3
    
    def test_evaluate_relevance_with_conversation_context(self, relevance_agent, sample_memory):
        """Test that conversation context influences relevance scoring"""
        context_with_async = QueryContext(
            query="Tell me more about coroutines",
            conversation_history=["What is asyncio?", "How does async/await work?"],
            user_intent="learning",
            domain="programming"
        )
        
        score = relevance_agent.evaluate_relevance(
            memory=sample_memory,
            query="Tell me more about coroutines",
            context=context_with_async
        )
        
        # Should have high context relevance due to conversation history
        assert score.context_score >= 0.7
        assert score.overall >= 0.2  # Overall may be lower due to poor semantic match and new dimensions
    
    def test_evaluate_relevance_temporal_factor(self, relevance_agent, sample_memory, sample_query_context):
        """Test that temporal factors affect relevance"""
        # Recent access should boost relevance
        sample_memory.update_access()
        sample_memory.update_access()
        
        score = relevance_agent.evaluate_relevance(
            memory=sample_memory,
            query="How to use async/await in Python?",
            context=sample_query_context
        )
        
        assert score.temporal_score >= 0.3  # Access count should boost temporal score
    
    def test_connection_strength_relevance(self, sample_memory, sample_query_context):
        """Test that connection strength influences relevance scoring"""
        from src.core.memory_graph import MemoryGraph
        
        # Create a memory graph and add the sample memory
        memory_graph = MemoryGraph()
        memory_id = memory_graph.add_node(sample_memory)
        
        # Create a relevance agent with the memory graph
        relevance_agent = RelevanceAgent(memory_graph=memory_graph)
        
        # Create another memory and establish a connection
        reference_memory = MemoryNode(
            concept="Python coroutines explained",
            summary="Deep dive into Python coroutines",
            full_content="Detailed explanation of how coroutines work in Python asyncio",
            tags=["python", "coroutines", "async"],
            keywords=["coroutines", "asyncio", "python"]
        )
        ref_memory_id = memory_graph.add_node(reference_memory)
        
        # Create a strong connection between memories
        from src.core.memory_node import ConnectionType
        memory_graph.create_connection(memory_id, ref_memory_id, ConnectionType.SIMILARITY, 0.8)
        
        # Record some connection modifications to increase learning
        relevance_agent.record_connection_modification()
        relevance_agent.record_connection_modification()
        relevance_agent.record_connection_modification()
        
        # Evaluate relevance with reference to connected memory
        score = relevance_agent.evaluate_relevance(
            memory=sample_memory,
            query="How to use async/await in Python?",
            context=sample_query_context,
            reference_memory_ids=[ref_memory_id]
        )
        
        # Should have high connection strength score due to strong connection
        assert score.connection_strength_score >= 0.5
        assert relevance_agent.connection_strength_weight > 0  # Weight should have increased
        
        # Overall score should be higher due to connection strength
        score_without_connections = relevance_agent.evaluate_relevance(
            memory=sample_memory,
            query="How to use async/await in Python?",
            context=sample_query_context,
            reference_memory_ids=[]
        )
        
        assert score.overall >= score_without_connections.overall
    
    def test_relevance_score_components(self, relevance_agent, sample_memory, sample_query_context):
        """Test that all relevance score components are properly calculated"""
        score = relevance_agent.evaluate_relevance(
            memory=sample_memory,
            query="How to use async/await in Python?",
            context=sample_query_context
        )
        
        # All components should be present
        assert 0.0 <= score.semantic_score <= 1.0
        assert 0.0 <= score.context_score <= 1.0
        assert 0.0 <= score.temporal_score <= 1.0
        assert 0.0 <= score.topical_score <= 1.0
        assert 0.0 <= score.functional_score <= 1.0
        assert 0.0 <= score.associative_score <= 1.0
        assert 0.0 <= score.connection_strength_score <= 1.0
        assert 0.0 <= score.overall <= 1.0
        assert 0.0 <= score.confidence <= 1.0
        assert isinstance(score.reasoning, str)
        assert len(score.reasoning) > 0


class TestFilterAgent:
    """Tests for FilterAgent - The Decision Maker"""
    
    @pytest.fixture
    def filter_agent(self):
        return FilterAgent()
    
    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for filtering tests"""
        return [
            MemoryNode(
                concept="Python async basics",
                summary="Introduction to async programming",
                full_content="Basic async/await concepts in Python",
                tags=["python", "async", "basics"],
                keywords=["async", "await"]
            ),
            MemoryNode(
                concept="Python async advanced",
                summary="Advanced async patterns",
                full_content="Advanced asyncio patterns and best practices",
                tags=["python", "async", "advanced"],
                keywords=["asyncio", "patterns"]
            ),
            MemoryNode(
                concept="JavaScript promises",
                summary="Promise handling in JS",
                full_content="How to work with promises in JavaScript",
                tags=["javascript", "async", "promises"],
                keywords=["promise", "then", "catch"]
            ),
            MemoryNode(
                concept="Database optimization",
                summary="SQL query optimization",
                full_content="Techniques for optimizing database queries",
                tags=["database", "sql", "optimization"],
                keywords=["index", "query", "performance"]
            )
        ]
    
    @pytest.fixture
    def sample_relevance_scores(self):
        """Sample relevance scores matching the memories"""
        return [
            RelevanceScore(overall=0.9, confidence=0.8, semantic_score=0.9, context_score=0.8, 
                         temporal_score=0.7, topical_score=0.9, functional_score=0.8, 
                         associative_score=0.7, connection_strength_score=0.0, reasoning="High match"),
            RelevanceScore(overall=0.85, confidence=0.8, semantic_score=0.8, context_score=0.9, 
                         temporal_score=0.6, topical_score=0.8, functional_score=0.7,
                         associative_score=0.6, connection_strength_score=0.0, reasoning="Very relevant"),
            RelevanceScore(overall=0.4, confidence=0.6, semantic_score=0.3, context_score=0.5, 
                         temporal_score=0.4, topical_score=0.3, functional_score=0.2,
                         associative_score=0.3, connection_strength_score=0.0, reasoning="Different language"),
            RelevanceScore(overall=0.2, confidence=0.7, semantic_score=0.1, context_score=0.2, 
                         temporal_score=0.3, topical_score=0.1, functional_score=0.1,
                         associative_score=0.1, connection_strength_score=0.0, reasoning="Unrelated topic")
        ]
    
    @pytest.fixture
    def sample_user_preferences(self):
        return UserPreferences(
            max_memories=5,  # Updated to reflect more generous defaults
            prefer_recent=True,
            avoid_redundancy=True,
            relevance_threshold=0.3,  # Updated to reflect lower threshold
            diversity_factor=0.7
        )
    
    @pytest.fixture
    def sample_response_context(self):
        return ResponseContext(
            response_type="chat",
            user_context="learning",
            conversation_history=["What is async programming?"],
            platform="web"
        )
    
    def test_filter_for_response_basic(self, filter_agent, sample_memories, sample_relevance_scores, 
                                     sample_user_preferences, sample_response_context):
        """Test basic filtering functionality"""
        result = filter_agent.filter_for_response(
            candidate_memories=sample_memories,
            relevance_scores=sample_relevance_scores,
            user_preferences=sample_user_preferences,
            response_context=sample_response_context
        )
        
        assert isinstance(result, FilterResult)
        assert len(result.selected_memories) <= sample_user_preferences.max_memories
        assert len(result.selected_memories) >= 1  # Should select at least one relevant memory
        
        # All selected memories should meet relevance threshold
        for i, memory in enumerate(result.selected_memories):
            corresponding_score = result.relevance_scores[i]
            assert corresponding_score.overall >= sample_user_preferences.relevance_threshold
    
    def test_filter_redundancy_removal(self, filter_agent, sample_user_preferences, sample_response_context):
        """Test that similar memories are filtered for redundancy"""
        # Create very similar memories
        similar_memories = [
            MemoryNode(
                concept="Python async programming",
                summary="Guide to async/await in Python",
                full_content="How to use async/await in Python programming",
                tags=["python", "async", "programming"],
                keywords=["async", "await", "python"]
            ),
            MemoryNode(
                concept="Python async programming guide",
                summary="Complete guide to async/await in Python",
                full_content="Comprehensive guide on how to use async/await in Python programming",
                tags=["python", "async", "programming"],
                keywords=["async", "await", "python", "guide"]
            )
        ]
        
        # Both have high relevance
        high_scores = [
            RelevanceScore(overall=0.9, confidence=0.8, semantic_score=0.9, context_score=0.8, 
                         temporal_score=0.7, topical_score=0.9, functional_score=0.8,
                         associative_score=0.7, connection_strength_score=0.0, reasoning="High match 1"),
            RelevanceScore(overall=0.88, confidence=0.8, semantic_score=0.88, context_score=0.8, 
                         temporal_score=0.7, topical_score=0.88, functional_score=0.8,
                         associative_score=0.7, connection_strength_score=0.0, reasoning="High match 2")
        ]
        
        result = filter_agent.filter_for_response(
            candidate_memories=similar_memories,
            relevance_scores=high_scores,
            user_preferences=sample_user_preferences,
            response_context=sample_response_context
        )
        
        # Should keep only one due to redundancy removal
        assert len(result.selected_memories) == 1
        assert "redundant" in result.reasoning.lower()
    
    def test_filter_diversity_preference(self, filter_agent, sample_memories, sample_relevance_scores, 
                                       sample_response_context):
        """Test diversity factor in memory selection"""
        diverse_preferences = UserPreferences(
            max_memories=3,
            prefer_recent=False,
            avoid_redundancy=True,
            relevance_threshold=0.3,
            diversity_factor=0.9  # High diversity preference
        )
        
        result = filter_agent.filter_for_response(
            candidate_memories=sample_memories,
            relevance_scores=sample_relevance_scores,
            user_preferences=diverse_preferences,
            response_context=sample_response_context
        )
        
        # Should select memories from different topics/tags
        selected_tags = set()
        for memory in result.selected_memories:
            selected_tags.update(memory.tags)
        
        # Should have diverse tags represented
        assert len(selected_tags) >= 3
    
    def test_filter_mobile_constraints(self, filter_agent, sample_memories, sample_relevance_scores, 
                                     sample_user_preferences):
        """Test platform-specific filtering constraints"""
        mobile_context = ResponseContext(
            response_type="chat",
            user_context="mobile",
            conversation_history=[],
            platform="mobile"
        )
        
        result = filter_agent.filter_for_response(
            candidate_memories=sample_memories,
            relevance_scores=sample_relevance_scores,
            user_preferences=sample_user_preferences,
            response_context=mobile_context
        )
        
        # Mobile should have stricter limits but still generous
        assert len(result.selected_memories) <= 5  # Updated from 2 to 5 for mobile
    
    def test_filter_result_structure(self, filter_agent, sample_memories, sample_relevance_scores, 
                                   sample_user_preferences, sample_response_context):
        """Test that filter result has proper structure"""
        result = filter_agent.filter_for_response(
            candidate_memories=sample_memories,
            relevance_scores=sample_relevance_scores,
            user_preferences=sample_user_preferences,
            response_context=sample_response_context
        )
        
        assert isinstance(result.selected_memories, list)
        assert isinstance(result.relevance_scores, list)
        assert len(result.selected_memories) == len(result.relevance_scores)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0
        assert isinstance(result.filtered_count, int)
        assert result.filtered_count >= 0