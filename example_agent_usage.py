#!/usr/bin/env python3
"""
Example demonstrating how to use RelevanceAgent and FilterAgent together
for intelligent memory filtering and selection.
"""

from src.agents import (
    RelevanceAgent, FilterAgent, 
    QueryContext, UserPreferences, ResponseContext,
    QueryType
)
from src.core.memory_node import MemoryNode


def example_agent_workflow():
    """Demonstrate the complete RelevanceAgent -> FilterAgent workflow"""
    
    # Initialize agents
    relevance_agent = RelevanceAgent()
    filter_agent = FilterAgent()
    
    # Create sample memories
    memories = [
        MemoryNode(
            concept="Python async programming",
            summary="Complete guide to async/await in Python",
            full_content="Comprehensive tutorial covering asyncio, async/await syntax, coroutines, and best practices for asynchronous programming in Python.",
            tags=["python", "async", "programming", "tutorial"],
            keywords=["asyncio", "await", "coroutines", "async", "python"]
        ),
        MemoryNode(
            concept="JavaScript Promises",
            summary="Understanding JavaScript Promises",
            full_content="Guide to working with Promises in JavaScript, including async/await syntax and error handling.",
            tags=["javascript", "async", "promises"],
            keywords=["promise", "async", "await", "javascript"]
        ),
        MemoryNode(
            concept="Database optimization",
            summary="SQL query optimization techniques",
            full_content="Best practices for optimizing database queries, indexing strategies, and performance tuning.",
            tags=["database", "sql", "optimization"],
            keywords=["index", "query", "performance", "sql"]
        ),
        MemoryNode(
            concept="Python asyncio patterns",
            summary="Advanced asyncio programming patterns",
            full_content="Advanced patterns and best practices for asyncio programming in Python, including task management and concurrency patterns.",
            tags=["python", "asyncio", "advanced"],
            keywords=["asyncio", "patterns", "concurrency", "python"]
        )
    ]
    
    # Set up query context
    query = "How to use async/await in Python?"
    context = QueryContext(
        query=query,
        conversation_history=["What is asyncio?", "Can you explain coroutines?"],
        user_intent="learning",
        domain="programming",
        query_type=QueryType.SEARCH
    )
    
    print("=== Memory Evaluation and Filtering Example ===")
    print(f"Query: {query}")
    print(f"Context domain: {context.domain}")
    print(f"Conversation history: {context.conversation_history}")
    print()
    
    # Step 1: Evaluate relevance for each memory
    print("Step 1: RelevanceAgent - Evaluating each memory")
    print("-" * 50)
    
    relevance_scores = []
    for i, memory in enumerate(memories):
        score = relevance_agent.evaluate_relevance(memory, query, context)
        relevance_scores.append(score)
        
        print(f"Memory {i+1}: {memory.concept}")
        print(f"  Overall Score: {score.overall:.3f}")
        print(f"  Semantic: {score.semantic_score:.3f}")
        print(f"  Context: {score.context_score:.3f}")
        print(f"  Temporal: {score.temporal_score:.3f}")
        print(f"  Topical: {score.topical_score:.3f}")
        print(f"  Confidence: {score.confidence:.3f}")
        print(f"  Reasoning: {score.reasoning}")
        print()
    
    # Step 2: Apply FilterAgent for final selection
    print("Step 2: FilterAgent - Making final selection decisions")
    print("-" * 50)
    
    user_preferences = UserPreferences(
        max_memories=3,
        prefer_recent=True,
        avoid_redundancy=True,
        relevance_threshold=0.4,
        diversity_factor=0.7
    )
    
    response_context = ResponseContext(
        response_type="chat",
        user_context="learning",
        conversation_history=context.conversation_history,
        platform="web"
    )
    
    filter_result = filter_agent.filter_for_response(
        candidate_memories=memories,
        relevance_scores=relevance_scores,
        user_preferences=user_preferences,
        response_context=response_context
    )
    
    print(f"Filter Result:")
    print(f"  Selected: {len(filter_result.selected_memories)} memories")
    print(f"  Filtered out: {filter_result.filtered_count} memories")
    print(f"  Reasoning: {filter_result.reasoning}")
    print(f"  Filter details: {filter_result.filter_details}")
    print()
    
    # Step 3: Show final selected memories
    print("Step 3: Final Selected Memories")
    print("-" * 50)
    
    for i, (memory, score) in enumerate(zip(filter_result.selected_memories, filter_result.relevance_scores)):
        print(f"Selected Memory {i+1}: {memory.concept}")
        print(f"  Relevance Score: {score.overall:.3f}")
        print(f"  Tags: {memory.tags}")
        print(f"  Summary: {memory.summary}")
        print()
    
    return filter_result


if __name__ == "__main__":
    example_agent_workflow()