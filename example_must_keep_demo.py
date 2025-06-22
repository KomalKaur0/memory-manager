#!/usr/bin/env python3
"""
Demonstration of the must-keep flagging system and more generous memory limits.
"""

from src.agents import (
    RelevanceAgent, FilterAgent, 
    QueryContext, UserPreferences, ResponseContext,
    QueryType
)
from src.core.memory_node import MemoryNode


def demonstrate_must_keep_system():
    """Demonstrate the must-keep flagging system with various memory types"""
    
    # Initialize agents
    relevance_agent = RelevanceAgent()
    filter_agent = FilterAgent()
    
    # Create diverse memories including some that should trigger must-keep flags
    memories = [
        # High-importance critical memory
        MemoryNode(
            concept="Python async programming fundamentals",
            summary="Essential guide to async/await in Python",
            full_content="Critical foundational knowledge for asynchronous programming in Python, covering asyncio, async/await syntax, and core concepts.",
            tags=["python", "async", "programming", "fundamental", "essential"],
            keywords=["asyncio", "await", "coroutines", "async", "python"],
            importance_score=0.95  # High importance should trigger must-keep
        ),
        
        # Recently accessed memory
        MemoryNode(
            concept="Python async patterns",
            summary="Advanced async patterns in Python",
            full_content="Advanced patterns for asyncio programming including task management and concurrency patterns.",
            tags=["python", "asyncio", "advanced"],
            keywords=["asyncio", "patterns", "concurrency", "python"],
            access_count=6  # High access count should trigger must-keep
        ),
        
        # Memory with exact concept match
        MemoryNode(
            concept="async await Python tutorial",
            summary="How to use async/await in Python",
            full_content="Complete tutorial on using async/await syntax in Python with practical examples.",
            tags=["python", "async", "tutorial"],
            keywords=["async", "await", "python", "tutorial"]
        ),
        
        # Critical tagged memory
        MemoryNode(
            concept="Asyncio best practices",
            summary="Critical best practices for asyncio",
            full_content="Important guidelines and best practices for writing efficient asyncio code.",
            tags=["python", "asyncio", "critical", "best-practices"],
            keywords=["asyncio", "practices", "guidelines"]
        ),
        
        # Regular relevant memory
        MemoryNode(
            concept="JavaScript Promises",
            summary="Understanding JavaScript Promises",
            full_content="Guide to working with Promises in JavaScript.",
            tags=["javascript", "async", "promises"],
            keywords=["promise", "async", "javascript"]
        ),
        
        # Somewhat relevant memory
        MemoryNode(
            concept="Database async operations",
            summary="Async database operations",
            full_content="How to perform asynchronous database operations.",
            tags=["database", "async"],
            keywords=["database", "async", "operations"]
        ),
        
        # Low relevance memory
        MemoryNode(
            concept="CSS animations",
            summary="Creating smooth CSS animations",
            full_content="Guide to creating beautiful CSS animations.",
            tags=["css", "animation", "frontend"],
            keywords=["css", "animation", "smooth"]
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
    
    print("=== Must-Keep Flagging System Demonstration ===")
    print(f"Query: {query}")
    print(f"Total candidate memories: {len(memories)}")
    print()
    
    # Step 1: Evaluate relevance and identify must-keep memories
    print("Step 1: RelevanceAgent - Evaluating memories and checking must-keep criteria")
    print("-" * 70)
    
    relevance_scores = []
    must_keep_count = 0
    
    for i, memory in enumerate(memories):
        score = relevance_agent.evaluate_relevance(memory, query, context)
        relevance_scores.append(score)
        
        if score.must_keep:
            must_keep_count += 1
        
        print(f"Memory {i+1}: {memory.concept}")
        print(f"  Overall Score: {score.overall:.3f}")
        print(f"  Must-Keep: {'YES' if score.must_keep else 'NO'}")
        print(f"  Access Count: {memory.access_count}")
        print(f"  Importance: {memory.importance_score:.2f}")
        print(f"  Tags: {memory.tags}")
        print(f"  Reasoning: {score.reasoning}")
        print()
    
    print(f"Must-keep memories identified: {must_keep_count}")
    print()
    
    # Step 2: Test with restrictive user preferences
    print("Step 2: FilterAgent - Testing with restrictive preferences")
    print("-" * 70)
    
    restrictive_preferences = UserPreferences(
        max_memories=3,          # Very restrictive limit
        prefer_recent=True,
        avoid_redundancy=True,
        relevance_threshold=0.7,  # High threshold
        diversity_factor=0.8
    )
    
    response_context = ResponseContext(
        response_type="chat",
        user_context="learning",
        conversation_history=context.conversation_history,
        platform="mobile"  # More restrictive platform
    )
    
    filter_result = filter_agent.filter_for_response(
        candidate_memories=memories,
        relevance_scores=relevance_scores,
        user_preferences=restrictive_preferences,
        response_context=response_context
    )
    
    print(f"Restrictive Filtering Result:")
    print(f"  Selected: {len(filter_result.selected_memories)} memories")
    print(f"  Filtered out: {filter_result.filtered_count} memories")
    print(f"  Reasoning: {filter_result.reasoning}")
    print()
    
    must_keep_in_result = sum(1 for score in filter_result.relevance_scores if score.must_keep)
    print(f"Must-keep memories in final result: {must_keep_in_result}")
    print()
    
    # Step 3: Show final selected memories
    print("Step 3: Final Selected Memories (with must-keep protection)")
    print("-" * 70)
    
    for i, (memory, score) in enumerate(zip(filter_result.selected_memories, filter_result.relevance_scores)):
        must_keep_indicator = " 🔒 MUST-KEEP" if score.must_keep else ""
        print(f"Selected Memory {i+1}: {memory.concept}{must_keep_indicator}")
        print(f"  Relevance Score: {score.overall:.3f}")
        print(f"  Must-Keep Flag: {score.must_keep}")
        print(f"  Tags: {memory.tags}")
        print(f"  Access Count: {memory.access_count}")
        print(f"  Importance: {memory.importance_score:.2f}")
        print()
    
    # Step 4: Demonstrate generous limits
    print("Step 4: Testing with generous preferences (new defaults)")
    print("-" * 70)
    
    generous_preferences = UserPreferences()  # Use new generous defaults
    
    generous_context = ResponseContext(
        response_type="chat",
        user_context="learning", 
        conversation_history=context.conversation_history,
        platform="web"  # More generous platform
    )
    
    generous_result = filter_agent.filter_for_response(
        candidate_memories=memories,
        relevance_scores=relevance_scores,
        user_preferences=generous_preferences,
        response_context=generous_context
    )
    
    print(f"Generous Filtering Result:")
    print(f"  Max memories allowed: {generous_preferences.max_memories}")
    print(f"  Relevance threshold: {generous_preferences.relevance_threshold}")
    print(f"  Platform limit (web): {filter_agent.platform_limits['web']['max_memories']}")
    print(f"  Selected: {len(generous_result.selected_memories)} memories")
    print(f"  Filtered out: {generous_result.filtered_count} memories")
    print(f"  Reasoning: {generous_result.reasoning}")
    print()
    
    return filter_result, generous_result


if __name__ == "__main__":
    demonstrate_must_keep_system()