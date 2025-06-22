#!/usr/bin/env python3
"""
Complete demonstration of all three agents working together:
RelevanceAgent + FilterAgent + ConnectionAgent
"""

from datetime import datetime, timedelta
from src.agents import (
    RelevanceAgent, FilterAgent, ConnectionAgent,
    QueryContext, UserPreferences, ResponseContext,
    AccessEvent, ConnectionAnalysisType,
    QueryType
)
from src.core.memory_node import MemoryNode
from src.core.memory_graph import MemoryGraph


def demonstrate_complete_agent_system():
    """Demonstrate all three agents working together in a realistic scenario"""
    
    # Initialize the memory graph and agents
    memory_graph = MemoryGraph()
    relevance_agent = RelevanceAgent()
    filter_agent = FilterAgent()
    connection_agent = ConnectionAgent(memory_graph)
    
    print("=== Complete AI Memory Agent System Demonstration ===")
    print("Showing RelevanceAgent + FilterAgent + ConnectionAgent working together")
    print()
    
    # Step 1: Create and add memories to the graph
    print("Step 1: Setting up memory graph with diverse memories")
    print("-" * 60)
    
    memories = [
        MemoryNode(
            concept="Python async programming fundamentals",
            summary="Essential guide to async/await in Python",
            full_content="Core concepts of asynchronous programming in Python including asyncio, async/await syntax, coroutines, and event loops.",
            tags=["python", "async", "programming", "fundamental"],
            keywords=["asyncio", "await", "coroutines", "async", "python"],
            importance_score=0.95
        ),
        MemoryNode(
            concept="asyncio task management",
            summary="Managing asyncio tasks and concurrency",
            full_content="Advanced techniques for creating, managing, and coordinating asyncio tasks in Python applications.",
            tags=["python", "asyncio", "tasks", "concurrency"],
            keywords=["tasks", "asyncio", "create_task", "gather", "python"]
        ),
        MemoryNode(
            concept="async database operations",
            summary="Asynchronous database interactions",
            full_content="How to perform database operations asynchronously using asyncpg, aiomysql, and other async database drivers.",
            tags=["database", "async", "python"],
            keywords=["asyncpg", "database", "async", "sql"]
        ),
        MemoryNode(
            concept="Python error handling patterns",
            summary="Best practices for error handling in Python",
            full_content="Comprehensive guide to exception handling, try/catch blocks, and error recovery strategies in Python.",
            tags=["python", "error-handling", "exceptions"],
            keywords=["try", "except", "exceptions", "error", "python"]
        ),
        MemoryNode(
            concept="asyncio error handling",
            summary="Error handling in async Python code",
            full_content="Special considerations for handling errors in asynchronous Python code, including task exceptions and timeouts.",
            tags=["python", "asyncio", "error-handling"],
            keywords=["async", "exceptions", "timeout", "error", "asyncio"]
        )
    ]
    
    # Add memories to graph
    memory_ids = []
    for memory in memories:
        memory_id = memory_graph.add_node(memory)
        memory_ids.append(memory_id)
        print(f"Added: {memory.concept} (ID: {memory_id[:8]}...)")
    
    print(f"\nTotal memories in graph: {len(memory_graph.nodes)}")
    print()
    
    # Step 2: Simulate user interactions and access patterns
    print("Step 2: Simulating user access patterns")
    print("-" * 60)
    
    # Simulate a user session exploring async programming
    base_time = datetime.now()
    access_events = [
        AccessEvent(memory_ids[0], "How to use async/await in Python?", base_time, "user1", "session1", 1),
        AccessEvent(memory_ids[1], "asyncio task management", base_time + timedelta(minutes=5), "user1", "session1", 2),
        AccessEvent(memory_ids[4], "async error handling", base_time + timedelta(minutes=10), "user1", "session1", 3),
        AccessEvent(memory_ids[0], "async programming basics", base_time + timedelta(minutes=30), "user2", "session2", 1),
        AccessEvent(memory_ids[2], "database async operations", base_time + timedelta(minutes=35), "user2", "session2", 2),
        AccessEvent(memory_ids[1], "asyncio tasks", base_time + timedelta(minutes=45), "user1", "session3", 1),
        AccessEvent(memory_ids[4], "error handling async", base_time + timedelta(minutes=50), "user1", "session3", 2),
    ]
    
    for event in access_events:
        connection_agent.record_access_event(event)
        # Update access count on the actual memory
        memory = memory_graph.get_node(event.memory_id)
        if memory:
            memory.update_access()
    
    print(f"Recorded {len(access_events)} access events across 3 sessions")
    print("Access patterns:")
    for event in access_events:
        memory = memory_graph.get_node(event.memory_id)
        print(f"  {event.timestamp.strftime('%H:%M')} - {memory.concept[:30]}... (Session: {event.session_id})")
    print()
    
    # Step 3: ConnectionAgent analyzes and suggests connections
    print("Step 3: ConnectionAgent - Analyzing and suggesting memory connections")
    print("-" * 60)
    
    # Analyze different types of connections
    connection_suggestions = connection_agent.analyze_and_suggest_connections(
        memory_ids=memory_ids,
        analysis_types=[
            ConnectionAnalysisType.SEMANTIC,
            ConnectionAnalysisType.TEMPORAL,
            ConnectionAnalysisType.CONTEXTUAL,
            ConnectionAnalysisType.SEQUENTIAL,
            ConnectionAnalysisType.ASSOCIATIVE
        ]
    )
    
    print(f"ConnectionAgent found {len(connection_suggestions)} connection suggestions:")
    for i, suggestion in enumerate(connection_suggestions[:5]):  # Show top 5
        source_mem = memory_graph.get_node(suggestion.source_id)
        target_mem = memory_graph.get_node(suggestion.target_id)
        print(f"  {i+1}. {source_mem.concept[:25]}... → {target_mem.concept[:25]}...")
        print(f"     Type: {suggestion.connection_type.value}")
        print(f"     Analysis: {suggestion.analysis_type.value}")
        print(f"     Confidence: {suggestion.confidence_score:.3f}")
        print(f"     Weight: {suggestion.suggested_weight:.3f}")
        print(f"     Reasoning: {suggestion.reasoning}")
        print()
    
    # Apply high-confidence suggestions
    applied_counts = connection_agent.apply_connection_suggestions(
        connection_suggestions, min_confidence=0.7
    )
    print(f"Applied connections: {applied_counts}")
    print()
    
    # Step 4: User queries memory system
    print("Step 4: User queries the system")
    print("-" * 60)
    
    query = "How to handle errors in async Python code?"
    context = QueryContext(
        query=query,
        conversation_history=["What is asyncio?", "How do I manage async tasks?"],
        user_intent="learning",
        domain="programming",
        query_type=QueryType.SEARCH
    )
    
    print(f"User Query: {query}")
    print(f"Context: {context.domain} domain, intent: {context.user_intent}")
    print(f"Conversation history: {context.conversation_history}")
    print()
    
    # Step 5: RelevanceAgent evaluates all memories
    print("Step 5: RelevanceAgent - Evaluating memory relevance")
    print("-" * 60)
    
    all_memories = list(memory_graph.nodes.values())
    relevance_scores = []
    
    for memory in all_memories:
        score = relevance_agent.evaluate_relevance(memory, query, context)
        relevance_scores.append(score)
    
    # Sort by relevance
    memory_score_pairs = list(zip(all_memories, relevance_scores))
    memory_score_pairs.sort(key=lambda x: x[1].overall, reverse=True)
    
    print("Relevance evaluation results:")
    for memory, score in memory_score_pairs:
        must_keep_flag = " 🔒" if score.must_keep else ""
        print(f"  {memory.concept[:40]}{must_keep_flag}")
        print(f"    Overall: {score.overall:.3f}, Semantic: {score.semantic_score:.3f}, Context: {score.context_score:.3f}")
        print(f"    Reasoning: {score.reasoning}")
        print()
    
    # Step 6: FilterAgent makes final selection
    print("Step 6: FilterAgent - Making final memory selection")
    print("-" * 60)
    
    user_preferences = UserPreferences(
        max_memories=4,
        prefer_recent=True,
        avoid_redundancy=True,
        relevance_threshold=0.2,  # Lower threshold to see more results
        diversity_factor=0.8
    )
    
    response_context = ResponseContext(
        response_type="chat",
        user_context="learning",
        conversation_history=context.conversation_history,
        platform="web"
    )
    
    filter_result = filter_agent.filter_for_response(
        candidate_memories=[pair[0] for pair in memory_score_pairs],
        relevance_scores=[pair[1] for pair in memory_score_pairs],
        user_preferences=user_preferences,
        response_context=response_context
    )
    
    print(f"FilterAgent results:")
    print(f"  Selected: {len(filter_result.selected_memories)} memories")
    print(f"  Filtered out: {filter_result.filtered_count} memories")
    print(f"  Reasoning: {filter_result.reasoning}")
    print()
    
    # Step 7: Show final curated response
    print("Step 7: Final curated memory response")
    print("-" * 60)
    
    print(f"In response to: '{query}'")
    print(f"The AI memory system selected {len(filter_result.selected_memories)} relevant memories:")
    print()
    
    for i, (memory, score) in enumerate(zip(filter_result.selected_memories, filter_result.relevance_scores)):
        must_keep_indicator = " 🔒 MUST-KEEP" if score.must_keep else ""
        print(f"{i+1}. {memory.concept}{must_keep_indicator}")
        print(f"   Relevance: {score.overall:.3f} | Access count: {memory.access_count}")
        print(f"   Tags: {memory.tags}")
        print(f"   Summary: {memory.summary}")
        
        # Show connections to other selected memories
        connected_to = []
        for other_memory in filter_result.selected_memories:
            if other_memory.id != memory.id and other_memory.id in memory.connections:
                connection = memory.connections[other_memory.id]
                connected_to.append(f"{other_memory.concept[:20]}... (weight: {connection.weight:.2f})")
        
        if connected_to:
            print(f"   Connected to: {', '.join(connected_to)}")
        print()
    
    # Step 8: ConnectionAgent suggests strengthening based on this query
    print("Step 8: ConnectionAgent - Suggesting connection strengthening")
    print("-" * 60)
    
    for memory in filter_result.selected_memories:
        strengthening_suggestions = connection_agent.suggest_connection_strengthening(
            memory.id, context={'query': query}
        )
        
        if strengthening_suggestions:
            print(f"Strengthening suggestions for '{memory.concept[:30]}...':")
            for suggestion in strengthening_suggestions[:2]:  # Top 2 suggestions
                target_memory = memory_graph.get_node(suggestion.target_id)
                change_direction = "↑" if suggestion.weight_change > 0 else "↓"
                print(f"  {change_direction} {target_memory.concept[:25]}...")
                print(f"    Current: {suggestion.current_weight:.3f} → Suggested: {suggestion.suggested_weight:.3f}")
                print(f"    Change: {suggestion.weight_change:+.3f} (confidence: {suggestion.confidence:.3f})")
                print(f"    Reasoning: {suggestion.reasoning}")
            print()
    
    # Step 9: System statistics
    print("Step 9: System Statistics")
    print("-" * 60)
    
    connection_stats = connection_agent.get_connection_statistics()
    
    print("Memory System Statistics:")
    print(f"  Total memories: {len(memory_graph.nodes)}")
    print(f"  Total connections: {connection_stats['total_connections']}")
    print(f"  Average connections per memory: {connection_stats['average_connections_per_node']:.2f}")
    print(f"  Access events recorded: {connection_stats['total_access_events']}")
    print(f"  Co-access patterns discovered: {connection_stats['unique_co_access_pairs']}")
    print(f"  Active sessions tracked: {connection_stats['active_sessions']}")
    print()
    
    must_keep_count = sum(1 for score in filter_result.relevance_scores if score.must_keep)
    print("Agent Performance:")
    print(f"  RelevanceAgent: Evaluated {len(all_memories)} memories")
    print(f"  FilterAgent: Selected {len(filter_result.selected_memories)} from {len(all_memories)} candidates")
    print(f"  ConnectionAgent: Generated {len(connection_suggestions)} connection suggestions")
    print(f"  Must-keep memories protected: {must_keep_count}")
    
    return {
        'memories': all_memories,
        'connections': connection_suggestions,
        'final_selection': filter_result.selected_memories,
        'stats': connection_stats
    }


if __name__ == "__main__":
    result = demonstrate_complete_agent_system()