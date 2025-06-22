#!/usr/bin/env python3
"""
Test script to verify Claude fallback logging works correctly.
"""

import logging
import io
from unittest.mock import Mock, patch

from src.agents.relevance_agent import RelevanceAgent, QueryContext
from src.core.memory_node import MemoryNode

def test_claude_fallback_logging():
    """Test that Claude fallback usage is properly logged."""
    
    # Set up logging to capture messages
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    
    # Get the logger for relevance_agent and add our handler
    logger = logging.getLogger('src.agents.relevance_agent')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # Create a memory for testing
    memory = MemoryNode(
        concept="Python async programming",
        summary="How to use async/await in Python",
        full_content="Async programming allows you to write concurrent code..."
    )
    
    # Create a query context
    context = QueryContext(
        query="How do I use asyncio?",
        conversation_history=["Tell me about Python", "What about async programming?"]
    )
    
    # Test 1: Claude not available (no client)
    print("\n=== Test 1: Claude not available ===")
    agent_no_client = RelevanceAgent(claude_client=None)
    agent_no_client._calculate_associative_relevance(memory, "test query", context)
    
    # Test 2: Claude disabled
    print("\n=== Test 2: Claude disabled ===")
    agent_disabled = RelevanceAgent(config={'enable_claude_associative': False})
    agent_disabled._calculate_associative_relevance(memory, "test query", context)
    
    # Test 3: Claude API error
    print("\n=== Test 3: Claude API error ===")
    mock_client = Mock()
    mock_client.messages.create.side_effect = Exception("API timeout")
    
    agent_error = RelevanceAgent(claude_client=mock_client)
    agent_error._calculate_associative_relevance(memory, "test query", context)
    
    # Get the captured log messages
    log_output = log_capture.getvalue()
    print(f"\n=== Captured Log Messages ===")
    print(log_output)
    
    # Verify the log messages
    assert "Claude fallback used for memory 'Python async programming': Claude not available" in log_output
    assert "Claude API error for associative scoring of memory 'Python async programming': Anthropic API error: API timeout" in log_output
    
    print("\n✅ All Claude fallback logging tests passed!")
    
    # Clean up
    logger.removeHandler(handler)
    log_capture.close()

def test_conversation_history_length_with_logging():
    """Test that conversation history length works with logging."""
    
    # Set up logging
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    
    logger = logging.getLogger('src.agents.relevance_agent')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # Test with different conversation history lengths
    conversation_history = [
        "Message 1 about Python",
        "Message 2 about async programming", 
        "Message 3 about coroutines",
        "Message 4 about event loops",
        "Message 5 about tasks"
    ]
    
    context = QueryContext(
        query="How do I use asyncio?",
        conversation_history=conversation_history
    )
    
    memory = MemoryNode(
        concept="Async programming",
        summary="Introduction to async programming",
        full_content="Async programming is a programming paradigm..."
    )
    
    # Test with different configs
    configs = [1, 2, 3, 5]
    
    for length in configs:
        print(f"\n=== Testing conversation history length: {length} ===")
        agent = RelevanceAgent(
            config={'conversation_history_length': length},
            claude_client=None  # Force fallback
        )
        
        # This should trigger the fallback logging
        score = agent._calculate_associative_relevance(memory, "test query", context)
        print(f"Score: {score}")
    
    # Get the captured log messages
    log_output = log_capture.getvalue()
    print(f"\n=== Captured Log Messages ===")
    print(log_output)
    
    # Verify we got the expected number of fallback messages
    fallback_count = log_output.count("Claude fallback used for memory 'Async programming'")
    assert fallback_count == len(configs), f"Expected {len(configs)} fallback messages, got {fallback_count}"
    
    print(f"\n✅ Conversation history length test passed! Got {fallback_count} fallback messages.")
    
    # Clean up
    logger.removeHandler(handler)
    log_capture.close()

if __name__ == "__main__":
    print("Testing Claude fallback logging...")
    
    test_claude_fallback_logging()
    test_conversation_history_length_with_logging()
    
    print("\n🎉 All logging tests passed!")
    print("\nSummary of changes:")
    print("- Added configurable conversation_history_length parameter")
    print("- Added proper logging for Claude fallback usage")
    print("- Logs include memory concept and reason for fallback") 