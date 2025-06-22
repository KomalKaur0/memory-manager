#!/usr/bin/env python3
"""
Test script for model switching functionality
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.model_client import UnifiedModelClient, ModelProvider, get_model_client_from_env
from src.agents.relevance_agent import RelevanceAgent
from src.core.memory_node import MemoryNode

async def test_model_switching():
    """Test the model switching functionality."""
    print("Testing Model Switching Functionality")
    print("=" * 50)
    
    # Test 1: Initialize with different providers
    print("1. Testing UnifiedModelClient initialization...")
    
    # Test with mock provider (should always work)
    mock_client = UnifiedModelClient(ModelProvider.MOCK)
    print(f"   Mock client initialized: {mock_client.get_current_provider().value}")
    
    # Test response generation
    response = mock_client.generate_response("Test prompt", max_tokens=50)
    print(f"   Mock response: {response.content[:50]}...")
    
    # Test 2: Check available providers
    print("\n2. Checking available providers...")
    available = mock_client.get_available_providers()
    print(f"   Available providers: {[p.value for p in available]}")
    
    # Test 3: Test provider switching
    print("\n3. Testing provider switching...")
    
    # Try switching to each provider
    for provider in [ModelProvider.CLAUDE, ModelProvider.GEMINI, ModelProvider.MOCK]:
        print(f"   Attempting to switch to {provider.value}...")
        success = mock_client.switch_provider(provider)
        current = mock_client.get_current_provider().value
        status = "✓" if success else "✗"
        print(f"   {status} Switch result: {success}, Current: {current}")
        
        # Test response with current provider
        if success or provider == ModelProvider.MOCK:
            response = mock_client.generate_response("What is 2+2?", max_tokens=20)
            if response.error:
                print(f"     Error: {response.error}")
            else:
                print(f"     Response: {response.content[:30]}...")
    
    # Test 4: Test with environment configuration
    print("\n4. Testing environment-based configuration...")
    
    # Test with different environment settings
    test_cases = [
        ("claude", "AI_MODEL_PROVIDER=claude"),
        ("gemini", "AI_MODEL_PROVIDER=gemini"), 
        ("mock", "AI_MODEL_PROVIDER=mock"),
        ("invalid", "AI_MODEL_PROVIDER=invalid")
    ]
    
    for provider, env_setting in test_cases:
        print(f"   Testing {env_setting}...")
        os.environ['AI_MODEL_PROVIDER'] = provider
        
        try:
            env_client = get_model_client_from_env()
            current = env_client.get_current_provider().value
            print(f"     ✓ Created client with provider: {current}")
        except Exception as e:
            print(f"     ✗ Error: {e}")
    
    # Test 5: Test with RelevanceAgent integration
    print("\n5. Testing RelevanceAgent integration...")
    
    # Create a simple memory node for testing
    test_memory = MemoryNode(
        id="test_memory_1",
        concept="Python programming",
        summary="Basic concepts of Python programming language",
        content="Python is a high-level programming language.",
        full_content="Python is a high-level programming language with clear syntax.",
        tags=["programming", "python", "tutorial"]
    )
    
    # Test with mock client
    os.environ['AI_MODEL_PROVIDER'] = 'mock'
    client = get_model_client_from_env()
    agent = RelevanceAgent(model_client=client)
    
    print(f"   RelevanceAgent created with {client.get_current_provider().value} provider")
    
    # Create a simple query context
    from src.agents.relevance_agent import QueryContext, QueryType
    context = QueryContext(
        query="How do I learn Python?",
        conversation_history=["I want to learn programming"],
        query_type=QueryType.SEARCH
    )
    
    # Test relevance evaluation
    try:
        score = agent.evaluate_relevance(test_memory, "How do I learn Python?", context)
        print(f"   ✓ Relevance evaluation successful: {score.overall:.2f}")
        print(f"     Reasoning: {score.reasoning[:80]}...")
    except Exception as e:
        print(f"   ✗ Relevance evaluation error: {e}")
    
    print("\n" + "=" * 50)
    print("Model switching test completed!")

def test_api_keys():
    """Test API key detection and provider availability."""
    print("\nAPI Key Status:")
    print("-" * 30)
    
    claude_key = os.getenv('CLAUDE_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    print(f"CLAUDE_API_KEY: {'✓ Set' if claude_key and claude_key != 'your_claude_api_key_here' else '✗ Not set'}")
    print(f"GEMINI_API_KEY: {'✓ Set' if gemini_key and gemini_key != 'your_gemini_api_key_here' else '✗ Not set'}")
    
    # Test provider availability
    client = UnifiedModelClient(ModelProvider.MOCK)
    
    for provider in ModelProvider:
        available = client.is_provider_available(provider)
        status = "✓ Available" if available else "✗ Not available"
        print(f"{provider.value.upper()}: {status}")

if __name__ == "__main__":
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        test_api_keys()
        asyncio.run(test_model_switching())
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()