#!/usr/bin/env python3
"""
Complete Agent System Example - Demonstrates Co-Access Connection Learning

This example showcases the full AI memory system with:
1. Memory creation and storage
2. Intelligent memory retrieval using RelevanceAgent and FilterAgent
3. Co-access connection creation through usage patterns
4. Connection strengthening based on mutual helpfulness
5. Response quality evaluation and feedback loops

The system demonstrates how non-semantic connections are built through actual
usage patterns, creating a more human-like associative memory structure.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import List, Dict

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.memory_graph import MemoryGraph
from src.core.memory_node import MemoryNode, ConnectionType
from src.agents.relevance_agent import RelevanceAgent, QueryContext, QueryType
from src.agents.filter_agent import FilterAgent, UserPreferences, ResponseContext
from src.agents.connection_agent import ConnectionAgent
from src.agents.mock_claude_client import MockClaudeClient
from src.retrieval.embedding_search import EmbeddingSearch, EmbeddingConfig
import uuid


class CompleteAgentSystemDemo:
    """Demonstrates the complete memory system with co-access learning"""
    
    def __init__(self):
        # Initialize core components
        self.memory_graph = MemoryGraph()
        
        # Use simplified search for demo (mock implementation)
        self.embedding_search = None  # Not needed for this demo
        self.connection_agent = ConnectionAgent(self.memory_graph)
        self.claude_client = MockClaudeClient()
        
        # Initialize agents
        self.relevance_agent = RelevanceAgent(memory_graph=self.memory_graph, claude_client=self.claude_client)
        self.filter_agent = FilterAgent()
        
        # Demo state
        self.conversation_history = []
        self.session_stats = {
            'total_queries': 0,
            'memories_created': 0,
            'connections_created': 0,
            'connections_strengthened': 0
        }
    
    async def setup_demo_memories(self):
        """Create a set of demo memories for testing co-access patterns"""
        print("🧠 Setting up demo memory collection...")
        
        demo_memories = [
            {
                "concept": "Python async programming fundamentals",
                "summary": "Core concepts of asynchronous programming in Python using async/await",
                "content": "Python async programming allows handling multiple operations concurrently using async/await syntax. Key concepts include event loops, coroutines, and tasks.",
                "tags": ["python", "async", "programming", "concurrency"],
                "keywords": ["async", "await", "python", "asyncio", "coroutines", "event loop"]
            },
            {
                "concept": "FastAPI async request handling",
                "summary": "How FastAPI handles asynchronous requests and responses efficiently",
                "content": "FastAPI provides excellent support for async request handling, allowing high throughput web applications with async/await syntax for route handlers.",
                "tags": ["fastapi", "async", "web", "api", "performance"],
                "keywords": ["fastapi", "async", "requests", "api", "web", "performance"]
            },
            {
                "concept": "Database connection pooling",
                "summary": "Efficient database connection management using connection pools",
                "content": "Connection pooling optimizes database performance by reusing connections, reducing overhead and improving scalability for web applications.",
                "tags": ["database", "performance", "optimization", "scaling"],
                "keywords": ["database", "connection", "pool", "performance", "optimization"]
            },
            {
                "concept": "Memory management in Python",
                "summary": "Understanding Python's memory allocation and garbage collection",
                "content": "Python uses automatic memory management with reference counting and cycle detection for garbage collection, helping prevent memory leaks.",
                "tags": ["python", "memory", "optimization", "performance"],
                "keywords": ["python", "memory", "garbage", "collection", "optimization"]
            },
            {
                "concept": "API rate limiting strategies",
                "summary": "Implementing rate limiting to protect API endpoints from abuse",
                "content": "Rate limiting protects APIs by controlling request frequency, using techniques like token bucket, sliding window, and fixed window algorithms.",
                "tags": ["api", "security", "performance", "protection"],
                "keywords": ["rate", "limiting", "api", "security", "protection", "throttling"]
            },
            {
                "concept": "Microservices communication patterns",
                "summary": "Best practices for service-to-service communication in distributed systems",
                "content": "Microservices communicate through HTTP APIs, message queues, and event streams, requiring careful design for reliability and performance.",
                "tags": ["microservices", "architecture", "communication", "distributed"],
                "keywords": ["microservices", "communication", "distributed", "architecture", "apis"]
            }
        ]
        
        for memory_data in demo_memories:
            memory = MemoryNode(
                concept=memory_data["concept"],
                summary=memory_data["summary"],
                full_content=memory_data["content"],
                tags=memory_data["tags"],
                keywords=memory_data["keywords"]
            )
            
            memory_id = self.memory_graph.add_node(memory)
            print(f"  📝 Created memory: {memory.concept}")
            self.session_stats['memories_created'] += 1
        
        print(f"✅ Created {len(demo_memories)} demo memories\n")
    
    async def simulate_intelligent_conversation(self):
        """Simulate a conversation that demonstrates co-access learning"""
        print("🗣️  Starting intelligent conversation simulation...\n")
        
        # Conversation queries that will create co-access patterns
        queries = [
            "How do I optimize Python async performance?",
            "What's the best way to handle concurrent API requests?",
            "How can I prevent memory leaks in Python web applications?",
            "What are effective strategies for scaling web APIs?",
            "How do I implement efficient async database operations?",
            "What patterns help with microservice performance optimization?"
        ]
        
        for i, query in enumerate(queries):
            print(f"🔍 Query {i+1}: {query}")
            await self._process_query_with_co_access_learning(query, f"session_{i+1}")
            print()
    
    async def _process_query_with_co_access_learning(self, query: str, session_id: str):
        """Process a single query with full co-access learning pipeline"""
        self.session_stats['total_queries'] += 1
        
        # Step 1: Get all memories as candidates (simplified for demo)
        print("  🔎 Searching for relevant memories...")
        candidate_memories = list(self.memory_graph.nodes.values())
        
        if not candidate_memories:
            print("  ❌ No memories found")
            return
        
        print(f"  📋 Found {len(candidate_memories)} candidate memories")
        
        # Step 2: Use RelevanceAgent for intelligent scoring
        # Extract just the query strings from conversation history
        conversation_strings = [item["query"] for item in self.conversation_history[-3:]]
        
        query_context = QueryContext(
            query=query,
            conversation_history=conversation_strings,
            user_intent="information_seeking",
            domain="programming",
            query_type=QueryType.SEARCH
        )
        
        # Use first 3 memory IDs as reference for connection scoring
        reference_memory_ids = [list(self.memory_graph.nodes.keys())[i] for i in range(min(3, len(candidate_memories)))]
        
        memory_relevance_scores = []
        for memory in candidate_memories:
            relevance_score = self.relevance_agent.evaluate_relevance(
                memory=memory,
                query=query,
                context=query_context,
                reference_memory_ids=reference_memory_ids
            )
            memory_relevance_scores.append((memory, relevance_score))
        
        print(f"  🎯 Evaluated relevance for {len(memory_relevance_scores)} memories")
        
        # Step 3: Use FilterAgent for final selection
        user_preferences = UserPreferences(
            max_memories=4,
            prefer_recent=True,
            avoid_redundancy=True,
            relevance_threshold=0.3,
            diversity_factor=0.7
        )
        
        response_context = ResponseContext(
            response_type="information",
            user_context="learning",
            conversation_history=conversation_strings,
            platform="demo"
        )
        
        memories_for_filtering = [item[0] for item in memory_relevance_scores]
        relevance_scores = [item[1] for item in memory_relevance_scores]
        
        filter_result = self.filter_agent.filter_for_response(
            candidate_memories=memories_for_filtering,
            relevance_scores=relevance_scores,
            user_preferences=user_preferences,
            response_context=response_context
        )
        
        selected_memories = filter_result.selected_memories
        final_scores = filter_result.relevance_scores
        
        print(f"  ✅ Selected {len(selected_memories)} final memories:")
        for memory, score in zip(selected_memories, final_scores):
            print(f"    • {memory.concept} (score: {score.overall:.3f})")
        
        # Step 4: Simulate response generation and quality evaluation
        response = self._generate_demo_response(query, selected_memories)
        response_quality = self._evaluate_demo_response_quality(query, selected_memories, response)
        
        print(f"  📝 Generated response (quality: {response_quality:.3f})")
        
        # Step 5: Record co-access with feedback for connection learning
        if len(selected_memories) >= 2:
            memory_ids = [memory.id for memory in selected_memories]
            relevance_scores_list = [score.overall for score in final_scores]
            
            print(f"  🔗 Recording co-access for {len(memory_ids)} memories...")
            
            # Track connections before
            initial_connections = sum(len(node.connections) for node in self.memory_graph.nodes.values())
            
            self.connection_agent.record_co_access_with_feedback(
                memory_ids=memory_ids,
                query=query,
                relevance_scores=relevance_scores_list,
                response_quality=response_quality,
                session_id=session_id
            )
            
            # Track connections after
            final_connections = sum(len(node.connections) for node in self.memory_graph.nodes.values())
            new_connections = final_connections - initial_connections
            
            if new_connections > 0:
                self.session_stats['connections_created'] += new_connections
                print(f"  ✨ Created {new_connections} new co-access connections")
            else:
                self.session_stats['connections_strengthened'] += 1
                print(f"  💪 Strengthened existing connections")
            
            # Record connection modifications for learning
            for _ in range(len(selected_memories) - 1):
                self.relevance_agent.record_connection_modification()
        
        # Update conversation history
        self.conversation_history.append({"query": query, "response": response})
        
        # Update memory access counts
        for memory in selected_memories:
            memory.update_access()
    
    def _generate_demo_response(self, query: str, memories: List[MemoryNode]) -> str:
        """Generate a demo response using selected memories"""
        if not memories:
            return f"I don't have specific information about '{query}' yet."
        
        memory_insights = [f"• {memory.concept}: {memory.summary}" for memory in memories[:3]]
        
        response = f"""Based on my analysis of {len(memories)} relevant memories:

{chr(10).join(memory_insights)}

These memories provide complementary insights that help address your question about {query.lower()}."""
        
        return response
    
    def _evaluate_demo_response_quality(self, query: str, memories: List[MemoryNode], response: str) -> float:
        """Evaluate response quality for demonstration"""
        base_quality = 0.6
        
        # Boost for using multiple complementary memories
        if len(memories) >= 2:
            base_quality += 0.2
        
        # Boost for semantic relevance (simplified heuristic)
        query_words = set(query.lower().split())
        memory_words = set()
        for memory in memories:
            memory_words.update(memory.concept.lower().split())
            memory_words.update(memory.keywords)
        
        overlap = len(query_words & memory_words) / max(len(query_words), 1)
        base_quality += overlap * 0.2
        
        return min(1.0, base_quality)
    
    def analyze_connection_patterns(self):
        """Analyze the connection patterns that emerged from co-access learning"""
        print("📊 Analyzing emergent connection patterns...\n")
        
        # Get all connections by type
        connection_types = {}
        total_connections = 0
        
        for node in self.memory_graph.nodes.values():
            for target_id, connection in node.connections.items():
                conn_type = connection.connection_type
                if conn_type not in connection_types:
                    connection_types[conn_type] = []
                
                target_node = self.memory_graph.get_node(target_id)
                if target_node:
                    connection_types[conn_type].append({
                        'source': node.concept[:30] + "...",
                        'target': target_node.concept[:30] + "...",
                        'weight': connection.weight
                    })
                    total_connections += 1
        
        print(f"🔗 Total connections created: {total_connections}")
        
        for conn_type, connections in connection_types.items():
            print(f"\n{conn_type.value.upper()} connections ({len(connections)}):")
            
            # Sort by weight (strongest first)
            connections.sort(key=lambda x: x['weight'], reverse=True)
            
            for conn in connections[:5]:  # Show top 5
                print(f"  • {conn['source']} → {conn['target']} (weight: {conn['weight']:.3f})")
            
            if len(connections) > 5:
                print(f"  ... and {len(connections) - 5} more connections")
    
    def display_session_statistics(self):
        """Display comprehensive session statistics"""
        print("📈 Session Statistics:")
        print(f"  • Total queries processed: {self.session_stats['total_queries']}")
        print(f"  • Memories created: {self.session_stats['memories_created']}")
        print(f"  • New connections created: {self.session_stats['connections_created']}")
        print(f"  • Connections strengthened: {self.session_stats['connections_strengthened']}")
        
        # Connection agent statistics
        agent_stats = self.connection_agent.get_connection_statistics()
        print(f"  • Total access events: {agent_stats['total_access_events']}")
        print(f"  • Unique co-access pairs: {agent_stats['unique_co_access_pairs']}")
        print(f"  • Average connections per memory: {agent_stats['average_connections_per_node']:.2f}")
        
        # Relevance agent learning progress
        print(f"  • Connection strength weight: {self.relevance_agent.connection_strength_weight:.3f}")
        print(f"  • Connection modifications recorded: {self.relevance_agent.connection_modification_count}")


async def main():
    """Run the complete agent system demonstration"""
    print("🚀 AI Memory System - Complete Agent Demo")
    print("=" * 50)
    print("This demo showcases intelligent memory retrieval and co-access learning\n")
    
    # Initialize the demo system
    demo = CompleteAgentSystemDemo()
    
    try:
        # Set up demo memories
        await demo.setup_demo_memories()
        
        # Run intelligent conversation simulation
        await demo.simulate_intelligent_conversation()
        
        # Analyze results
        print("🧪 Analysis Phase")
        print("=" * 30)
        demo.analyze_connection_patterns()
        print()
        demo.display_session_statistics()
        
        print("\n✨ Demo completed successfully!")
        print("\nKey achievements:")
        print("• Demonstrated intelligent memory selection using RelevanceAgent and FilterAgent")
        print("• Showed co-access connection creation based on usage patterns")
        print("• Illustrated connection strengthening through repeated helpful access")
        print("• Proved non-semantic connections emerge from actual usage patterns")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set environment variables for the demo
    os.environ.setdefault('VOYAGER_LITE_API_KEY', 'demo-key')
    os.environ.setdefault('CLAUDE_API_KEY', 'demo-claude-key')
    
    # Run the async demo
    asyncio.run(main())