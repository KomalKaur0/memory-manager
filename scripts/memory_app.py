#!/usr/bin/env python3
"""
Simple Memory System Application
A standalone script to interact with your AI memory system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# from datetime import datetime
from ..src.core.memory_node import MemoryNode
from ..src.core.memory_graph import MemoryGraph
from ..src.storage.graph_database import MemoryGraphDatabase
from ..src.storage.vector_store import MemoryVectorStore
from ..src.retrieval.hybrid_retriever import HybridRetriever, RetrievalConfig, RetrievalResult
from ..src.agents.filter_agent import FilterAgent
from ..src.retrieval.embedding_search import EmbeddingSearch, EmbeddingConfig
from ..src.retrieval.graph_traversal import GraphTraversal, TraversalConfig

class SimpleMemoryApp:
    def __init__(self):
        print("🧠 Initializing AI Memory System...")
        try:
            #TODO: fill in these things
            u = ''
            user = ''
            pw = ''
            self.graph_db = MemoryGraphDatabase(uri=u, username=user, password=pw)
            w_url = ''
            w_api_key = ''
            em = ''
            self.vector_store = MemoryVectorStore(weaviate_url = w_url, 
                                                  weaviate_api_key = w_api_key, 
                                                  embedding_model = em)
            self.memory_graph = MemoryGraph(0.01)
            self.ec = EmbeddingConfig()
            self.retreival_config = RetrievalConfig()
            self.es = EmbeddingSearch(self.ec, self.vector_store)
            self.traversal_config = TraversalConfig()
            self.gt = GraphTraversal(self.memory_graph, self.traversal_config)
            self.retriever = HybridRetriever(self.es, self.gt, self.memory_graph, self.retreival_config)
            self.filter_agent = FilterAgent()
            
            # Connect to databases
            self.graph_db.connect()
            self.vector_store.connect()
            print("✅ Memory system ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            print("Make sure Neo4j and Weaviate are running!")
            sys.exit(1)
        
    def add_memory(self, concept, content, keywords=None, tags=None):
        """Add a new memory to the system"""
        try:
            memory = MemoryNode(
                concept=concept,
                keywords=keywords or [],
                tags=tags or [],
                summary=content[:200] + "..." if len(content) > 200 else content,
                full_content=content,
            )
            
            success = self.memory_graph.add_node(memory)
            if success:
                print(f"✅ Added memory: '{concept}'")
                return memory.id
            else:
                print(f"❌ Failed to add memory: '{concept}'")
                return None
                
        except Exception as e:
            print(f"❌ Error adding memory: {e}")
            return None
    
    def search_memories(self, query, max_results=5):
        """Search for memories related to a query"""
        try:
            print(f"🔍 Searching for: '{query}'")
            
            # Get results from retriever
            results = self.retriever.retrieve(query, max_results=max_results * 2)
            
            if not results:
                print("No memories found.")
                return []
            
            # Simple filtering (without AI agent for now)
            # Just take the top results
            filtered_results = results.memories
            
            # print(f"\n📚 Found {len(filtered_results)} relevant memories:")
            # for i, memory in enumerate(filtered_results, 1):
            #     print(f"\n{i}. {memory.concept}")
            #     if memory.tags:
            #         print(f"   Tags: {', '.join(memory.tags)}")
            #     print(f"   Summary: {memory.summary}")
                
            return filtered_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def show_connections(self, memory_id):
        """Show connections for a specific memory"""
        try:
            connections = self.memory_graph.get_connected_nodes(memory_id)
            if connections:
                print(f"\n🔗 Connections for memory {memory_id}:")
                for conn in connections:
                    print(f"  → {conn[2]}: {conn[0]} (weight: {conn[1]:.3f})")
            else:
                print("No connections found.")
        except Exception as e:
            print(f"❌ Error getting connections: {e}")
    
    def get_stats(self):
        """Show system statistics"""
        try:
            stats = self.memory_graph.get_graph_statistics()
            print(f"\n📊 Memory System Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def load_sample_data(self):
        """Load some sample memories for testing"""
        sample_memories = [
            {
                "concept": "Python Programming",
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "keywords": ["python", "programming", "language"],
                "tags": ["programming", "beginner-friendly"]
            },
            {
                "concept": "Machine Learning",
                "content": "Machine learning enables computers to learn from data without explicit programming.",
                "keywords": ["machine learning", "AI", "data"],
                "tags": ["AI", "data-science"]
            },
            {
                "concept": "Graph Databases",
                "content": "Graph databases store data as nodes and edges, perfect for connected data.",
                "keywords": ["graph", "database", "nodes", "relationships"],
                "tags": ["database", "graph-theory"]
            }
        ]
        
        print("📚 Loading sample data...")
        for memory_data in sample_memories:
            self.add_memory(**memory_data)
        print("✅ Sample data loaded!")
    
    def run(self):
        """Run the interactive interface"""
        print("\n🎯 AI Memory System - Interactive Mode")
        print("Commands:")
        print("  add     - Add a new memory")
        print("  search  - Search memories")
        print("  conn    - Show connections for a memory")
        print("  stats   - Show system statistics")
        print("  sample  - Load sample data")
        print("  help    - Show this help")
        print("  quit    - Exit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command in ['quit', 'q', 'exit']:
                    print("👋 Goodbye!")
                    break
                    
                elif command == 'add':
                    concept = input("Concept: ").strip()
                    if not concept:
                        print("Concept cannot be empty!")
                        continue
                        
                    content = input("Content: ").strip()
                    if not content:
                        print("Content cannot be empty!")
                        continue
                    
                    keywords_input = input("Keywords (comma-separated, optional): ").strip()
                    keywords = [k.strip() for k in keywords_input.split(',') if k.strip()] if keywords_input else []
                    
                    tags_input = input("Tags (comma-separated, optional): ").strip()
                    tags = [t.strip() for t in tags_input.split(',') if t.strip()] if tags_input else []
                    
                    self.add_memory(concept, content, keywords, tags)
                    
                elif command == 'search':
                    query = input("Search query: ").strip()
                    if query:
                        self.search_memories(query)
                    else:
                        print("Search query cannot be empty!")
                        
                elif command == 'conn':
                    memory_id = input("Memory ID: ").strip()
                    if memory_id:
                        self.show_connections(memory_id)
                    else:
                        print("Memory ID cannot be empty!")
                        
                elif command == 'stats':
                    self.get_stats()
                    
                elif command == 'sample':
                    self.load_sample_data()
                    
                elif command == 'help':
                    print("\nCommands: add, search, conn, stats, sample, help, quit")
                    
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main entry point"""
    print("=" * 50)
    print("🧠 AI Memory System")
    print("=" * 50)
    
    try:
        app = SimpleMemoryApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()