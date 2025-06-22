#!/usr/bin/env python3
"""
Populate demo data for AI Memory System with realistic memory clusters

This script creates comprehensive demo data with 3D positioning for visualization.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.memory_graph import MemoryGraph
from src.core.memory_node import MemoryNode, ConnectionType
from src.visualization.spatial_layout import SpatialLayoutEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_demo_data():
    """Create comprehensive demo data with realistic memory clusters"""
    
    # Initialize components
    memory_graph = MemoryGraph()
    layout_engine = SpatialLayoutEngine(space_size=80.0, clustering_strength=0.8)
    
    # Demo memory data organized by clusters
    demo_memories = [
        # AI Research Cluster
        {
            "concept": "Transformer Architecture",
            "summary": "Deep dive into attention mechanisms and transformer models",
            "content": "Transformers use self-attention to process sequences in parallel. Key innovations include multi-head attention, positional encoding, and layer normalization. The architecture has revolutionized NLP and is now being applied to computer vision and other domains.",
            "tags": ["AI", "research", "transformers"],
            "keywords": ["attention", "transformer", "NLP", "self-attention", "multi-head"]
        },
        {
            "concept": "Large Language Models",
            "summary": "Understanding the scaling laws and emergent capabilities of LLMs",
            "content": "Large language models like GPT and Claude demonstrate emergent capabilities as they scale. Key factors include model size, training data quality, and compute resources. These models show surprising abilities in reasoning, creativity, and knowledge synthesis.",
            "tags": ["AI", "research", "LLM"],
            "keywords": ["language", "model", "scaling", "emergent", "GPT", "Claude"]
        },
        {
            "concept": "Memory Systems in AI",
            "summary": "Exploring different approaches to AI memory and knowledge representation",
            "content": "AI memory systems range from simple key-value stores to complex graph databases. Modern approaches include vector databases for semantic search, memory networks for reasoning, and hybrid systems combining multiple approaches.",
            "tags": ["AI", "research", "memory"],
            "keywords": ["memory", "knowledge", "vector", "database", "semantic", "hybrid"]
        },
        
        # Programming & Development Cluster
        {
            "concept": "React TypeScript Best Practices",
            "summary": "Modern React development patterns with TypeScript",
            "content": "Best practices include strict typing, component composition, custom hooks for logic reuse, and proper state management. Use Zustand for simple state, Redux Toolkit for complex apps, and React Query for server state.",
            "tags": ["programming", "frontend", "react"],
            "keywords": ["React", "TypeScript", "hooks", "state", "Zustand", "patterns"]
        },
        {
            "concept": "FastAPI Backend Architecture",
            "summary": "Building scalable Python APIs with FastAPI",
            "content": "FastAPI provides automatic API documentation, dependency injection, and async support. Key patterns include separating business logic from routes, using Pydantic for data validation, and implementing proper error handling.",
            "tags": ["programming", "backend", "python"],
            "keywords": ["FastAPI", "Python", "API", "async", "Pydantic", "validation"]
        },
        {
            "concept": "3D Visualization with Three.js",
            "summary": "Creating interactive 3D experiences for web applications",
            "content": "Three.js enables complex 3D graphics in browsers. Key concepts include scenes, cameras, renderers, materials, and geometries. For React integration, use @react-three/fiber for declarative 3D components.",
            "tags": ["programming", "frontend", "3D"],
            "keywords": ["Three.js", "3D", "WebGL", "React", "fiber", "visualization"]
        },
        
        # Machine Learning & Data Science Cluster
        {
            "concept": "Vector Embeddings",
            "summary": "Understanding semantic representations in high-dimensional space",
            "content": "Vector embeddings map discrete objects to continuous vector spaces where semantic similarity corresponds to geometric proximity. Applications include word embeddings, sentence transformers, and multimodal encoders.",
            "tags": ["ML", "embeddings", "vectors"],
            "keywords": ["embedding", "vector", "semantic", "similarity", "transformer", "space"]
        },
        {
            "concept": "Dimensionality Reduction",
            "summary": "Techniques for visualizing high-dimensional data",
            "content": "Methods like PCA, t-SNE, and UMAP reduce dimensionality while preserving important structure. PCA maximizes variance, t-SNE preserves local neighborhoods, and UMAP balances local and global structure.",
            "tags": ["ML", "visualization", "analysis"],
            "keywords": ["PCA", "t-SNE", "UMAP", "dimensionality", "reduction", "visualization"]
        },
        {
            "concept": "Graph Neural Networks",
            "summary": "Machine learning on graph-structured data",
            "content": "GNNs operate on graph structures, aggregating information from node neighborhoods. Variants include Graph Convolutional Networks (GCN), GraphSAGE, and Graph Attention Networks (GAT). Applications span social networks, molecules, and knowledge graphs.",
            "tags": ["ML", "graphs", "neural_networks"],
            "keywords": ["GNN", "graph", "neural", "network", "GCN", "GraphSAGE", "GAT"]
        },
        
        # Personal & Productivity Cluster
        {
            "concept": "Quarterly Planning",
            "summary": "Strategic goal setting and progress tracking",
            "content": "Effective quarterly planning involves setting 3-5 key objectives with measurable results. Use OKRs (Objectives and Key Results) framework. Review progress weekly and adjust tactics while maintaining strategic focus.",
            "tags": ["productivity", "planning", "goals"],
            "keywords": ["quarterly", "planning", "OKRs", "objectives", "goals", "strategy"]
        },
        {
            "concept": "Deep Work Principles",
            "summary": "Strategies for focused, high-value cognitive work",
            "content": "Deep work requires eliminating distractions, creating rituals, and building concentration stamina. Key strategies include time blocking, notification management, and creating dedicated workspace environments.",
            "tags": ["productivity", "focus", "work"],
            "keywords": ["deep", "work", "focus", "concentration", "distraction", "productivity"]
        },
        {
            "concept": "Knowledge Management Systems",
            "summary": "Personal systems for capturing and connecting insights",
            "content": "Effective knowledge management combines capture, organization, and retrieval. Popular methods include Zettelkasten for networked notes, PARA for project organization, and spaced repetition for memory consolidation.",
            "tags": ["productivity", "knowledge", "systems"],
            "keywords": ["knowledge", "management", "Zettelkasten", "PARA", "notes", "system"]
        },
        
        # Psychology & Learning Cluster
        {
            "concept": "Cognitive Load Theory",
            "summary": "Understanding mental processing limitations in learning",
            "content": "Cognitive load theory describes how working memory limitations affect learning. Three types: intrinsic (task complexity), extraneous (poor instruction), and germane (schema building). Design reduces extraneous load.",
            "tags": ["psychology", "learning", "cognition"],
            "keywords": ["cognitive", "load", "working", "memory", "learning", "schema"]
        },
        {
            "concept": "Deliberate Practice",
            "summary": "Systematic approach to skill development and expertise",
            "content": "Deliberate practice involves focused effort on improving specific weaknesses, immediate feedback, and progressively increasing difficulty. Key elements include expert guidance, clear goals, and consistent effort over time.",
            "tags": ["psychology", "learning", "practice"],
            "keywords": ["deliberate", "practice", "expertise", "skill", "feedback", "improvement"]
        }
    ]
    
    # Create memory nodes
    nodes = []
    for i, memory_data in enumerate(demo_memories):
        node = MemoryNode(
            concept=memory_data["concept"],
            summary=memory_data["summary"],
            full_content=memory_data["content"],
            tags=memory_data["tags"],
            keywords=memory_data["keywords"]
        )
        
        # Add some random access data
        import random
        node.access_count = random.randint(1, 25)
        node.importance_score = random.uniform(0.3, 0.9)
        
        nodes.append(node)
        memory_graph.add_node(node)
    
    # Generate 3D layout
    logger.info(f"Generating 3D layout for {len(nodes)} nodes...")
    positions = layout_engine.generate_initial_layout(nodes)
    
    # Update node positions
    for node_id, position in positions.items():
        node = memory_graph.get_node(node_id)
        if node:
            node.position_3d = position
    
    # Create meaningful connections based on content relationships
    connections_to_create = [
        # AI Research connections
        (0, 1, ConnectionType.SIMILARITY, 0.85),  # Transformer <-> LLM
        (0, 2, ConnectionType.CONTEXT, 0.75),    # Transformer <-> Memory Systems
        (1, 2, ConnectionType.SIMILARITY, 0.70), # LLM <-> Memory Systems
        
        # Programming connections
        (3, 4, ConnectionType.CONTEXT, 0.80),    # React <-> FastAPI
        (3, 5, ConnectionType.SIMILARITY, 0.75), # React <-> Three.js
        
        # ML connections
        (6, 7, ConnectionType.SIMILARITY, 0.90), # Embeddings <-> Dimensionality Reduction
        (6, 8, ConnectionType.CONTEXT, 0.70),    # Embeddings <-> GNN
        (7, 8, ConnectionType.SIMILARITY, 0.65), # Dimensionality <-> GNN
        
        # Cross-cluster connections
        (2, 6, ConnectionType.SIMILARITY, 0.80), # Memory Systems <-> Embeddings
        (5, 7, ConnectionType.CONTEXT, 0.60),    # Three.js <-> Dimensionality Reduction
        (1, 11, ConnectionType.CONTEXT, 0.55),   # LLM <-> Knowledge Management
        (11, 12, ConnectionType.SIMILARITY, 0.75), # Knowledge Management <-> Cognitive Load
        (12, 13, ConnectionType.SIMILARITY, 0.80), # Cognitive Load <-> Deliberate Practice
        
        # Productivity cluster connections
        (9, 10, ConnectionType.CONTEXT, 0.70),   # Quarterly Planning <-> Deep Work
        (10, 11, ConnectionType.SIMILARITY, 0.65), # Deep Work <-> Knowledge Management
    ]
    
    logger.info(f"Creating {len(connections_to_create)} connections...")
    for source_idx, target_idx, conn_type, weight in connections_to_create:
        if source_idx < len(nodes) and target_idx < len(nodes):
            source_node = nodes[source_idx]
            target_node = nodes[target_idx]
            
            # Create bidirectional connections
            memory_graph.create_connection(source_node.id, target_node.id, conn_type, weight)
            memory_graph.create_connection(target_node.id, source_node.id, conn_type, weight * 0.9)
    
    logger.info("Demo data creation completed!")
    logger.info(f"Created {len(nodes)} memory nodes")
    logger.info(f"Generated 3D positions in space size {layout_engine.space_size}")
    logger.info(f"Cluster information: {layout_engine.get_cluster_info()}")
    
    return memory_graph, layout_engine

async def main():
    """Main function to populate demo data"""
    logger.info("Starting demo data population...")
    
    try:
        memory_graph, layout_engine = await create_demo_data()
        
        # Print summary
        print("\n" + "="*60)
        print("DEMO DATA SUMMARY")
        print("="*60)
        print(f"Total Memory Nodes: {len(memory_graph.nodes)}")
        
        # Group by tags
        tag_groups = {}
        for node in memory_graph.nodes.values():
            primary_tag = node.tags[0] if node.tags else "untagged"
            if primary_tag not in tag_groups:
                tag_groups[primary_tag] = []
            tag_groups[primary_tag].append(node.concept)
        
        for tag, concepts in tag_groups.items():
            print(f"\n{tag.upper()} Cluster ({len(concepts)} nodes):")
            for concept in concepts:
                print(f"  • {concept}")
        
        # Print sample node with 3D position
        sample_node = next(iter(memory_graph.nodes.values()))
        print(f"\nSample Node Position:")
        print(f"  {sample_node.concept}")
        print(f"  3D Position: {sample_node.position_3d}")
        print(f"  Tags: {sample_node.tags}")
        print(f"  Connections: {len(sample_node.connections)}")
        
        print("\n" + "="*60)
        print("Demo data is ready for the AI Memory System!")
        print("Start the backend server to use this data.")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error creating demo data: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())