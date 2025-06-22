"""
FastAPI main application for AI Memory System
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.memory_api import memory_router
from src.api.search_api import search_router
from src.api.admin_api import admin_router
from src.api.chat_api import chat_router
from src.api.analytics_api import analytics_router
from src.core.memory_graph import MemoryGraph
from src.retrieval.embedding_search_mock import EmbeddingSearch
from src.retrieval.hybrid_retriever_simple import HybridRetriever
from src.agents.relevance_agent import get_claude_client_from_env
from src.agents.connection_agent import ConnectionAgent
from src.retrieval.embedding_search import get_embedding_config_from_env
from src.api.websocket_manager import websocket_manager
from src.visualization.spatial_layout import SpatialLayoutEngine
from src.core.memory_node import MemoryNode, ConnectionType

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
memory_graph: MemoryGraph = None
embedding_search: EmbeddingSearch = None
hybrid_retriever: HybridRetriever = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Initialize and cleanup application resources"""
    global memory_graph, embedding_search, hybrid_retriever
    
    logger.info("Starting AI Memory System...")
    
    # Initialize core components
    memory_graph = MemoryGraph(decay_rate=0.01)
    
    # Initialize embedding search with API key from .env
    embedding_config = get_embedding_config_from_env()
    embedding_search = EmbeddingSearch(embedding_config)
    await embedding_search.initialize()
    
    hybrid_retriever = HybridRetriever(
        memory_graph=memory_graph,
        embedding_search=embedding_search
    )
    
    # Initialize connection agent for co-access tracking
    connection_agent = ConnectionAgent(memory_graph=memory_graph)
    
    # Initialize Claude client with API key from .env
    claude_client = get_claude_client_from_env()
    
    # Initialize spatial layout engine
    spatial_layout_engine = SpatialLayoutEngine()
    
    # Create demo data if memory graph is empty
    if len(memory_graph.nodes) == 0:
        logger.info("Creating demo memory data...")
        await create_demo_memory_data(memory_graph, spatial_layout_engine, hybrid_retriever)
    
    # Store instances in app state for access in routes
    app.state.memory_graph = memory_graph
    app.state.embedding_search = embedding_search
    app.state.hybrid_retriever = hybrid_retriever
    app.state.connection_agent = connection_agent
    app.state.claude_client = claude_client
    app.state.websocket_manager = websocket_manager
    app.state.spatial_layout_engine = spatial_layout_engine
    
    logger.info("AI Memory System initialized successfully")
    
    yield
    
    logger.info("Shutting down AI Memory System...")
    if embedding_search:
        await embedding_search.cleanup()

# Create FastAPI application
app = FastAPI(
    title="AI Memory System API",
    description="Adaptive graph-based memory system for AI with semantic similarity and learned associations",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",  # Expo dev server
        "http://localhost:19006", # Expo web dev server  
        "http://127.0.0.1:8081",  # Alternative localhost
        "http://127.0.0.1:19006", # Alternative localhost
        "*",  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(memory_router, prefix="/api/memory", tags=["memory"])
app.include_router(search_router, prefix="/api/search", tags=["search"])
app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(analytics_router, prefix="/api/analytics", tags=["analytics"])

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "AI Memory System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health checks
        graph_healthy = app.state.memory_graph is not None
        embedding_healthy = app.state.embedding_search is not None and app.state.embedding_search.is_ready()
        
        return {
            "status": "healthy" if graph_healthy and embedding_healthy else "degraded",
            "components": {
                "memory_graph": "healthy" if graph_healthy else "unhealthy",
                "embedding_search": "healthy" if embedding_healthy else "unhealthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

async def create_demo_memory_data(memory_graph, layout_engine, hybrid_retriever):
    """Create comprehensive demo data with realistic memory clusters"""
    
    demo_memories = [
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
            "concept": "React TypeScript Best Practices",
            "summary": "Modern React development patterns with TypeScript",
            "content": "Best practices include strict typing, component composition, custom hooks for logic reuse, and proper state management. Use Zustand for simple state, Redux Toolkit for complex apps, and React Query for server state.",
            "tags": ["programming", "frontend", "react"],
            "keywords": ["React", "TypeScript", "hooks", "state", "Zustand", "patterns"]
        },
        {
            "concept": "Vector Embeddings",
            "summary": "Understanding semantic representations in high-dimensional space",
            "content": "Vector embeddings map discrete objects to continuous vector spaces where semantic similarity corresponds to geometric proximity. Applications include word embeddings, sentence transformers, and multimodal encoders.",
            "tags": ["ML", "embeddings", "vectors"],
            "keywords": ["embedding", "vector", "semantic", "similarity", "transformer", "space"]
        },
        {
            "concept": "3D Visualization Techniques",
            "summary": "Creating interactive 3D experiences for web applications",
            "content": "Modern 3D visualization combines WebGL, Three.js, and React for interactive experiences. Key concepts include scene graphs, camera controls, material systems, and performance optimization techniques.",
            "tags": ["programming", "frontend", "3D"],
            "keywords": ["Three.js", "3D", "WebGL", "React", "visualization", "interactive"]
        }
    ]
    
    # Create memory nodes
    nodes = []
    for memory_data in demo_memories:
        node = MemoryNode(
            concept=memory_data["concept"],
            summary=memory_data["summary"],
            full_content=memory_data["content"],
            tags=memory_data["tags"],
            keywords=memory_data["keywords"]
        )
        
        # Add some realistic metadata
        import random
        node.access_count = random.randint(1, 15)
        node.importance_score = random.uniform(0.4, 0.9)
        
        nodes.append(node)
        memory_graph.add_node(node)
        
        # Generate embedding for the content
        try:
            embedding = await hybrid_retriever.embedding_search.get_embedding(
                f"{node.concept} {node.summary} {node.full_content}"
            )
            hybrid_retriever.embedding_search.store_embedding(node.id, embedding)
        except Exception as e:
            logger.warning(f"Failed to generate embedding for {node.concept}: {e}")
    
    # Generate 3D layout
    logger.info(f"Generating 3D layout for {len(nodes)} nodes...")
    positions = layout_engine.generate_initial_layout(nodes)
    
    # Update node positions
    for node_id, position in positions.items():
        node = memory_graph.get_node(node_id)
        if node:
            node.position_3d = position
    
    # Create comprehensive connections between related memories
    connections_to_create = [
        # AI Research cluster connections (strong relationships)
        (0, 1, ConnectionType.SIMILARITY, 0.85),  # Transformer <-> LLM
        (1, 0, ConnectionType.SIMILARITY, 0.82),  # LLM <-> Transformer
        
        # Programming cluster connections
        (2, 4, ConnectionType.CONTEXT, 0.75),    # React <-> 3D Visualization
        (4, 2, ConnectionType.CONTEXT, 0.72),    # 3D Visualization <-> React
        
        # Cross-domain connections (AI + Programming)
        (1, 3, ConnectionType.SIMILARITY, 0.65), # LLM <-> Vector Embeddings
        (3, 1, ConnectionType.SIMILARITY, 0.62), # Vector Embeddings <-> LLM
        (3, 4, ConnectionType.CONTEXT, 0.60),    # Vector Embeddings <-> 3D Visualization
        (4, 3, ConnectionType.CONTEXT, 0.58),    # 3D Visualization <-> Vector Embeddings
        
        # More AI connections
        (0, 3, ConnectionType.CONTEXT, 0.55),    # Transformer <-> Vector Embeddings
        (3, 0, ConnectionType.CONTEXT, 0.53),    # Vector Embeddings <-> Transformer
        
        # Technical implementation connections
        (2, 3, ConnectionType.CONTEXT, 0.50),    # React <-> Vector Embeddings
        (3, 2, ConnectionType.CONTEXT, 0.48),    # Vector Embeddings <-> React
    ]
    
    logger.info(f"Creating {len(connections_to_create)} connections...")
    for source_idx, target_idx, conn_type, weight in connections_to_create:
        if source_idx < len(nodes) and target_idx < len(nodes):
            source_node = nodes[source_idx]
            target_node = nodes[target_idx]
            
            success = memory_graph.create_connection(source_node.id, target_node.id, conn_type, weight)
            if success:
                logger.info(f"  ✅ Connected {source_node.concept} -> {target_node.concept} ({weight})")
            else:
                logger.warning(f"  ❌ Failed to connect {source_node.concept} -> {target_node.concept}")
    
    logger.info(f"Created {len(nodes)} demo memory nodes with 3D positioning")
    return memory_graph

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )