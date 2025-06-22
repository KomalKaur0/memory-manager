"""
FastAPI main application for AI Memory System
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.memory_api import memory_router
from src.api.search_api import search_router
from src.api.admin_api import admin_router
from src.api.chat_api import chat_router
from src.core.memory_graph import MemoryGraph
from src.retrieval.embedding_search_mock import EmbeddingSearch
from src.retrieval.hybrid_retriever_simple import HybridRetriever
from src.agents.relevance_agent import get_claude_client_from_env
from src.agents.connection_agent import ConnectionAgent
from src.retrieval.embedding_search import get_embedding_config_from_env

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
    embedding_search = EmbeddingSearch(config=embedding_config)
    await embedding_search.initialize()
    
    hybrid_retriever = HybridRetriever(
        memory_graph=memory_graph,
        embedding_search=embedding_search
    )
    
    # Initialize connection agent for co-access tracking
    connection_agent = ConnectionAgent(memory_graph=memory_graph)
    
    # Initialize Claude client with API key from .env
    claude_client = get_claude_client_from_env()
    
    # Store instances in app state for access in routes
    app.state.memory_graph = memory_graph
    app.state.embedding_search = embedding_search
    app.state.hybrid_retriever = hybrid_retriever
    app.state.connection_agent = connection_agent
    app.state.claude_client = claude_client
    
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
    allow_origins=["*"],  # Configure based on your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(memory_router, prefix="/api/memory", tags=["memory"])
app.include_router(search_router, prefix="/api/search", tags=["search"])
app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )