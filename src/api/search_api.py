"""
Search API endpoints for memory retrieval and discovery
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Pydantic models
class SearchRequest(BaseModel):
    """Request model for memory search"""
    query: str = Field(..., description="Search query text")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    use_graph_expansion: bool = Field(default=True, description="Whether to use graph-based expansion")
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")

class SearchResult(BaseModel):
    """Response model for search results"""
    memory_id: str
    concept: str
    summary: str
    content: str
    tags: List[str]
    embedding_score: float
    graph_score: float
    combined_score: float
    source: str

class SearchResponse(BaseModel):
    """Response model for search operations"""
    results: List[SearchResult]
    query: str
    total_found: int
    search_time: float
    explanation: str

# Router instance
search_router = APIRouter()

def get_hybrid_retriever(request: Request):
    """Dependency to get hybrid retriever instance"""
    if not hasattr(request.app.state, 'hybrid_retriever'):
        raise HTTPException(status_code=503, detail="Hybrid retriever not initialized")
    return request.app.state.hybrid_retriever

@search_router.post("/memories", response_model=SearchResponse)
async def search_memories(
    search_request: SearchRequest,
    hybrid_retriever = Depends(get_hybrid_retriever)
):
    """Search for memories using hybrid retrieval"""
    try:
        start_time = datetime.now()
        
        results = await hybrid_retriever.search_memories(
            query=search_request.query,
            max_results=search_request.max_results,
            use_graph_expansion=search_request.use_graph_expansion
        )
        
        search_results = []
        for result in results:
            node = result["node"]
            search_result = SearchResult(
                memory_id=result["memory_id"],
                concept=node.concept,
                summary=node.summary,
                content=node.full_content,
                tags=node.tags,
                embedding_score=result["embedding_score"],
                graph_score=result["graph_score"],
                combined_score=result["combined_score"],
                source=result["source"]
            )
            search_results.append(search_result)
        
        explanation = hybrid_retriever.explain_results(results, search_request.query)
        search_time = (datetime.now() - start_time).total_seconds()
        
        return SearchResponse(
            results=search_results,
            query=search_request.query,
            total_found=len(results),
            search_time=search_time,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search operation failed: {str(e)}")