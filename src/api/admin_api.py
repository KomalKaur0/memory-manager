"""
Admin API endpoints for system management and monitoring
"""
import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Pydantic models
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    components: Dict[str, str]
    timestamp: str

# Router instance
admin_router = APIRouter()

def get_memory_graph(request: Request):
    """Dependency to get memory graph instance"""
    if not hasattr(request.app.state, 'memory_graph'):
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    return request.app.state.memory_graph

@admin_router.get("/health", response_model=HealthResponse)
async def detailed_health_check(
    memory_graph = Depends(get_memory_graph)
):
    """Detailed health check of all system components"""
    try:
        components = {"memory_graph": "healthy"}
        
        return HealthResponse(
            status="healthy",
            components=components,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")