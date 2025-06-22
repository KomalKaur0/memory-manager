"""
Admin API endpoints for system management and monitoring
"""
import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from src.agents.model_client import ModelProvider

logger = logging.getLogger(__name__)

# Pydantic models
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    components: Dict[str, str]
    timestamp: str

class ModelSwitchRequest(BaseModel):
    """Request model for switching AI models"""
    provider: str = Field(..., description="Model provider: claude, gemini, or mock")

class ModelStatusResponse(BaseModel):
    """Response model for model status"""
    current_provider: str
    available_providers: list[str]
    provider_status: Dict[str, bool]

# Router instance
admin_router = APIRouter()

def get_memory_graph(request: Request):
    """Dependency to get memory graph instance"""
    if not hasattr(request.app.state, 'memory_graph'):
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    return request.app.state.memory_graph

def get_model_client(request: Request):
    """Dependency to get model client instance"""
    if not hasattr(request.app.state, 'claude_client'):
        raise HTTPException(status_code=503, detail="Model client not initialized")
    return request.app.state.claude_client

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

@admin_router.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status(
    model_client = Depends(get_model_client)
):
    """Get current AI model status and available providers"""
    try:
        if hasattr(model_client, 'get_current_provider'):
            # Unified model client
            current_provider = model_client.get_current_provider().value
            available_providers = [p.value for p in model_client.get_available_providers()]
            provider_status = {
                p.value: model_client.is_provider_available(p) 
                for p in ModelProvider
            }
        else:
            # Legacy client - assume Claude
            current_provider = "claude"
            available_providers = ["claude"]
            provider_status = {"claude": True, "gemini": False, "mock": False}
        
        return ModelStatusResponse(
            current_provider=current_provider,
            available_providers=available_providers,
            provider_status=provider_status
        )
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model status")

@admin_router.post("/model/switch")
async def switch_model_provider(
    request: ModelSwitchRequest,
    model_client = Depends(get_model_client)
):
    """Switch to a different AI model provider"""
    try:
        if not hasattr(model_client, 'switch_provider'):
            raise HTTPException(
                status_code=400, 
                detail="Model switching not supported by current client"
            )
        
        success = model_client.switch_provider(request.provider)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to switch to provider: {request.provider}"
            )
        
        logger.info(f"Successfully switched to model provider: {request.provider}")
        
        return {
            "message": f"Successfully switched to {request.provider}",
            "current_provider": model_client.get_current_provider().value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to switch model provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))