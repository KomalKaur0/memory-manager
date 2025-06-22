"""
Memory API endpoints for managing memory nodes and connections
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ..core.memory_node import MemoryNode, ConnectionType
from ..core.memory_graph import MemoryGraph

logger = logging.getLogger(__name__)

# Pydantic models for API
class MemoryNodeCreate(BaseModel):
    """Request model for creating a new memory node"""
    concept: str = Field(..., description="Main concept of the memory")
    summary: str = Field(..., description="Brief summary of the memory")
    content: str = Field(..., description="Full content of the memory")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    keywords: List[str] = Field(default_factory=list, description="Key terms")
    
class MemoryNodeResponse(BaseModel):
    """Response model for memory node data"""
    id: str
    concept: str
    summary: str
    content: str
    tags: List[str]
    keywords: List[str]
    connections: Dict[str, dict]
    access_count: int
    importance_score: float

class MemoryAccessEvent(BaseModel):
    """Model for memory access events"""
    node_id: str
    access_type: str = Field(..., pattern="^(read|write|strengthen|traverse)$")
    timestamp: float
    connection_id: Optional[str] = None
    weight_change: Optional[float] = None

class ConnectionCreate(BaseModel):
    """Request model for creating connections"""
    source_id: str
    target_id: str
    connection_type: str
    initial_weight: float = Field(default=0.0, ge=0.0, le=1.0)

# Router instance
memory_router = APIRouter()

def get_memory_graph(request: Request) -> MemoryGraph:
    """Dependency to get memory graph instance"""
    if not hasattr(request.app.state, 'memory_graph'):
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    return request.app.state.memory_graph

def get_hybrid_retriever(request: Request):
    """Dependency to get hybrid retriever instance"""
    if not hasattr(request.app.state, 'hybrid_retriever'):
        raise HTTPException(status_code=503, detail="Hybrid retriever not initialized")
    return request.app.state.hybrid_retriever

@memory_router.get("/nodes", response_model=Dict[str, MemoryNodeResponse])
async def get_all_memory_nodes(
    limit: Optional[int] = 100,
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """Get all memory nodes with optional limit"""
    try:
        nodes = {}
        node_items = list(memory_graph.nodes.items())
        
        if limit:
            node_items = node_items[:limit]
            
        for node_id, node in node_items:
            nodes[node_id] = _convert_node_to_response(node)
            
        return nodes
    except Exception as e:
        logger.error(f"Error fetching memory nodes: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch memory nodes")

@memory_router.get("/nodes/{node_id}", response_model=MemoryNodeResponse)
async def get_memory_node(
    node_id: str,
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """Get a specific memory node by ID"""
    try:
        node = memory_graph.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail="Memory node not found")
        return _convert_node_to_response(node)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching memory node {node_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch memory node")

@memory_router.post("/nodes", response_model=MemoryNodeResponse)
async def create_memory_node(
    node_data: MemoryNodeCreate,
    memory_graph: MemoryGraph = Depends(get_memory_graph),
    hybrid_retriever = Depends(get_hybrid_retriever)
):
    """Create a new memory node"""
    try:
        # Generate embedding for the content
        embedding = await hybrid_retriever.embedding_search.get_embedding(
            f"{node_data.concept} {node_data.summary} {node_data.content}"
        )
        
        # Create the memory node
        node = MemoryNode(
            concept=node_data.concept,
            summary=node_data.summary,
            full_content=node_data.content,
            tags=node_data.tags,
            keywords=node_data.keywords
        )
        
        # Add to graph
        node_id = memory_graph.add_node(node)
        
        # Store embedding in the embedding search service
        hybrid_retriever.embedding_search.store_embedding(node_id, embedding)
        
        # Auto-connect to similar nodes
        await _auto_connect_node(node_id, memory_graph, hybrid_retriever)
        
        return _convert_node_to_response(node)
        
    except Exception as e:
        logger.error(f"Error creating memory node: {e}")
        raise HTTPException(status_code=500, detail="Failed to create memory node")

@memory_router.put("/nodes/{node_id}", response_model=MemoryNodeResponse)
async def update_memory_node(
    node_id: str,
    node_data: MemoryNodeCreate,
    memory_graph: MemoryGraph = Depends(get_memory_graph),
    hybrid_retriever = Depends(get_hybrid_retriever)
):
    """Update an existing memory node"""
    try:
        # Get existing node
        existing_node = memory_graph.get_node(node_id)
        if not existing_node:
            raise HTTPException(status_code=404, detail="Memory node not found")
        
        # Update fields
        existing_node.concept = node_data.concept
        existing_node.summary = node_data.summary
        existing_node.full_content = node_data.content
        existing_node.tags = node_data.tags
        existing_node.keywords = node_data.keywords
        
        # Regenerate embedding
        embedding = await hybrid_retriever.embedding_search.get_embedding(
            f"{node_data.concept} {node_data.summary} {node_data.content}"
        )
        # Store embedding in the embedding search service
        hybrid_retriever.embedding_search.store_embedding(node_id, embedding)
        
        return _convert_node_to_response(existing_node)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating memory node {node_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update memory node")

@memory_router.delete("/nodes/{node_id}")
async def delete_memory_node(
    node_id: str,
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """Delete a memory node"""
    try:
        success = memory_graph.remove_node(node_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory node not found")
        return {"message": "Memory node deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory node {node_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete memory node")

@memory_router.post("/connections")
async def create_connection(
    connection_data: ConnectionCreate,
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """Create a connection between two memory nodes"""
    try:
        success = memory_graph.create_connection(
            source_id=connection_data.source_id,
            target_id=connection_data.target_id,
            connection_type=ConnectionType(connection_data.connection_type),
            initial_weight=connection_data.initial_weight
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to create connection - nodes may not exist")
            
        return {"message": "Connection created successfully"}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid connection type: {e}")
    except Exception as e:
        logger.error(f"Error creating connection: {e}")
        raise HTTPException(status_code=500, detail="Failed to create connection")

@memory_router.post("/access")
async def record_memory_access(
    access_event: MemoryAccessEvent,
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """Record a memory access event"""
    try:
        # Update node access
        node = memory_graph.get_node(access_event.node_id)
        if not node:
            raise HTTPException(status_code=404, detail="Memory node not found")
        
        # If strengthening a connection
        if access_event.access_type == "strengthen" and access_event.connection_id:
            memory_graph.strengthen_connection(
                access_event.node_id,
                access_event.connection_id,
                access_event.weight_change or 0.1
            )
        
        return {"message": "Memory access recorded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording memory access: {e}")
        raise HTTPException(status_code=500, detail="Failed to record memory access")

@memory_router.get("/nodes/{node_id}/connected")
async def get_connected_nodes(
    node_id: str,
    max_depth: int = 2,
    min_weight: float = 0.1,
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """Get nodes connected to a specific node"""
    try:
        connected = memory_graph.get_connected_nodes(
            node_id=node_id,
            max_depth=max_depth,
            min_weight=min_weight
        )
        
        result = []
        for connected_id, weight, depth in connected:
            node = memory_graph.get_node(connected_id)
            if node:
                result.append({
                    "node": _convert_node_to_response(node),
                    "weight": weight,
                    "depth": depth
                })
        
        return result
    except Exception as e:
        logger.error(f"Error getting connected nodes for {node_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get connected nodes")

# WebSocket for real-time memory updates
@memory_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time memory access updates"""
    await websocket.accept()
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_json()
            
            # Echo back for now (extend with real-time memory access updates)
            await websocket.send_json({
                "type": "memory_access",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

# Helper functions
def _convert_node_to_response(node: MemoryNode) -> MemoryNodeResponse:
    """Convert a MemoryNode to API response format"""
    return MemoryNodeResponse(
        id=node.id,
        concept=node.concept,
        summary=node.summary,
        content=node.full_content,
        tags=node.tags,
        keywords=node.keywords,
        connections={
            conn_id: {
                "target_node_id": conn.target_node_id,
                "connection_type": conn.connection_type.value,
                "weight": conn.weight,
                "usage_count": conn.usage_count
            }
            for conn_id, conn in node.connections.items()
        },
        access_count=node.access_count,
        importance_score=node.importance_score
    )

async def _auto_connect_node(node_id: str, memory_graph: MemoryGraph, hybrid_retriever):
    """Automatically connect a new node to similar existing nodes"""
    try:
        node = memory_graph.get_node(node_id)
        if not node:
            return
        
        # Get node embedding from the service
        embedding = await hybrid_retriever.embedding_search.get_embedding(
            f"{node.concept} {node.summary} {node.full_content}"
        )
        
        # Find similar nodes using embedding similarity
        similar_results = await hybrid_retriever.embedding_search.find_similar(
            embedding, 
            k=5, 
            threshold=0.7
        )
        
        for similar_id, similarity in similar_results:
            if similar_id != node_id and similar_id in memory_graph.nodes:
                # Create bidirectional connections
                memory_graph.create_connection(
                    node_id, similar_id, ConnectionType.SIMILARITY, similarity * 0.5
                )
                memory_graph.create_connection(
                    similar_id, node_id, ConnectionType.SIMILARITY, similarity * 0.5
                )
        
    except Exception as e:
        logger.warning(f"Failed to auto-connect node {node_id}: {e}")