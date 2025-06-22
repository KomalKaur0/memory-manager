"""
Chat API endpoints for AI conversation with memory integration
"""
import logging
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

logger = logging.getLogger(__name__)

# Pydantic models
class ChatMessage(BaseModel):
    """Chat message model"""
    id: str
    content: str
    role: str = Field(..., pattern="^(user|assistant)$")
    timestamp: float
    memory_accesses: List[Dict[str, Any]] = Field(default_factory=list)
    thinking: bool = False

class SendMessageRequest(BaseModel):
    """Request model for sending a chat message"""
    content: str = Field(..., description="Message content")
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="Recent conversation")

class ChatResponse(BaseModel):
    """Response model for chat messages"""
    message: ChatMessage
    retrieved_memories: List[Dict[str, Any]] = Field(default_factory=list)
    memory_access_events: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time: float

# Router instance
chat_router = APIRouter()

def get_hybrid_retriever(request: Request):
    """Dependency to get hybrid retriever instance"""
    if not hasattr(request.app.state, 'hybrid_retriever'):
        raise HTTPException(status_code=503, detail="Hybrid retriever not initialized")
    return request.app.state.hybrid_retriever

def get_memory_graph(request: Request):
    """Dependency to get memory graph instance"""
    if not hasattr(request.app.state, 'memory_graph'):
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    return request.app.state.memory_graph

@chat_router.post("/send", response_model=ChatResponse)
async def send_message(
    request: SendMessageRequest,
    hybrid_retriever = Depends(get_hybrid_retriever),
    memory_graph = Depends(get_memory_graph)
):
    """Send a chat message and get AI response with memory retrieval"""
    try:
        start_time = datetime.now()
        
        # Search for relevant memories
        search_results = await hybrid_retriever.search_memories(
            query=request.content,
            max_results=5,
            use_graph_expansion=True
        )
        
        # Record memory access events
        memory_access_events = []
        for result in search_results:
            memory_access_events.append({
                "node_id": result["memory_id"],
                "access_type": "read",
                "timestamp": datetime.now().timestamp(),
                "similarity_score": result["combined_score"]
            })
            
            # Update memory access count
            node = memory_graph.get_node(result["memory_id"])
            if node:
                node.update_access()
        
        # Generate AI response (simplified for MVP)
        ai_response = await _generate_ai_response(request.content, search_results)
        
        # Create response message
        response_message = ChatMessage(
            id=f"msg_{datetime.now().timestamp()}",
            content=ai_response,
            role="assistant",
            timestamp=datetime.now().timestamp(),
            memory_accesses=memory_access_events
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            message=response_message,
            retrieved_memories=[{
                "memory_id": r["memory_id"],
                "concept": r["node"].concept,
                "summary": r["node"].summary,
                "score": r["combined_score"],
                "source": r["source"]
            } for r in search_results],
            memory_access_events=memory_access_events,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat message processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@chat_router.post("/stream")
async def stream_chat_response(
    request: SendMessageRequest,
    hybrid_retriever = Depends(get_hybrid_retriever),
    memory_graph = Depends(get_memory_graph)
):
    """Stream chat response with real-time memory access updates"""
    
    async def generate_stream():
        try:
            # Start thinking phase
            yield f"data: {json.dumps({'type': 'thinking', 'data': {'status': 'searching_memories'}})}\n\n"
            
            # Search for relevant memories
            search_results = await hybrid_retriever.search_memories(
                query=request.content,
                max_results=5,
                use_graph_expansion=True
            )
            
            # Send memory access events
            for result in search_results:
                memory_event = {
                    "type": "memory_access",
                    "data": {
                        "node_id": result["memory_id"],
                        "concept": result["node"].concept,
                        "access_type": "read",
                        "score": result["combined_score"],
                        "timestamp": datetime.now().timestamp()
                    }
                }
                yield f"data: {json.dumps(memory_event)}\n\n"
                
                # Small delay to simulate real-time access
                await asyncio.sleep(0.2)
            
            # Generate response
            yield f"data: {json.dumps({'type': 'thinking', 'data': {'status': 'generating_response'}})}\n\n"
            
            ai_response = await _generate_ai_response(request.content, search_results)
            
            # Send the final response
            response_data = {
                "type": "response",
                "data": {
                    "content": ai_response,
                    "memory_count": len(search_results),
                    "timestamp": datetime.now().timestamp()
                }
            }
            yield f"data: {json.dumps(response_data)}\n\n"
            
            # Signal completion
            yield f"data: {json.dumps({'type': 'complete', 'data': {}})}\n\n"
            
        except Exception as e:
            error_data = {
                "type": "error",
                "data": {"message": str(e)}
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@chat_router.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat with memory visualization"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_content = data.get("content", "")
            
            if not message_content:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Empty message content"}
                })
                continue
            
            # Send thinking status
            await websocket.send_json({
                "type": "thinking",
                "data": {"status": "processing", "message": "Searching memories..."}
            })
            
            # Simulate memory access (replace with actual retrieval)
            await asyncio.sleep(0.5)
            
            # Send mock memory access event
            await websocket.send_json({
                "type": "memory_access",
                "data": {
                    "node_id": "mock_node_1",
                    "concept": "Mock Memory Access",
                    "access_type": "read",
                    "timestamp": datetime.now().timestamp()
                }
            })
            
            # Send response
            await websocket.send_json({
                "type": "response",
                "data": {
                    "content": f"I received your message: '{message_content}'. This is a mock response.",
                    "timestamp": datetime.now().timestamp()
                }
            })
            
            # Send completion signal
            await websocket.send_json({
                "type": "complete",
                "data": {}
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)}
            })
        except:
            pass

@chat_router.get("/history")
async def get_chat_history(
    limit: int = 50,
    offset: int = 0
):
    """Get chat message history (placeholder implementation)"""
    try:
        # This would integrate with actual message storage
        mock_messages = [
            {
                "id": f"msg_{i}",
                "content": f"Mock message {i}",
                "role": "user" if i % 2 == 0 else "assistant",
                "timestamp": datetime.now().timestamp() - (i * 60),
                "memory_accesses": []
            }
            for i in range(min(limit, 10))
        ]
        
        return {
            "messages": mock_messages[offset:offset + limit],
            "total": len(mock_messages),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Chat history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

async def _generate_ai_response(user_message: str, search_results: List[Dict]) -> str:
    """
    Generate AI response based on user message and retrieved memories
    
    This is a simplified implementation for MVP.
    In production, this would integrate with an actual LLM.
    """
    # Simple response generation based on retrieved memories
    if not search_results:
        return f"I understand you're asking about '{user_message}', but I don't have specific memories related to this topic yet."
    
    # Create response mentioning relevant memories
    memory_concepts = [result["node"].concept for result in search_results[:3]]
    
    response_parts = [
        f"Based on what I remember about {', '.join(memory_concepts)}, "
        f"here's my response to '{user_message}':"
    ]
    
    # Add insights from top memories
    for i, result in enumerate(search_results[:2]):
        memory = result["node"]
        response_parts.append(
            f"\n\n{i+1}. From my memory about '{memory.concept}': {memory.summary}"
        )
    
    # Add synthetic reasoning
    response_parts.append(
        f"\n\nThis connects to your question because these memories show relevant patterns "
        f"and context that help inform my understanding."
    )
    
    return "".join(response_parts)