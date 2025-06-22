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
import uuid
import random

from src.agents.relevance_agent import RelevanceAgent, QueryContext, QueryType
from src.agents.filter_agent import FilterAgent, UserPreferences, ResponseContext
from .websocket_manager import websocket_manager

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

def get_connection_agent(request: Request):
    """Dependency to get connection agent instance"""
    if not hasattr(request.app.state, 'connection_agent'):
        raise HTTPException(status_code=503, detail="Connection agent not initialized")
    return request.app.state.connection_agent

def get_claude_client(request: Request):
    """Dependency to get Claude client instance"""
    if not hasattr(request.app.state, 'claude_client'):
        raise HTTPException(status_code=503, detail="Claude client not initialized")
    return request.app.state.claude_client

@chat_router.post("/send", response_model=ChatResponse)
async def send_message(
    request: SendMessageRequest,
    hybrid_retriever = Depends(get_hybrid_retriever),
    memory_graph = Depends(get_memory_graph),
    connection_agent = Depends(get_connection_agent),
    claude_client = Depends(get_claude_client)
):
    """Send a chat message and get AI response with intelligent memory retrieval and co-access feedback"""
    try:
        start_time = datetime.now()
        session_id = str(uuid.uuid4())  # Generate unique session ID
        
        # Create query context from conversation history
        conversation_history = [msg.content for msg in request.conversation_history[-5:]]  # Last 5 messages
        query_context = QueryContext(
            query=request.content,
            conversation_history=conversation_history,
            user_intent="chat",
            domain="general",
            query_type=QueryType.SEARCH
        )
        
        # Step 1: Search for candidate memories
        search_results = await hybrid_retriever.search_memories(
            query=request.content,
            max_results=15,  # Get more candidates for filtering
            use_graph_expansion=True
        )
        
        # Step 2: Get all memories for relevance evaluation
        candidate_memories = []
        for result in search_results:
            node = memory_graph.get_node(result["memory_id"])
            if node:
                candidate_memories.append(node)
        
        # Step 3: Use RelevanceAgent to evaluate and score memories
        relevance_agent = RelevanceAgent(memory_graph=memory_graph, claude_client=claude_client)
        
        # Get reference memory IDs (recently accessed memories in this session)
        reference_memory_ids = [result["memory_id"] for result in search_results[:3]]
        
        # Evaluate relevance for each memory
        memory_relevance_scores = []
        for memory in candidate_memories:
            relevance_score = relevance_agent.evaluate_relevance(
                memory=memory,
                query=request.content,
                context=query_context,
                reference_memory_ids=reference_memory_ids
            )
            memory_relevance_scores.append((memory, relevance_score))
        
        # Step 4: Use FilterAgent to select final memories
        filter_agent = FilterAgent()
        user_preferences = UserPreferences(
            max_memories=5,
            prefer_recent=True,
            avoid_redundancy=True,
            relevance_threshold=0.3,
            diversity_factor=0.7
        )
        response_context = ResponseContext(
            response_type="chat",
            user_context="conversation",
            conversation_history=conversation_history,
            platform="api"
        )
        
        # Extract memories and scores for filtering
        memories_for_filtering = [item[0] for item in memory_relevance_scores]
        relevance_scores = [item[1] for item in memory_relevance_scores]
        
        filter_result = filter_agent.filter_for_response(
            candidate_memories=memories_for_filtering,
            relevance_scores=relevance_scores,
            user_preferences=user_preferences,
            response_context=response_context
        )
        
        # Step 5: Record memory access events and update access counts
        selected_memories = filter_result.selected_memories
        final_relevance_scores = filter_result.relevance_scores
        
        memory_access_events = []
        for i, memory in enumerate(selected_memories):
            relevance_score = final_relevance_scores[i]
            
            access_event = {
                "node_id": memory.id,
                "concept": memory.concept,
                "access_type": "read",
                "timestamp": datetime.now().timestamp(),
                "relevance_score": relevance_score.overall,
                "confidence": relevance_score.confidence,
                "reasoning": relevance_score.reasoning,
                "position_3d": list(memory.position_3d)
            }
            memory_access_events.append(access_event)
            
            # Broadcast memory access event in real-time
            await websocket_manager.broadcast_memory_access(access_event)
            
            # Update memory access count
            memory.update_access()
        
        # Step 6: Generate AI response
        ai_response = await _generate_ai_response_with_context(
            request.content, selected_memories, final_relevance_scores, claude_client
        )
        
        # Step 7: Evaluate response quality and record co-access feedback
        response_quality = await _evaluate_response_quality(
            query=request.content,
            memories=selected_memories,
            response=ai_response,
            claude_client=claude_client
        )
        
        # Step 8: Record co-access with feedback for connection learning
        if len(selected_memories) >= 2:
            memory_ids = [memory.id for memory in selected_memories]
            relevance_scores_list = [score.overall for score in final_relevance_scores]
            
            connection_agent.record_co_access_with_feedback(
                memory_ids=memory_ids,
                query=request.content,
                relevance_scores=relevance_scores_list,
                response_quality=response_quality,
                session_id=session_id
            )
            
            # Record connection modifications for relevance agent learning
            for _ in range(len(selected_memories) - 1):  # Number of potential connections
                relevance_agent.record_connection_modification()
        
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
                "memory_id": memory.id,
                "concept": memory.concept,
                "summary": memory.summary,
                "relevance_score": score.overall,
                "confidence": score.confidence,
                "reasoning": score.reasoning,
                "must_keep": score.must_keep
            } for memory, score in zip(selected_memories, final_relevance_scores)],
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
async def websocket_chat(
    websocket: WebSocket,
    hybrid_retriever = Depends(get_hybrid_retriever),
    memory_graph = Depends(get_memory_graph),
    connection_agent = Depends(get_connection_agent),
    claude_client = Depends(get_claude_client)
):
    """WebSocket endpoint for real-time chat with memory visualization"""
    connection_id = str(uuid.uuid4())
    
    # Establish connection
    connected = await websocket_manager.connect(websocket, connection_id, "chat")
    if not connected:
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_content = data.get("content", "")
            conversation_history = data.get("conversation_history", [])
            
            if not message_content:
                await websocket_manager.send_to_connection(connection_id, {
                    "type": "error",
                    "data": {"message": "Empty message content"}
                })
                continue
            
            # Send thinking status
            await websocket_manager.send_thinking_status(
                connection_id, "searching", "Searching for relevant memories..."
            )
            
            # Process message using the same logic as send_message endpoint
            try:
                session_id = str(uuid.uuid4())
                
                # Create query context
                query_context = QueryContext(
                    query=message_content,
                    conversation_history=conversation_history[-5:],
                    user_intent="chat",
                    domain="general",
                    query_type=QueryType.SEARCH
                )
                
                # Search for candidate memories
                search_results = await hybrid_retriever.search_memories(
                    query=message_content,
                    max_results=15,
                    use_graph_expansion=True
                )
                
                # Send thinking status
                await websocket_manager.send_thinking_status(
                    connection_id, "processing", "Evaluating memory relevance..."
                )
                
                # Process with agents (similar to send_message)
                candidate_memories = []
                for result in search_results:
                    node = memory_graph.get_node(result["memory_id"])
                    if node:
                        candidate_memories.append(node)
                
                # Use RelevanceAgent
                relevance_agent = RelevanceAgent(memory_graph=memory_graph, claude_client=claude_client)
                reference_memory_ids = [result["memory_id"] for result in search_results[:3]]
                
                memory_relevance_scores = []
                for memory in candidate_memories:
                    relevance_score = relevance_agent.evaluate_relevance(
                        memory=memory,
                        query=message_content,
                        context=query_context,
                        reference_memory_ids=reference_memory_ids
                    )
                    memory_relevance_scores.append((memory, relevance_score))
                
                # Use FilterAgent
                filter_agent = FilterAgent()
                user_preferences = UserPreferences(
                    max_memories=5,
                    prefer_recent=True,
                    avoid_redundancy=True,
                    relevance_threshold=0.3,
                    diversity_factor=0.7
                )
                response_context = ResponseContext(
                    response_type="chat",
                    user_context="conversation",
                    conversation_history=conversation_history,
                    platform="websocket"
                )
                
                memories_for_filtering = [item[0] for item in memory_relevance_scores]
                relevance_scores = [item[1] for item in memory_relevance_scores]
                
                filter_result = filter_agent.filter_for_response(
                    candidate_memories=memories_for_filtering,
                    relevance_scores=relevance_scores,
                    user_preferences=user_preferences,
                    response_context=response_context
                )
                
                # Send thinking status
                await websocket_manager.send_thinking_status(
                    connection_id, "generating", "Generating response with selected memories..."
                )
                
                # Record memory access events (will broadcast automatically)
                selected_memories = filter_result.selected_memories
                final_relevance_scores = filter_result.relevance_scores
                
                for i, memory in enumerate(selected_memories):
                    relevance_score = final_relevance_scores[i]
                    
                    access_event = {
                        "node_id": memory.id,
                        "concept": memory.concept,
                        "access_type": "read",
                        "timestamp": datetime.now().timestamp(),
                        "relevance_score": relevance_score.overall,
                        "confidence": relevance_score.confidence,
                        "reasoning": relevance_score.reasoning,
                        "position_3d": list(memory.position_3d)
                    }
                    
                    # Broadcast to all connections (including this one)
                    await websocket_manager.broadcast_memory_access(access_event)
                    
                    # Update memory access count
                    memory.update_access()
                    
                    # Small delay for visualization effect
                    await asyncio.sleep(0.3)
                
                # Generate AI response
                ai_response = await _generate_ai_response_with_context(
                    message_content, selected_memories, final_relevance_scores, claude_client
                )
                
                # Evaluate response quality and record co-access feedback
                response_quality = await _evaluate_response_quality(
                    query=message_content,
                    memories=selected_memories,
                    response=ai_response,
                    claude_client=claude_client
                )
                
                # Record co-access with feedback
                if len(selected_memories) >= 2:
                    memory_ids = [memory.id for memory in selected_memories]
                    relevance_scores_list = [score.overall for score in final_relevance_scores]
                    
                    connection_agent.record_co_access_with_feedback(
                        memory_ids=memory_ids,
                        query=message_content,
                        relevance_scores=relevance_scores_list,
                        response_quality=response_quality,
                        session_id=session_id
                    )
                
                # Send final response
                await websocket_manager.send_to_connection(connection_id, {
                    "type": "response",
                    "data": {
                        "content": ai_response,
                        "memory_count": len(selected_memories),
                        "processing_time": 0,  # Calculate if needed
                        "response_quality": response_quality,
                        "timestamp": datetime.now().timestamp()
                    }
                })
                
                # Send completion signal
                await websocket_manager.send_to_connection(connection_id, {
                    "type": "complete",
                    "data": {"session_id": session_id}
                })
                
            except Exception as e:
                logger.error(f"WebSocket message processing error: {e}")
                await websocket_manager.send_to_connection(connection_id, {
                    "type": "error",
                    "data": {"message": f"Failed to process message: {str(e)}"}
                })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        websocket_manager.disconnect(connection_id)

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

async def _generate_ai_response_with_context(
    user_message: str, 
    selected_memories: List, 
    relevance_scores: List,
    claude_client=None
) -> str:
    """
    Generate AI response based on user message and intelligently selected memories using Claude
    """
    if not selected_memories:
        return f"I understand you're asking about '{user_message}', but I don't have specific memories related to this topic yet."
    
    # Prepare memory context for Claude
    memory_context = []
    for i, (memory, score) in enumerate(zip(selected_memories, relevance_scores)):
        context_entry = f"Memory {i+1} (relevance: {score.overall:.2f}):\n"
        context_entry += f"- Concept: {memory.concept}\n"
        context_entry += f"- Summary: {memory.summary}\n"
        context_entry += f"- Content: {memory.full_content[:300]}{'...' if len(memory.full_content) > 300 else ''}\n"
        context_entry += f"- Tags: {', '.join(memory.tags)}\n"
        if score.must_keep:
            context_entry += "- Note: Critical memory\n"
        memory_context.append(context_entry)
    
    # Try to use Claude for intelligent response generation
    if claude_client:
        try:
            # Create prompt for Claude to generate response using memory context
            claude_prompt = f"""You are an AI assistant with access to a memory system. A user has asked you a question, and your memory system has retrieved relevant memories to help you answer.

User's Question: {user_message}

Relevant Memories Retrieved:
{chr(10).join(memory_context)}

Based on these memories and your general knowledge, please provide a helpful, accurate response to the user's question. Use the memory information to inform your answer, but don't just list the memories - synthesize them into a coherent, useful response.

If the memories don't contain enough information to fully answer the question, acknowledge what you can answer based on the memories and what limitations you have.

Keep your response conversational and helpful."""

            # Call Claude API
            if hasattr(claude_client, 'messages') and hasattr(claude_client.messages, 'create'):
                message = claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": claude_prompt}]
                )
                claude_response = message.content[0].text if message.content else ""
                
                if claude_response.strip():
                    return claude_response.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate response with Claude: {e}. Falling back to template response.")
    
    # Fallback to template response if Claude is not available or fails
    memory_insights = []
    for i, (memory, score) in enumerate(zip(selected_memories, relevance_scores)):
        insight = f"From my memory about '{memory.concept}' (relevance: {score.overall:.2f}): {memory.summary}"
        if score.must_keep:
            insight += " [Critical memory]"
        memory_insights.append(insight)
    
    response_parts = [
        f"Based on my analysis of {len(selected_memories)} relevant memories, here's my response to '{user_message}':",
        "",
        *[f"{i+1}. {insight}" for i, insight in enumerate(memory_insights[:3])],  # Top 3 insights
        "",
        "These memories provide relevant context and insights that inform my understanding of your question."
    ]
    
    return "\n".join(response_parts)

async def _evaluate_response_quality(
    query: str,
    memories: List,
    response: str,
    claude_client
) -> float:
    """
    Evaluate the quality of the response given the query, memories used, and generated response
    """
    try:
        if not claude_client:
            # Fallback heuristic scoring
            return _heuristic_response_quality(query, memories, response)
        
        # Prepare context for Claude evaluation
        memory_context = "\n".join([
            f"- {memory.concept}: {memory.summary}" 
            for memory in memories[:3]
        ])
        
        evaluation_prompt = f"""
Evaluate the quality of this AI response given the context:

Query: {query}

Memories used:
{memory_context}

Response: {response}

Rate the response quality on a scale of 0.0 to 1.0 considering:
1. How well the response addresses the query
2. How effectively it uses the provided memories
3. Coherence and helpfulness of the response
4. Appropriateness of memory selection

Respond with just a number between 0.0 and 1.0.
"""
        
        # Call Claude for evaluation
        if hasattr(claude_client, 'messages') and hasattr(claude_client.messages, 'create'):
            message = claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,
                messages=[{"role": "user", "content": evaluation_prompt}]
            )
            response_text = message.content[0].text if message.content else "0.5"
        else:
            response_text = "0.5"
        
        # Parse the score
        import re
        match = re.search(r'(0\.\d+|1\.0)', response_text)
        if match:
            return float(match.group(1))
        else:
            return 0.5
            
    except Exception as e:
        logger.warning(f"Failed to evaluate response quality with Claude: {e}")
        return _heuristic_response_quality(query, memories, response)

def _heuristic_response_quality(query: str, memories: List, response: str) -> float:
    """Fallback heuristic for response quality scoring"""
    base_score = 0.5
    
    # Boost for using multiple memories
    if len(memories) >= 2:
        base_score += 0.1
    
    # Boost for mentioning memory concepts in response
    memory_concepts = [memory.concept.lower() for memory in memories]
    response_lower = response.lower()
    
    concept_mentions = sum(1 for concept in memory_concepts if concept in response_lower)
    if concept_mentions > 0:
        base_score += min(0.3, concept_mentions * 0.1)
    
    # Boost for response length (indicates thoughtfulness)
    if len(response) > 200:
        base_score += 0.1
    
    return min(1.0, base_score)

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