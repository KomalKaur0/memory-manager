"""
WebSocket manager for real-time memory access visualization
"""
import asyncio
import json
import logging
from typing import Dict, List, Set
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    Manages WebSocket connections for real-time memory access visualization
    """
    
    def __init__(self):
        # Active connections by connection ID
        self.active_connections: Dict[str, WebSocket] = {}
        # Connection metadata
        self.connection_metadata: Dict[str, Dict] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str, 
                     connection_type: str = "chat") -> bool:
        """
        Accept a new WebSocket connection
        
        Args:
            websocket: The WebSocket connection
            connection_id: Unique identifier for this connection
            connection_type: Type of connection ('chat', 'memory_viz', etc.)
            
        Returns:
            True if connection was successful
        """
        try:
            await websocket.accept()
            self.active_connections[connection_id] = websocket
            self.connection_metadata[connection_id] = {
                "type": connection_type,
                "connected_at": datetime.now().timestamp(),
                "last_activity": datetime.now().timestamp()
            }
            logger.info(f"WebSocket connection established: {connection_id} (type: {connection_type})")
            return True
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection {connection_id}: {e}")
            return False
    
    def disconnect(self, connection_id: str):
        """
        Remove a WebSocket connection
        
        Args:
            connection_id: ID of connection to remove
        """
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            del self.connection_metadata[connection_id]
            logger.info(f"WebSocket connection disconnected: {connection_id}")
    
    async def send_to_connection(self, connection_id: str, message: Dict):
        """
        Send a message to a specific connection
        
        Args:
            connection_id: Target connection ID
            message: Message data to send
        """
        if connection_id not in self.active_connections:
            logger.warning(f"Attempted to send to non-existent connection: {connection_id}")
            return False
            
        try:
            websocket = self.active_connections[connection_id]
            await websocket.send_json(message)
            
            # Update last activity
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["last_activity"] = datetime.now().timestamp()
            
            return True
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected during send: {connection_id}")
            self.disconnect(connection_id)
            return False
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            self.disconnect(connection_id)
            return False
    
    async def broadcast_to_type(self, connection_type: str, message: Dict):
        """
        Broadcast a message to all connections of a specific type
        
        Args:
            connection_type: Type of connections to target
            message: Message data to broadcast
        """
        target_connections = [
            conn_id for conn_id, metadata in self.connection_metadata.items()
            if metadata.get("type") == connection_type
        ]
        
        if not target_connections:
            logger.debug(f"No active connections of type '{connection_type}' for broadcast")
            return
        
        # Send to all target connections
        tasks = []
        for connection_id in target_connections:
            tasks.append(self.send_to_connection(connection_id, message))
        
        # Execute all sends concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for result in results if result is True)
        logger.debug(f"Broadcast to {successful}/{len(target_connections)} connections of type '{connection_type}'")
    
    async def broadcast_memory_access(self, memory_access_event: Dict):
        """
        Broadcast a memory access event to all relevant connections
        
        Args:
            memory_access_event: Memory access event data
        """
        message = {
            "type": "memory_access",
            "data": memory_access_event,
            "timestamp": datetime.now().timestamp()
        }
        
        # Broadcast to chat and memory visualization connections
        await self.broadcast_to_type("chat", message)
        await self.broadcast_to_type("memory_viz", message)
    
    async def broadcast_memory_update(self, node_id: str, update_type: str, node_data: Dict = None):
        """
        Broadcast a memory node update (creation, modification, deletion)
        
        Args:
            node_id: ID of the memory node
            update_type: Type of update ('created', 'updated', 'deleted', 'connection_changed')
            node_data: Optional node data for the update
        """
        message = {
            "type": "memory_update",
            "data": {
                "node_id": node_id,
                "update_type": update_type,
                "node_data": node_data,
                "timestamp": datetime.now().timestamp()
            }
        }
        
        # Broadcast to memory visualization connections
        await self.broadcast_to_type("memory_viz", message)
    
    async def broadcast_connection_change(self, source_id: str, target_id: str, 
                                        connection_type: str, weight: float, change_type: str):
        """
        Broadcast a connection change event
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            connection_type: Type of connection
            weight: Connection weight
            change_type: Type of change ('created', 'strengthened', 'weakened', 'deleted')
        """
        message = {
            "type": "connection_change",
            "data": {
                "source_id": source_id,
                "target_id": target_id,
                "connection_type": connection_type,
                "weight": weight,
                "change_type": change_type,
                "timestamp": datetime.now().timestamp()
            }
        }
        
        # Broadcast to memory visualization connections
        await self.broadcast_to_type("memory_viz", message)
    
    async def send_thinking_status(self, connection_id: str, status: str, message: str = ""):
        """
        Send thinking/processing status to a specific connection
        
        Args:
            connection_id: Target connection ID
            status: Status type ('searching', 'processing', 'generating', 'complete')
            message: Optional status message
        """
        status_message = {
            "type": "thinking",
            "data": {
                "status": status,
                "message": message,
                "timestamp": datetime.now().timestamp()
            }
        }
        
        await self.send_to_connection(connection_id, status_message)
    
    def get_connection_stats(self) -> Dict:
        """
        Get statistics about active connections
        
        Returns:
            Dictionary with connection statistics
        """
        total_connections = len(self.active_connections)
        connections_by_type = {}
        
        for metadata in self.connection_metadata.values():
            conn_type = metadata.get("type", "unknown")
            connections_by_type[conn_type] = connections_by_type.get(conn_type, 0) + 1
        
        return {
            "total_connections": total_connections,
            "connections_by_type": connections_by_type,
            "active_connection_ids": list(self.active_connections.keys())
        }
    
    async def cleanup_stale_connections(self, max_idle_seconds: int = 3600):
        """
        Remove connections that have been idle for too long
        
        Args:
            max_idle_seconds: Maximum idle time before cleanup
        """
        current_time = datetime.now().timestamp()
        stale_connections = []
        
        for conn_id, metadata in self.connection_metadata.items():
            last_activity = metadata.get("last_activity", 0)
            if current_time - last_activity > max_idle_seconds:
                stale_connections.append(conn_id)
        
        for conn_id in stale_connections:
            logger.info(f"Cleaning up stale WebSocket connection: {conn_id}")
            self.disconnect(conn_id)
        
        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale WebSocket connections")

# Global WebSocket manager instance
websocket_manager = WebSocketManager()