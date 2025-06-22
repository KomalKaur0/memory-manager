"""
Analytics API endpoints for memory statistics and access patterns
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

from ..core.memory_node import MemoryNode, ConnectionType
from ..core.memory_graph import MemoryGraph
from ..visualization.spatial_layout import SpatialLayoutEngine

logger = logging.getLogger(__name__)

# Pydantic models for analytics responses
class MemoryStats(BaseModel):
    """Overall memory statistics"""
    total_nodes: int
    total_connections: int
    avg_connections_per_node: float
    most_connected_nodes: List[Dict[str, Any]]
    connection_type_distribution: Dict[str, int]
    access_frequency_distribution: Dict[str, int]
    importance_score_distribution: Dict[str, int]

class AccessPattern(BaseModel):
    """Memory access pattern information"""
    node_id: str
    access_count: int
    last_accessed: Optional[float]
    connections_used: int
    cluster_info: Dict[str, Any]

class ConnectionStrengthStats(BaseModel):
    """Connection strength analytics"""
    strongest_connections: List[Dict[str, Any]]
    weakest_connections: List[Dict[str, Any]]
    avg_connection_weight: float
    weight_distribution: Dict[str, int]
    connection_usage_stats: Dict[str, int]

class ClusterAnalytics(BaseModel):
    """3D clustering analytics"""
    cluster_count: int
    cluster_sizes: Dict[str, int]
    cluster_centers: Dict[str, List[float]]
    inter_cluster_connections: int
    intra_cluster_connections: int

# Router instance
analytics_router = APIRouter()

def get_memory_graph(request: Request) -> MemoryGraph:
    """Dependency to get memory graph instance"""
    if not hasattr(request.app.state, 'memory_graph'):
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    return request.app.state.memory_graph

def get_spatial_layout_engine(request: Request) -> SpatialLayoutEngine:
    """Dependency to get spatial layout engine instance"""
    if not hasattr(request.app.state, 'spatial_layout_engine'):
        request.app.state.spatial_layout_engine = SpatialLayoutEngine()
    return request.app.state.spatial_layout_engine

@analytics_router.get("/memory-stats", response_model=MemoryStats)
async def get_memory_statistics(
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """Get comprehensive memory statistics"""
    try:
        nodes = list(memory_graph.nodes.values())
        
        if not nodes:
            return MemoryStats(
                total_nodes=0,
                total_connections=0,
                avg_connections_per_node=0.0,
                most_connected_nodes=[],
                connection_type_distribution={},
                access_frequency_distribution={},
                importance_score_distribution={}
            )
        
        # Basic counts
        total_nodes = len(nodes)
        total_connections = sum(len(node.connections) for node in nodes)
        avg_connections = total_connections / total_nodes if total_nodes > 0 else 0.0
        
        # Most connected nodes
        most_connected = sorted(
            [(node.id, node.concept, len(node.connections)) for node in nodes],
            key=lambda x: x[2],
            reverse=True
        )[:10]
        
        most_connected_nodes = [
            {
                "node_id": node_id,
                "concept": concept,
                "connection_count": count
            }
            for node_id, concept, count in most_connected
        ]
        
        # Connection type distribution
        connection_types = Counter()
        for node in nodes:
            for conn in node.connections.values():
                connection_types[conn.connection_type.value] += 1
        
        # Access frequency distribution
        access_buckets = defaultdict(int)
        for node in nodes:
            if node.access_count == 0:
                bucket = "never_accessed"
            elif node.access_count <= 5:
                bucket = "1-5_accesses"
            elif node.access_count <= 20:
                bucket = "6-20_accesses"
            elif node.access_count <= 50:
                bucket = "21-50_accesses"
            else:
                bucket = "50+_accesses"
            access_buckets[bucket] += 1
        
        # Importance score distribution
        importance_buckets = defaultdict(int)
        for node in nodes:
            if node.importance_score < 0.2:
                bucket = "very_low"
            elif node.importance_score < 0.4:
                bucket = "low"
            elif node.importance_score < 0.6:
                bucket = "medium"
            elif node.importance_score < 0.8:
                bucket = "high"
            else:
                bucket = "very_high"
            importance_buckets[bucket] += 1
        
        return MemoryStats(
            total_nodes=total_nodes,
            total_connections=total_connections,
            avg_connections_per_node=round(avg_connections, 2),
            most_connected_nodes=most_connected_nodes,
            connection_type_distribution=dict(connection_types),
            access_frequency_distribution=dict(access_buckets),
            importance_score_distribution=dict(importance_buckets)
        )
        
    except Exception as e:
        logger.error(f"Error getting memory statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get memory statistics")

@analytics_router.get("/access-patterns", response_model=List[AccessPattern])
async def get_access_patterns(
    limit: Optional[int] = 50,
    memory_graph: MemoryGraph = Depends(get_memory_graph),
    layout_engine: SpatialLayoutEngine = Depends(get_spatial_layout_engine)
):
    """Get memory access patterns and usage statistics"""
    try:
        nodes = list(memory_graph.nodes.values())
        
        # Sort by access count (most accessed first)
        sorted_nodes = sorted(nodes, key=lambda n: n.access_count, reverse=True)
        
        if limit:
            sorted_nodes = sorted_nodes[:limit]
        
        # Get cluster information
        cluster_info = layout_engine.get_cluster_info()
        
        patterns = []
        for node in sorted_nodes:
            # Count connections that have been used (weight > 0)
            connections_used = sum(1 for conn in node.connections.values() if conn.weight > 0)
            
            # Find which cluster this node belongs to (simplified)
            node_cluster = "unknown"
            for cluster_id, info in cluster_info.items():
                # This is a simplified cluster assignment - in practice you'd want
                # a more sophisticated method
                if node.tags and cluster_id.startswith("tag:"):
                    cluster_tag = cluster_id.split(":", 1)[1]
                    if cluster_tag in node.tags:
                        node_cluster = cluster_id
                        break
            
            patterns.append(AccessPattern(
                node_id=node.id,
                access_count=node.access_count,
                last_accessed=None,  # Would need to track this in the future
                connections_used=connections_used,
                cluster_info={
                    "cluster_id": node_cluster,
                    "position_3d": list(node.position_3d),
                    "concept": node.concept,
                    "tags": node.tags
                }
            ))
        
        return patterns
        
    except Exception as e:
        logger.error(f"Error getting access patterns: {e}")
        raise HTTPException(status_code=500, detail="Failed to get access patterns")

@analytics_router.get("/connection-strength", response_model=ConnectionStrengthStats)
async def get_connection_strength_statistics(
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """Get connection strength analytics"""
    try:
        # Collect all connections
        all_connections = []
        for node in memory_graph.nodes.values():
            for target_id, connection in node.connections.items():
                target_node = memory_graph.get_node(target_id)
                if target_node:
                    all_connections.append({
                        "source_id": node.id,
                        "source_concept": node.concept,
                        "target_id": target_id,
                        "target_concept": target_node.concept,
                        "weight": connection.weight,
                        "usage_count": connection.usage_count,
                        "connection_type": connection.connection_type.value
                    })
        
        if not all_connections:
            return ConnectionStrengthStats(
                strongest_connections=[],
                weakest_connections=[],
                avg_connection_weight=0.0,
                weight_distribution={},
                connection_usage_stats={}
            )
        
        # Sort by weight
        sorted_by_weight = sorted(all_connections, key=lambda c: c["weight"], reverse=True)
        
        # Strongest and weakest connections
        strongest_connections = sorted_by_weight[:10]
        weakest_connections = [c for c in sorted_by_weight[-10:] if c["weight"] > 0]
        
        # Average connection weight
        avg_weight = sum(c["weight"] for c in all_connections) / len(all_connections)
        
        # Weight distribution
        weight_buckets = defaultdict(int)
        for conn in all_connections:
            weight = conn["weight"]
            if weight == 0:
                bucket = "unused"
            elif weight <= 0.2:
                bucket = "very_weak"
            elif weight <= 0.4:
                bucket = "weak"
            elif weight <= 0.6:
                bucket = "medium"
            elif weight <= 0.8:
                bucket = "strong"
            else:
                bucket = "very_strong"
            weight_buckets[bucket] += 1
        
        # Usage count distribution
        usage_buckets = defaultdict(int)
        for conn in all_connections:
            usage = conn["usage_count"]
            if usage == 0:
                bucket = "never_used"
            elif usage <= 5:
                bucket = "1-5_uses"
            elif usage <= 20:
                bucket = "6-20_uses"
            else:
                bucket = "20+_uses"
            usage_buckets[bucket] += 1
        
        return ConnectionStrengthStats(
            strongest_connections=strongest_connections,
            weakest_connections=weakest_connections,
            avg_connection_weight=round(avg_weight, 3),
            weight_distribution=dict(weight_buckets),
            connection_usage_stats=dict(usage_buckets)
        )
        
    except Exception as e:
        logger.error(f"Error getting connection strength statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get connection strength statistics")

@analytics_router.get("/cluster-analytics", response_model=ClusterAnalytics)
async def get_cluster_analytics(
    memory_graph: MemoryGraph = Depends(get_memory_graph),
    layout_engine: SpatialLayoutEngine = Depends(get_spatial_layout_engine)
):
    """Get 3D clustering analytics"""
    try:
        cluster_info = layout_engine.get_cluster_info()
        nodes = list(memory_graph.nodes.values())
        
        # Count nodes per cluster (simplified assignment)
        cluster_sizes = defaultdict(int)
        for node in nodes:
            # Simplified cluster assignment
            if node.tags:
                cluster_key = f"tag:{node.tags[0]}"
            else:
                cluster_key = "misc:uncategorized"
            cluster_sizes[cluster_key] += 1
        
        # Analyze inter vs intra cluster connections
        inter_cluster_connections = 0
        intra_cluster_connections = 0
        
        for node in nodes:
            node_cluster = None
            if node.tags:
                node_cluster = f"tag:{node.tags[0]}"
            else:
                node_cluster = "misc:uncategorized"
                
            for target_id, connection in node.connections.items():
                target_node = memory_graph.get_node(target_id)
                if target_node:
                    target_cluster = None
                    if target_node.tags:
                        target_cluster = f"tag:{target_node.tags[0]}"
                    else:
                        target_cluster = "misc:uncategorized"
                    
                    if node_cluster == target_cluster:
                        intra_cluster_connections += 1
                    else:
                        inter_cluster_connections += 1
        
        return ClusterAnalytics(
            cluster_count=len(cluster_info),
            cluster_sizes=dict(cluster_sizes),
            cluster_centers={
                cluster_id: list(info["center"]) 
                for cluster_id, info in cluster_info.items()
            },
            inter_cluster_connections=inter_cluster_connections,
            intra_cluster_connections=intra_cluster_connections
        )
        
    except Exception as e:
        logger.error(f"Error getting cluster analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cluster analytics")

@analytics_router.get("/summary")
async def get_analytics_summary(
    memory_graph: MemoryGraph = Depends(get_memory_graph),
    layout_engine: SpatialLayoutEngine = Depends(get_spatial_layout_engine)
):
    """Get a summary of all analytics data"""
    try:
        # Get basic stats
        nodes = list(memory_graph.nodes.values())
        total_nodes = len(nodes)
        total_connections = sum(len(node.connections) for node in nodes)
        
        # Get most active nodes
        most_accessed = sorted(nodes, key=lambda n: n.access_count, reverse=True)[:5]
        
        # Get cluster count
        cluster_info = layout_engine.get_cluster_info()
        
        return {
            "overview": {
                "total_memory_nodes": total_nodes,
                "total_connections": total_connections,
                "total_clusters": len(cluster_info),
                "avg_connections_per_node": round(total_connections / total_nodes, 2) if total_nodes > 0 else 0
            },
            "most_accessed_memories": [
                {
                    "concept": node.concept,
                    "access_count": node.access_count,
                    "importance_score": node.importance_score
                }
                for node in most_accessed
            ],
            "cluster_overview": {
                cluster_id: {
                    "type": info.get("type", "unknown"),
                    "name": info.get("name", "unnamed")
                }
                for cluster_id, info in cluster_info.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics summary")