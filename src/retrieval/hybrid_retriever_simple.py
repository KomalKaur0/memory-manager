"""
Simplified Hybrid Retriever for MVP
Combines embedding search with memory graph traversal
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import time

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Simplified hybrid retriever combining embedding search and graph traversal"""
    
    def __init__(self, memory_graph, embedding_search):
        """
        Initialize hybrid retriever
        
        Args:
            memory_graph: MemoryGraph instance
            embedding_search: EmbeddingSearch instance
        """
        self.memory_graph = memory_graph
        self.embedding_search = embedding_search
        
    async def search_memories(self, 
                            query: str, 
                            max_results: int = 10,
                            use_graph_expansion: bool = True) -> List[Dict[str, Any]]:
        """
        Search memories using hybrid approach
        
        Args:
            query: Search query
            max_results: Maximum number of results
            use_graph_expansion: Whether to expand results using graph connections
            
        Returns:
            List of memory results with scores
        """
        results = []
        
        # Phase 1: Embedding-based search
        embedding_results = await self.embedding_search.search_by_text(
            query, k=max_results * 2, threshold=0.3  # Lower threshold for MVP
        )
        
        # Convert embedding results to memory details
        for memory_id, similarity in embedding_results:
            node = self.memory_graph.get_node(memory_id)
            if node:
                results.append({
                    "memory_id": memory_id,
                    "node": node,
                    "embedding_score": similarity,
                    "graph_score": 0.0,
                    "combined_score": similarity,
                    "source": "embedding"
                })
        
        # Phase 2: Graph expansion (if enabled)
        if use_graph_expansion and embedding_results:
            # Get top embedding results as seeds for graph traversal
            seed_ids = [result[0] for result in embedding_results[:5]]
            
            # Find connected memories
            connected_memories = {}
            for seed_id in seed_ids:
                connections = self.memory_graph.get_connected_nodes(
                    seed_id, max_depth=2, min_weight=0.3
                )
                
                for connected_id, weight, depth in connections:
                    if connected_id not in connected_memories:
                        connected_memories[connected_id] = {
                            "weight": weight,
                            "depth": depth,
                            "seed": seed_id
                        }
                    else:
                        # Keep strongest connection
                        if weight > connected_memories[connected_id]["weight"]:
                            connected_memories[connected_id] = {
                                "weight": weight,
                                "depth": depth,
                                "seed": seed_id
                            }
            
            # Add graph results that aren't already in embedding results
            existing_ids = {r["memory_id"] for r in results}
            for memory_id, conn_info in connected_memories.items():
                if memory_id not in existing_ids:
                    node = self.memory_graph.get_node(memory_id)
                    if node:
                        # Calculate graph score with depth penalty
                        graph_score = conn_info["weight"] * (0.8 ** conn_info["depth"])
                        
                        results.append({
                            "memory_id": memory_id,
                            "node": node,
                            "embedding_score": 0.0,
                            "graph_score": graph_score,
                            "combined_score": graph_score * 0.7,  # Weight graph results lower
                            "source": "graph",
                            "connection_path": [conn_info["seed"], memory_id],
                            "depth": conn_info["depth"]
                        })
                else:
                    # Update existing result with graph information
                    for result in results:
                        if result["memory_id"] == memory_id:
                            graph_score = conn_info["weight"] * (0.8 ** conn_info["depth"])
                            result["graph_score"] = graph_score
                            # Boost combined score for memories found by both methods
                            result["combined_score"] = (
                                result["embedding_score"] * 0.6 + graph_score * 0.4
                            ) * 1.2  # Bonus for multi-source
                            result["source"] = "both"
                            break
        
        # Sort by combined score and limit results
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:max_results]
    
    async def find_similar_to_memory(self, 
                                   memory_id: str, 
                                   max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find memories similar to a specific memory
        
        Args:
            memory_id: ID of the reference memory
            max_results: Maximum results to return
            
        Returns:
            List of similar memories
        """
        # Get the memory's embedding
        stored_embedding = self.embedding_search.get_stored_embedding(memory_id)
        if not stored_embedding:
            # Fallback to content-based search
            node = self.memory_graph.get_node(memory_id)
            if node:
                content = f"{node.concept} {node.summary} {node.content}"
                return await self.search_memories(content, max_results, use_graph_expansion=True)
            return []
        
        # Find similar embeddings
        similar_results = await self.embedding_search.find_similar(
            stored_embedding, k=max_results * 2, threshold=0.6
        )
        
        # Filter out the original memory and format results
        results = []
        for sim_memory_id, similarity in similar_results:
            if sim_memory_id != memory_id:
                node = self.memory_graph.get_node(sim_memory_id)
                if node:
                    results.append({
                        "memory_id": sim_memory_id,
                        "node": node,
                        "embedding_score": similarity,
                        "graph_score": 0.0,
                        "combined_score": similarity,
                        "source": "embedding"
                    })
        
        return results[:max_results]
    
    async def get_memory_cluster(self, 
                               seed_memory_ids: List[str], 
                               cluster_size: int = 10) -> List[Dict[str, Any]]:
        """
        Get a cluster of interconnected memories around seed memories
        
        Args:
            seed_memory_ids: Starting memories for cluster expansion
            cluster_size: Target size of the cluster
            
        Returns:
            List of connected memories forming a cluster
        """
        cluster_memories = {}
        
        # Start with seed memories
        for seed_id in seed_memory_ids:
            node = self.memory_graph.get_node(seed_id)
            if node:
                cluster_memories[seed_id] = {
                    "node": node,
                    "embedding_score": 1.0,
                    "graph_score": 1.0,
                    "combined_score": 1.0,
                    "source": "seed",
                    "depth": 0
                }
        
        # Expand cluster by following connections
        current_depth = 0
        max_depth = 3
        
        while len(cluster_memories) < cluster_size and current_depth < max_depth:
            current_level_ids = [
                mid for mid, info in cluster_memories.items() 
                if info["depth"] == current_depth
            ]
            
            for memory_id in current_level_ids:
                connections = self.memory_graph.get_connected_nodes(
                    memory_id, max_depth=1, min_weight=0.4
                )
                
                for connected_id, weight, _ in connections:
                    if connected_id not in cluster_memories:
                        node = self.memory_graph.get_node(connected_id)
                        if node:
                            # Score based on connection strength and depth
                            score = weight * (0.8 ** (current_depth + 1))
                            cluster_memories[connected_id] = {
                                "node": node,
                                "embedding_score": 0.0,
                                "graph_score": weight,
                                "combined_score": score,
                                "source": "graph",
                                "depth": current_depth + 1,
                                "connected_from": memory_id
                            }
                            
                            if len(cluster_memories) >= cluster_size:
                                break
                
                if len(cluster_memories) >= cluster_size:
                    break
            
            current_depth += 1
        
        # Convert to list and sort by combined score
        results = list(cluster_memories.values())
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Add memory_id to each result
        for i, (memory_id, result) in enumerate(cluster_memories.items()):
            if i < len(results):
                results[i]["memory_id"] = memory_id
        
        return results
    
    def explain_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Generate explanation for search results
        
        Args:
            results: Search results to explain
            query: Original query
            
        Returns:
            Human-readable explanation
        """
        if not results:
            return f"No memories found for query: '{query}'"
        
        explanation_parts = [
            f"Found {len(results)} memories for query: '{query}'"
        ]
        
        # Count by source
        embedding_count = sum(1 for r in results if r["source"] in ["embedding", "both"])
        graph_count = sum(1 for r in results if r["source"] in ["graph", "both"])
        both_count = sum(1 for r in results if r["source"] == "both")
        
        if embedding_count > 0:
            explanation_parts.append(f"- {embedding_count} found through semantic similarity")
        if graph_count > 0:
            explanation_parts.append(f"- {graph_count} found through memory connections")
        if both_count > 0:
            explanation_parts.append(f"- {both_count} found through both methods (high confidence)")
        
        # Top result details
        if results:
            top = results[0]
            explanation_parts.append(
                f"Top result: {top['node'].concept} "
                f"(score: {top['combined_score']:.3f}, source: {top['source']})"
            )
        
        return "\n".join(explanation_parts)