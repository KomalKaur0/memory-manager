"""
Hybrid Retriever Module - Orchestrator for AI Memory System

This module combines embedding-based search with graph traversal to create
intelligent, human-like memory retrieval. It orchestrates the two-phase
retrieval process and provides unified, ranked results with explanations.
"""

from typing import List, Dict, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import time
import hashlib
import json


class RetrievalMode(Enum):
    """Different retrieval modes for various use cases"""
    STANDARD = "standard"  # Balanced embedding + graph
    EMBEDDING_ONLY = "embedding_only"  # Skip graph traversal
    GRAPH_ONLY = "graph_only"  # Skip embedding search
    DEEP_EXPLORATION = "deep_exploration"  # Extended graph traversal
    FAST_LOOKUP = "fast_lookup"  # Quick results, minimal processing
    CONTEXTUAL = "contextual"  # Context-aware retrieval


@dataclass
class RetrievalContext:
    """Context information for memory retrieval"""
    query: str
    user_id: Optional[str] = None
    conversation_history: List[str] = field(default_factory=list)
    current_memories: List[str] = field(default_factory=list)
    time_context: Optional[datetime] = None
    location_context: Optional[str] = None
    tags_filter: List[str] = field(default_factory=list)
    semantic_filters: Dict[str, Any] = field(default_factory=dict)
    priority_memories: List[str] = field(default_factory=list)
    exclude_memories: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Complete result from hybrid retrieval operation"""
    memories: List[Dict[str, Any]]
    query: str
    total_candidates_found: int
    embedding_results_count: int
    graph_results_count: int
    retrieval_time: float
    explanation: str
    confidence_score: float
    seed_memories: List[str] = field(default_factory=list)
    retrieval_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryCandidate:
    """Internal representation of memory candidates during retrieval"""
    memory_id: str
    content: str
    embedding_score: float = 0.0
    connection_score: float = 0.0
    combined_score: float = 0.0
    source: str = "unknown"  # "embedding", "graph", "both"
    path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovery_depth: int = 0
    confidence: float = 0.0


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval operations"""
    # Phase weights
    embedding_weight: float = 0.6
    graph_weight: float = 0.4
    
    # Result limits
    max_total_results: int = 10
    max_embedding_candidates: int = 20
    max_graph_candidates: int = 30
    
    # Quality thresholds
    min_embedding_similarity: float = 0.7
    min_connection_strength: float = 0.3
    min_combined_score: float = 0.5
    
    # Graph traversal settings
    max_graph_depth: int = 3
    graph_exploration_width: int = 5
    
    # Scoring bonuses
    multi_source_bonus: float = 1.2  # Boost for memories found by both methods
    recency_bonus_days: int = 30
    recency_bonus_factor: float = 1.1
    path_diversity_bonus: float = 0.1
    
    # Context awareness
    use_conversation_context: bool = True
    context_influence_weight: float = 0.3
    temporal_context_window: int = 7  # days
    conversation_history_length: int = 3  # Number of recent messages to consider
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    parallel_processing: bool = True
    timeout_seconds: float = 30.0


@dataclass
class RetrievalStats:
    """Statistics tracking for retrieval operations"""
    total_retrievals: int = 0
    avg_retrieval_time: float = 0.0
    avg_results_returned: int = 0
    embedding_hit_rate: float = 0.0
    graph_discovery_rate: float = 0.0
    cache_hit_rate: float = 0.0
    query_expansion_rate: float = 0.0
    user_satisfaction_score: float = 0.0


class ResultFilter(ABC):
    """Abstract base class for result filtering"""
    
    @abstractmethod
    def filter(self, 
              candidates: List[MemoryCandidate], 
              context: RetrievalContext) -> List[MemoryCandidate]:
        """Filter memory candidates based on criteria"""
        pass


class HybridRetriever:
    """
    Hybrid retrieval engine combining embedding search and graph traversal.
    
    This is the main orchestrator of the memory retrieval system, combining
    semantic similarity search with graph-based associative exploration to
    provide intelligent, human-like memory recall.
    
    Key responsibilities:
    - Coordinate two-phase retrieval (embedding + graph)
    - Combine and rank results from multiple sources
    - Apply context-aware filtering and scoring
    - Generate explanations for retrieval decisions
    - Learn from usage patterns to improve retrieval
    - Manage performance and caching
    """
    
    def __init__(self,
                 embedding_search,
                 graph_traversal,
                 memory_graph,
                 config: RetrievalConfig,
                 logger=None):
        """
        Initialize the hybrid retriever.
        
        Args:
            embedding_search: EmbeddingSearch instance
            graph_traversal: GraphTraversal instance  
            memory_graph: Main memory graph instance
            config: Retrieval configuration
            logger: Logger for debugging and monitoring
        """
        self.embedding_search = embedding_search
        self.graph_traversal = graph_traversal
        self.memory_graph = memory_graph
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Result filtering and scoring
        self.result_filters: List[ResultFilter] = []
        self.custom_scorers = {}
        
        # Performance tracking
        self.stats = RetrievalStats()
        self.retrieval_history = []
        
        # Caching
        self.result_cache = {}
        self.context_cache = {}
        
        # Learning components
        self.query_patterns = {}
        self.successful_retrievals = []
        self.user_feedback = {}
    
    def retrieve(self,
                query: str,
                context: Optional[RetrievalContext] = None,
                mode: RetrievalMode = RetrievalMode.STANDARD,
                max_results: Optional[int] = None) -> RetrievalResult:
        """
        Main retrieval method combining embedding search and graph traversal.
        
        This is the primary interface for memory retrieval, orchestrating
        the two-phase process and returning ranked, explained results.
        
        Args:
            query: Search query text
            context: Additional context for retrieval
            mode: Retrieval mode to use
            max_results: Override default max results
            
        Returns:
            RetrievalResult with ranked memories and explanations
            
        Raises:
            RetrievalError: If retrieval operation fails
        """
        start_time = time.time()
        
        if context is None:
            context = RetrievalContext(query=query)
        
        if max_results:
            self.config.max_total_results = max_results
            
        # Check cache first
        cache_key = self._generate_cache_key(query, context, mode)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Phase 1: Embedding-based discovery
            embedding_candidates = []
            if mode in [RetrievalMode.STANDARD, RetrievalMode.EMBEDDING_ONLY, 
                       RetrievalMode.CONTEXTUAL]:
                embedding_candidates = self._phase_one_embedding_discovery(query, context)
            
            # Phase 2: Graph-based exploration  
            graph_candidates = []
            if mode in [RetrievalMode.STANDARD, RetrievalMode.GRAPH_ONLY,
                       RetrievalMode.DEEP_EXPLORATION, RetrievalMode.CONTEXTUAL]:
                graph_candidates = self._phase_two_graph_exploration(embedding_candidates, context)
            
            # Combine and rank results
            combined_candidates = self._combine_and_rank_results(
                embedding_candidates, graph_candidates, context)
            
            # Apply context weighting
            weighted_candidates = self._apply_context_weighting(combined_candidates, context)
            
            # Filter and deduplicate
            final_candidates = self._filter_and_deduplicate(weighted_candidates, context)
            
            # Limit results
            final_candidates = final_candidates[:self.config.max_total_results]
            
            # Generate explanation
            retrieval_stats = {
                "embedding_count": len(embedding_candidates),
                "graph_count": len(graph_candidates),
                "combined_count": len(combined_candidates),
                "final_count": len(final_candidates)
            }
            explanation = self._generate_explanation(final_candidates, context, retrieval_stats)
            
            # Calculate confidence
            confidence = self._calculate_overall_confidence(final_candidates)
            
            # Build result
            result = RetrievalResult(
                memories=[self._candidate_to_dict(c) for c in final_candidates],
                query=query,
                total_candidates_found=len(combined_candidates),
                embedding_results_count=len(embedding_candidates),
                graph_results_count=len(graph_candidates),
                retrieval_time=time.time() - start_time,
                explanation=explanation,
                confidence_score=confidence,
                seed_memories=[c.memory_id for c in embedding_candidates[:5]],
                metadata=retrieval_stats
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Record metrics
            self._record_retrieval_metrics(start_time, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(f"Retrieval operation failed: {e}")
    
    def retrieve_contextual(self,
                           query: str,
                           conversation_history: List[str],
                           current_memory_ids: List[str] = None,
                           **kwargs) -> RetrievalResult:
        """
        Context-aware retrieval using conversation history and current state.
        
        Adjusts retrieval based on recent conversation context and
        currently active memories to provide more relevant results.
        
        Args:
            query: Search query
            conversation_history: Recent conversation messages
            current_memory_ids: Currently active/relevant memories
            **kwargs: Additional context parameters
            
        Returns:
            Context-aware retrieval results
        """
        context = RetrievalContext(
            query=query,
            conversation_history=conversation_history,
            current_memories=current_memory_ids or [],
            **kwargs
        )
        return self.retrieve(query, context, RetrievalMode.CONTEXTUAL)
    
    def retrieve_similar_to_memory(self,
                                  memory_id: str,
                                  similarity_threshold: float = 0.8,
                                  include_graph_connections: bool = True) -> RetrievalResult:
        """
        Find memories similar to a specific existing memory.
        
        Combines embedding similarity with graph connections to find
        memories related to the given memory.
        
        Args:
            memory_id: Reference memory ID
            similarity_threshold: Minimum similarity score
            include_graph_connections: Whether to include graph-connected memories
            
        Returns:
            Memories similar to the reference memory
        """
        # Get memory content for similarity search
        memory_content = self.memory_graph.get_memory_content(memory_id)
        
        context = RetrievalContext(
            query=memory_content,
            current_memories=[memory_id],
            semantic_filters={"min_similarity": similarity_threshold}
        )
        
        mode = RetrievalMode.STANDARD if include_graph_connections else RetrievalMode.EMBEDDING_ONLY
        return self.retrieve(memory_content, context, mode)
    
    def retrieve_by_time_range(self,
                              query: str,
                              start_time: datetime,
                              end_time: datetime,
                              temporal_weight: float = 0.3) -> RetrievalResult:
        """
        Retrieve memories within a specific time range.
        
        Combines semantic search with temporal filtering,
        useful for "What did I think about X last week?" queries.
        
        Args:
            query: Search query
            start_time: Start of time range
            end_time: End of time range
            temporal_weight: How much to weight temporal relevance
            
        Returns:
            Time-filtered retrieval results
        """
        context = RetrievalContext(
            query=query,
            time_context=start_time,
            semantic_filters={
                "start_time": start_time,
                "end_time": end_time,
                "temporal_weight": temporal_weight
            }
        )
        
        return self.retrieve(query, context)
    
    def retrieve_connected_cluster(self,
                                  seed_memory_ids: List[str],
                                  cluster_size: int = 10) -> RetrievalResult:
        """
        Retrieve a connected cluster of memories around seed memories.
        
        Finds a coherent group of interconnected memories,
        useful for understanding a complete topic or context.
        
        Args:
            seed_memory_ids: Starting memories for cluster expansion
            cluster_size: Target size of memory cluster
            
        Returns:
            Connected cluster of related memories
        """
        # Use graph traversal to find connected memories
        connected_memories = []
        for seed_id in seed_memory_ids:
            results = self.graph_traversal.explore_from_seeds([seed_id], max_results=cluster_size)
            connected_memories.extend([r.memory_id for r in results])
        
        # Remove duplicates and original seeds
        unique_memories = list(set(connected_memories) - set(seed_memory_ids))
        
        context = RetrievalContext(
            query="connected cluster",
            current_memories=seed_memory_ids,
            priority_memories=unique_memories[:cluster_size]
        )
        
        return self.retrieve("connected cluster", context, RetrievalMode.GRAPH_ONLY)
    
    def explain_retrieval(self,
                         result: RetrievalResult,
                         include_technical_details: bool = False) -> str:
        """
        Generate detailed explanation of why memories were retrieved.
        
        Provides human-readable explanation of the retrieval process,
        including which phase found each memory and why.
        
        Args:
            result: RetrievalResult to explain
            include_technical_details: Whether to include technical scoring
            
        Returns:
            Human-readable explanation string
        """
        explanation_parts = [
            f"Retrieved {len(result.memories)} memories for query: '{result.query}'",
            f"Retrieval took {result.retrieval_time:.3f} seconds",
            f"Found {result.embedding_results_count} candidates via embedding search",
            f"Found {result.graph_results_count} candidates via graph traversal",
            f"Overall confidence: {result.confidence_score:.2f}"
        ]
        
        if include_technical_details:
            explanation_parts.extend([
                f"Embedding weight: {self.config.embedding_weight}",
                f"Graph weight: {self.config.graph_weight}",
                f"Total candidates evaluated: {result.total_candidates_found}"
            ])
        
        return "\n".join(explanation_parts)
    
    def _phase_one_embedding_discovery(self,
                                     query: str,
                                     context: RetrievalContext) -> List[MemoryCandidate]:
        """
        Phase 1: Embedding-based semantic discovery.
        
        Uses vector similarity to find semantically related memories
        as starting points for graph exploration.
        
        Args:
            query: Search query
            context: Retrieval context
            
        Returns:
            List of memory candidates from embedding search
        """
        try:
            search_results = self.embedding_search.search(
                query=query,
                max_results=self.config.max_embedding_candidates,
                min_similarity=self.config.min_embedding_similarity
            )
            
            candidates = []
            for result in search_results:
                candidate = MemoryCandidate(
                    memory_id=result.memory_id,
                    content=self.memory_graph.get_memory_content(result.memory_id),
                    embedding_score=result.similarity_score,
                    source="embedding",
                    metadata=result.metadata,
                    confidence=result.similarity_score
                )
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            self.logger.warning(f"Embedding search failed: {e}")
            return []
    
    def _phase_two_graph_exploration(self,
                                   seed_candidates: List[MemoryCandidate],
                                   context: RetrievalContext) -> List[MemoryCandidate]:
        """
        Phase 2: Graph-based associative exploration.
        
        Explores connections from embedding results to find
        associated memories through learned relationships.
        
        Args:
            seed_candidates: Starting points from embedding search
            context: Retrieval context
            
        Returns:
            List of memory candidates from graph traversal
        """
        try:
            seed_ids = [c.memory_id for c in seed_candidates[:5]]  # Use top 5 as seeds
            
            traversal_results = self.graph_traversal.explore_from_seeds(
                seed_ids=seed_ids,
                max_results=self.config.max_graph_candidates,
                max_depth=self.config.max_graph_depth
            )
            
            candidates = []
            for result in traversal_results:
                candidate = MemoryCandidate(
                    memory_id=result.memory_id,
                    content=self.memory_graph.get_memory_content(result.memory_id),
                    connection_score=result.connection_strength,
                    source="graph",
                    path=result.path,
                    discovery_depth=result.depth,
                    metadata=result.metadata,
                    confidence=result.traversal_score
                )
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            self.logger.warning(f"Graph traversal failed: {e}")
            return []
    
    def _combine_and_rank_results(self,
                                embedding_candidates: List[MemoryCandidate],
                                graph_candidates: List[MemoryCandidate],
                                context: RetrievalContext) -> List[MemoryCandidate]:
        """
        Combine results from both phases and rank by unified score.
        
        Merges candidates from embedding and graph phases,
        deduplicates, and ranks by combined scoring function.
        
        Args:
            embedding_candidates: Results from embedding search
            graph_candidates: Results from graph traversal
            context: Retrieval context
            
        Returns:
            Unified, ranked list of memory candidates
        """
        # Combine all candidates
        all_candidates = {}
        
        # Add embedding candidates
        for candidate in embedding_candidates:
            all_candidates[candidate.memory_id] = candidate
        
        # Merge graph candidates (may overlap with embedding)
        for candidate in graph_candidates:
            if candidate.memory_id in all_candidates:
                # Merge scores from both sources
                existing = all_candidates[candidate.memory_id]
                existing.connection_score = candidate.connection_score
                existing.source = "both"
                existing.path = candidate.path
                existing.discovery_depth = candidate.discovery_depth
            else:
                all_candidates[candidate.memory_id] = candidate
        
        # Calculate combined scores
        candidates_list = list(all_candidates.values())
        for candidate in candidates_list:
            candidate.combined_score = self._calculate_combined_score(candidate, context)
        
        # Sort by combined score
        candidates_list.sort(key=lambda c: c.combined_score, reverse=True)
        
        return candidates_list
    
    def _calculate_combined_score(self,
                                candidate: MemoryCandidate,
                                context: RetrievalContext) -> float:
        """
        Calculate unified score combining multiple signals.
        
        Combines embedding similarity, graph connection strength,
        recency, context relevance, and other factors.
        
        Args:
            candidate: Memory candidate to score
            context: Retrieval context
            
        Returns:
            Combined relevance score
        """
        # Base scores
        embedding_score = candidate.embedding_score * self.config.embedding_weight
        graph_score = candidate.connection_score * self.config.graph_weight
        
        # Multi-source bonus
        bonus = 1.0
        if candidate.source == "both":
            bonus = self.config.multi_source_bonus
        
        # Path diversity bonus
        path_bonus = 1.0
        if candidate.path and len(candidate.path) > 2:
            path_bonus = 1.0 + self.config.path_diversity_bonus
        
        # Combine all factors
        combined_score = (embedding_score + graph_score) * bonus * path_bonus
        
        return min(combined_score, 1.0)  # Cap at 1.0
    
    def _apply_context_weighting(self,
                               candidates: List[MemoryCandidate],
                               context: RetrievalContext) -> List[MemoryCandidate]:
        """
        Apply context-based score adjustments.
        
        Boosts scores for memories that align with conversation
        context, user preferences, temporal relevance, etc.
        
        Args:
            candidates: Memory candidates to adjust
            context: Retrieval context
            
        Returns:
            Context-weighted candidates
        """
        if not self.config.use_conversation_context:
            return candidates
        
        # Apply conversation context boost
        for candidate in candidates:
            context_boost = 1.0
            
            # Check if memory relates to conversation history
            if context.conversation_history:
                # Simple keyword overlap check
                memory_text = candidate.content.lower()
                for message in context.conversation_history[-self.config.conversation_history_length:]:  # Use configurable length
                    message_words = set(message.lower().split())
                    memory_words = set(memory_text.split())
                    overlap = len(message_words.intersection(memory_words))
                    if overlap > 2:
                        context_boost += self.config.context_influence_weight
            
            # Priority memory boost
            if candidate.memory_id in context.priority_memories:
                context_boost += 0.2
            
            candidate.combined_score *= context_boost
        
        return candidates
    
    def _filter_and_deduplicate(self,
                               candidates: List[MemoryCandidate],
                               context: RetrievalContext) -> List[MemoryCandidate]:
        """
        Apply filters and remove duplicates.
        
        Removes low-quality results, applies custom filters,
        and deduplicates memories found by multiple methods.
        
        Args:
            candidates: Memory candidates to filter
            context: Retrieval context
            
        Returns:
            Filtered and deduplicated candidates
        """
        # Apply quality threshold
        filtered = [c for c in candidates if c.combined_score >= self.config.min_combined_score]
        
        # Apply exclude list
        filtered = [c for c in filtered if c.memory_id not in context.exclude_memories]
        
        # Apply custom filters
        for filter_instance in self.result_filters:
            filtered = filter_instance.filter(filtered, context)
        
        # Remove duplicates (shouldn't happen but safety check)
        seen_ids = set()
        deduplicated = []
        for candidate in filtered:
            if candidate.memory_id not in seen_ids:
                deduplicated.append(candidate)
                seen_ids.add(candidate.memory_id)
        
        return deduplicated
    
    def _generate_explanation(self,
                            final_results: List[MemoryCandidate],
                            context: RetrievalContext,
                            retrieval_stats: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of retrieval results.
        
        Creates explanation covering why each memory was found,
        which phase discovered it, and connection paths.
        
        Args:
            final_results: Final ranked results
            context: Retrieval context
            retrieval_stats: Statistics from retrieval process
            
        Returns:
            Explanation string
        """
        if not final_results:
            return "No relevant memories found for this query."
        
        explanation_parts = [
            f"Found {len(final_results)} relevant memories:",
        ]
        
        # Count by source
        embedding_count = sum(1 for r in final_results if r.source in ["embedding", "both"])
        graph_count = sum(1 for r in final_results if r.source in ["graph", "both"])
        both_count = sum(1 for r in final_results if r.source == "both")
        
        if embedding_count > 0:
            explanation_parts.append(f"- {embedding_count} found through semantic similarity")
        if graph_count > 0:
            explanation_parts.append(f"- {graph_count} found through memory connections")
        if both_count > 0:
            explanation_parts.append(f"- {both_count} found through both methods (high confidence)")
        
        # Add top result details
        if final_results:
            top_result = final_results[0]
            explanation_parts.append(
                f"Top result: {top_result.memory_id} (score: {top_result.combined_score:.3f}, "
                f"source: {top_result.source})"
            )
        
        return "\n".join(explanation_parts)
    
    def _calculate_overall_confidence(self, candidates: List[MemoryCandidate]) -> float:
        """Calculate overall confidence in the retrieval results."""
        if not candidates:
            return 0.0
        
        # Average confidence of top 3 results
        top_candidates = candidates[:3]
        avg_confidence = sum(c.combined_score for c in top_candidates) / len(top_candidates)
        return min(avg_confidence, 1.0)
    
    def _candidate_to_dict(self, candidate: MemoryCandidate) -> Dict[str, Any]:
        """Convert MemoryCandidate to dictionary for result."""
        return {
            "memory_id": candidate.memory_id,
            "content": candidate.content,
            "score": candidate.combined_score,
            "source": candidate.source,
            "embedding_score": candidate.embedding_score,
            "connection_score": candidate.connection_score,
            "path": candidate.path,
            "depth": candidate.discovery_depth,
            "confidence": candidate.confidence,
            "metadata": candidate.metadata
        }
    
    def add_result_filter(self, filter_instance: ResultFilter) -> None:
        """Add a custom result filter to the retrieval pipeline."""
        self.result_filters.append(filter_instance)
    
    def remove_result_filter(self, filter_class: type) -> bool:
        """Remove a result filter from the pipeline."""
        original_length = len(self.result_filters)
        self.result_filters = [f for f in self.result_filters if not isinstance(f, filter_class)]
        return len(self.result_filters) < original_length
    
    def add_custom_scorer(self, 
                         name: str, 
                         scorer_func: callable,
                         weight: float = 1.0) -> None:
        """Add a custom scoring function to the retrieval pipeline."""
        self.custom_scorers[name] = {"func": scorer_func, "weight": weight}
    
    def get_retrieval_statistics(self) -> RetrievalStats:
        """Get comprehensive retrieval statistics."""
        return self.stats
    
    def clear_caches(self, max_age_seconds: Optional[int] = None) -> int:
        """Clear retrieval caches."""
        if max_age_seconds is None:
            cleared = len(self.result_cache)
            self.result_cache.clear()
            self.context_cache.clear()
            return cleared
        
        # Clear old entries
        current_time = time.time()
        cleared = 0
        
        old_keys = [k for k, v in self.result_cache.items() 
                   if current_time - v.get("timestamp", 0) > max_age_seconds]
        for key in old_keys:
            del self.result_cache[key]
            cleared += 1
        
        return cleared
    
    def _check_cache(self, cache_key: str) -> Optional[RetrievalResult]:
        """Check if results are cached for this query."""
        if not self.config.enable_caching:
            return None
        
        if cache_key in self.result_cache:
            cache_entry = self.result_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.config.cache_ttl_seconds:
                return cache_entry["result"]
            else:
                del self.result_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: RetrievalResult) -> None:
        """Cache retrieval results for future use."""
        if self.config.enable_caching:
            self.result_cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
    
    def _generate_cache_key(self, 
                           query: str, 
                           context: RetrievalContext,
                           mode: RetrievalMode) -> str:
        """Generate cache key for retrieval parameters."""
        key_components = [
            query,
            str(mode.value),
            str(context.user_id or ""),
            str(len(context.conversation_history)),
            str(sorted(context.current_memories)),
            str(sorted(context.tags_filter))
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _record_retrieval_metrics(self,
                                start_time: float,
                                result: RetrievalResult) -> None:
        """Record metrics for this retrieval operation."""
        self.stats.total_retrievals += 1
        self.stats.avg_retrieval_time = (
            (self.stats.avg_retrieval_time * (self.stats.total_retrievals - 1) + result.retrieval_time) /
            self.stats.total_retrievals
        )
        self.stats.avg_results_returned = (
            (self.stats.avg_results_returned * (self.stats.total_retrievals - 1) + len(result.memories)) /
            self.stats.total_retrievals
        )


# Exception classes for hybrid retrieval
class RetrievalError(Exception):
    """Base exception for retrieval operations"""
    pass

class RetrievalTimeoutError(RetrievalError):
    """Retrieval operation exceeded time limit"""
    pass

class InsufficientResultsError(RetrievalError):
    """Not enough quality results found"""
    pass

class ContextError(RetrievalError):
    """Invalid or insufficient context provided"""
    pass


# Built-in result filters
class RelevanceFilter(ResultFilter):
    """Filter memories below relevance threshold"""
    
    def __init__(self, min_score: float = 0.5):
        self.min_score = min_score
    
    def filter(self, 
              candidates: List[MemoryCandidate], 
              context: RetrievalContext) -> List[MemoryCandidate]:
        """Filter by minimum relevance score"""
        return [c for c in candidates if c.combined_score >= self.min_score]


class RecencyFilter(ResultFilter):
    """Filter memories based on recency requirements"""
    
    def __init__(self, max_age_days: Optional[int] = None):
        self.max_age_days = max_age_days
    
    def filter(self,
              candidates: List[MemoryCandidate],
              context: RetrievalContext) -> List[MemoryCandidate]:
        """Filter by memory age"""
        if self.max_age_days is None:
            return candidates
        
        cutoff_time = datetime.now() - timedelta(days=self.max_age_days)
        filtered = []
        
        for candidate in candidates:
            # Check if memory has timestamp in metadata
            created_time = candidate.metadata.get("created_at")
            if created_time:
                if isinstance(created_time, str):
                    created_time = datetime.fromisoformat(created_time)
                if created_time >= cutoff_time:
                    filtered.append(candidate)
            else:
                # Include if no timestamp available
                filtered.append(candidate)
        
        return filtered


class TagFilter(ResultFilter):
    """Filter memories based on tag requirements"""
    
    def __init__(self, required_tags: List[str] = None, excluded_tags: List[str] = None):
        self.required_tags = required_tags or []
        self.excluded_tags = excluded_tags or []
    
    def filter(self,
              candidates: List[MemoryCandidate],
              context: RetrievalContext) -> List[MemoryCandidate]:
        """Filter by tag requirements"""
        filtered = []
        
        for candidate in candidates:
            memory_tags = candidate.metadata.get("tags", [])
            
            # Check required tags
            if self.required_tags:
                if not all(tag in memory_tags for tag in self.required_tags):
                    continue
            
            # Check excluded tags
            if self.excluded_tags:
                if any(tag in memory_tags for tag in self.excluded_tags):
                    continue
            
            filtered.append(candidate)
        
        return filtered