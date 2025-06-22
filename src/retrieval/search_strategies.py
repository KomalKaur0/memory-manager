"""
Search Strategies Module - Advanced Search Methods for AI Memory System

This module provides specialized search strategies and fallback methods
for different scenarios when standard embedding + graph retrieval
may not be sufficient or optimal.
"""

from typing import List, Dict, Optional, Tuple, Any, Union, Set, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import re
import logging
import time
import json
from collections import defaultdict


class SearchStrategy(Enum):
    """Different search strategy types"""
    HYPOTHETICAL_DOCUMENT = "hypothetical_document"
    TEMPORAL_SEARCH = "temporal_search"
    CONCEPT_EXPANSION = "concept_expansion" 
    SIMILARITY_CLUSTERING = "similarity_clustering"
    KEYWORD_FALLBACK = "keyword_fallback"
    FUZZY_MATCHING = "fuzzy_matching"
    SEMANTIC_ROLES = "semantic_roles"
    CONVERSATIONAL_CONTEXT = "conversational_context"
    MULTI_MODAL = "multi_modal"


@dataclass
class StrategyResult:
    """Result from a specific search strategy"""
    memory_ids: List[str]
    scores: List[float]
    strategy_used: SearchStrategy
    confidence: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class QueryAnalysis:
    """Analysis of a search query to determine best strategy"""
    query: str
    query_type: str  # "factual", "temporal", "associative", "exploratory"
    entities: List[str]
    concepts: List[str]
    temporal_indicators: List[str]
    question_type: Optional[str]  # "what", "when", "where", "how", "why"
    complexity_score: float
    ambiguity_score: float
    suggested_strategies: List[SearchStrategy]


@dataclass
class SearchContext:
    """Extended context for search strategy execution"""
    original_query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: List[str] = field(default_factory=list)
    recent_memories: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    search_history: List[str] = field(default_factory=list)
    current_time: datetime = field(default_factory=datetime.now)
    available_strategies: Set[SearchStrategy] = field(default_factory=set)


class BaseSearchStrategy(ABC):
    """Abstract base class for search strategies"""
    
    @abstractmethod
    def can_handle(self, query: str, context: SearchContext) -> float:
        """
        Return confidence score (0-1) for handling this query
        
        Args:
            query: Search query
            context: Search context
            
        Returns:
            Confidence score for strategy applicability
        """
        pass
    
    @abstractmethod
    def execute(self, 
               query: str, 
               context: SearchContext,
               max_results: int = 10) -> StrategyResult:
        """
        Execute the search strategy
        
        Args:
            query: Search query
            context: Search context  
            max_results: Maximum results to return
            
        Returns:
            StrategyResult with found memories
        """
        pass


class SearchStrategies:
    """
    Advanced search strategies and fallback methods.
    
    This class provides specialized search approaches for scenarios
    where standard embedding + graph retrieval may not be optimal,
    such as abstract queries, temporal searches, or when dealing
    with limited or noisy data.
    
    Key responsibilities:
    - Analyze queries to determine best strategy
    - Generate hypothetical documents for better matching
    - Handle temporal and concept-based searches
    - Provide fallback strategies for edge cases
    - Expand queries intelligently
    - Cluster and organize search results
    """
    
    def __init__(self, 
                 hybrid_retriever,
                 memory_graph,
                 nlp_processor=None,
                 logger=None):
        """
        Initialize search strategies system.
        
        Args:
            hybrid_retriever: Main HybridRetriever instance
            memory_graph: Memory graph for direct access
            nlp_processor: NLP pipeline for query analysis
            logger: Logger for debugging
        """
        self.retriever = hybrid_retriever
        self.memory_graph = memory_graph
        self.nlp_processor = nlp_processor
        self.logger = logger or logging.getLogger(__name__)
        
        # Available strategies
        self.strategies: Dict[SearchStrategy, BaseSearchStrategy] = {}
        self._register_default_strategies()
        
        # Strategy performance tracking
        self.strategy_stats = defaultdict(lambda: {
            "total_uses": 0,
            "avg_confidence": 0.0,
            "avg_execution_time": 0.0,
            "success_rate": 0.0
        })
        self.query_patterns = {}
        
        # LLM integration for hypothetical generation
        self.llm_client = None  # For generating hypothetical documents
        
        # Temporal parsing patterns
        self.temporal_patterns = [
            (r'last (\d+) (day|week|month|year)s?', lambda m: self._parse_relative_time(m)),
            (r'(yesterday|today|tomorrow)', lambda m: self._parse_relative_day(m)),
            (r'(\d{4})', lambda m: self._parse_year(m)),
            (r'(january|february|march|april|may|june|july|august|september|october|november|december)', 
             lambda m: self._parse_month(m)),
        ]
    
    def search_with_strategy_selection(self,
                                     query: str,
                                     context: SearchContext,
                                     max_results: int = 10,
                                     fallback_enabled: bool = True) -> StrategyResult:
        """
        Automatically select and execute the best search strategy.
        
        Analyzes the query to determine the most appropriate search
        strategy and executes it, with fallback to other strategies
        if the first choice doesn't yield good results.
        
        Args:
            query: Search query
            context: Search context
            max_results: Maximum results to return
            fallback_enabled: Whether to try fallback strategies
            
        Returns:
            StrategyResult with best found memories
        """
        start_time = time.time()
        
        try:
            # Analyze query to get strategy recommendations
            analysis = self.analyze_query(query, context)
            
            # Try strategies in order of confidence
            best_result = None
            tried_strategies = []
            
            for strategy in analysis.suggested_strategies:
                if strategy not in self.strategies:
                    continue
                    
                try:
                    strategy_instance = self.strategies[strategy]
                    confidence = strategy_instance.can_handle(query, context)
                    
                    if confidence > 0.3:  # Minimum confidence threshold
                        result = strategy_instance.execute(query, context, max_results)
                        tried_strategies.append((strategy, result))
                        
                        # Keep best result
                        if (best_result is None or 
                            result.confidence > best_result.confidence or
                            len(result.memory_ids) > len(best_result.memory_ids)):
                            best_result = result
                        
                        # If we got good results, stop trying
                        if result.confidence > 0.7 and len(result.memory_ids) >= max_results // 2:
                            break
                            
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy} failed: {e}")
                    continue
            
            # Fallback to keyword search if no good results
            if (fallback_enabled and 
                (best_result is None or best_result.confidence < 0.4)):
                
                if SearchStrategy.KEYWORD_FALLBACK in self.strategies:
                    try:
                        fallback_strategy = self.strategies[SearchStrategy.KEYWORD_FALLBACK]
                        fallback_result = fallback_strategy.execute(query, context, max_results)
                        
                        if (best_result is None or 
                            len(fallback_result.memory_ids) > len(best_result.memory_ids)):
                            best_result = fallback_result
                            
                    except Exception as e:
                        self.logger.warning(f"Fallback strategy failed: {e}")
            
            # Return best result or empty result
            if best_result is None:
                best_result = StrategyResult(
                    memory_ids=[],
                    scores=[],
                    strategy_used=SearchStrategy.KEYWORD_FALLBACK,
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    explanation="No strategies yielded results"
                )
            
            # Update statistics
            self._update_strategy_stats(best_result.strategy_used, best_result)
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {e}")
            return StrategyResult(
                memory_ids=[],
                scores=[],
                strategy_used=SearchStrategy.KEYWORD_FALLBACK,
                confidence=0.0,
                execution_time=time.time() - start_time,
                explanation=f"Error in strategy selection: {e}"
            )
    
    def analyze_query(self, 
                     query: str, 
                     context: SearchContext) -> QueryAnalysis:
        """
        Analyze query to understand intent and suggest strategies.
        
        Uses NLP to extract entities, concepts, temporal indicators,
        and other features to determine the best search approach.
        
        Args:
            query: Search query to analyze
            context: Additional context
            
        Returns:
            QueryAnalysis with insights and strategy recommendations
        """
        # Extract basic features
        entities = self._extract_entities(query)
        concepts = self._extract_concepts(query)
        temporal_indicators = self._extract_temporal_expressions(query)
        question_type = self._detect_question_type(query)
        
        # Determine query type
        query_type = self._classify_query_type(query, temporal_indicators, question_type)
        
        # Calculate complexity and ambiguity
        complexity_score = self._calculate_complexity_score(query, entities, concepts)
        ambiguity_score = self._calculate_ambiguity_score(query)
        
        # Suggest strategies based on analysis
        suggested_strategies = self._suggest_strategies(
            query_type, temporal_indicators, question_type, 
            complexity_score, ambiguity_score, context
        )
        
        return QueryAnalysis(
            query=query,
            query_type=query_type,
            entities=entities,
            concepts=concepts,
            temporal_indicators=temporal_indicators,
            question_type=question_type,
            complexity_score=complexity_score,
            ambiguity_score=ambiguity_score,
            suggested_strategies=suggested_strategies
        )
    
    def hypothetical_document_search(self,
                                   query: str,
                                   num_hypotheticals: int = 3,
                                   creativity_level: float = 0.7,
                                   max_results: int = 10) -> StrategyResult:
        """
        Generate hypothetical ideal memories and search for similar ones.
        
        Creates example memories that would perfectly answer the query,
        then uses these to find actual memories with similar content.
        Especially useful for abstract or poorly-worded queries.
        
        Args:
            query: Abstract or difficult search query
            num_hypotheticals: Number of hypothetical memories to generate
            creativity_level: How creative/diverse the hypotheticals should be
            
        Returns:
            Memories similar to generated hypotheticals
        """
        start_time = time.time()
        
        try:
            # Generate hypothetical memory examples
            hypotheticals = self._generate_hypothetical_memories(query, num_hypotheticals)
            
            # Search for memories similar to each hypothetical
            all_results = []
            for hyp in hypotheticals:
                # Use hybrid retriever to find similar memories
                result = self.retriever.retrieve(hyp, max_results=10)
                all_results.extend([(mid, mem["score"]) for mid, mem in 
                                  zip([m["memory_id"] for m in result.memories],
                                      result.memories)])
            
            # Deduplicate and rank results
            memory_scores = defaultdict(list)
            for memory_id, score in all_results:
                memory_scores[memory_id].append(score)
            
            # Average scores for memories found multiple times
            final_results = []
            for memory_id, scores in memory_scores.items():
                avg_score = sum(scores) / len(scores)
                boost = min(len(scores) * 0.1, 0.3)  # Boost for multiple matches
                final_results.append((memory_id, avg_score + boost))
            
            # Sort by score and limit results
            final_results.sort(key=lambda x: x[1], reverse=True)
            memory_ids = [r[0] for r in final_results[:max_results]]
            scores = [r[1] for r in final_results[:max_results]]
            
            confidence = 0.8 if len(memory_ids) > 0 else 0.1
            
            return StrategyResult(
                memory_ids=memory_ids,
                scores=scores,
                strategy_used=SearchStrategy.HYPOTHETICAL_DOCUMENT,
                confidence=confidence,
                execution_time=time.time() - start_time,
                metadata={"hypotheticals": hypotheticals},
                explanation=f"Generated {len(hypotheticals)} hypothetical examples and found {len(memory_ids)} similar memories"
            )
            
        except Exception as e:
            self.logger.warning(f"Hypothetical document search failed: {e}")
            return StrategyResult(
                memory_ids=[],
                scores=[],
                strategy_used=SearchStrategy.HYPOTHETICAL_DOCUMENT,
                confidence=0.0,
                execution_time=time.time() - start_time,
                explanation=f"Failed to generate hypotheticals: {e}"
            )
    
    def temporal_search(self,
                       query: str,
                       time_expression: str,
                       content_weight: float = 0.7,
                       temporal_weight: float = 0.3) -> StrategyResult:
        """
        Search memories from specific time periods.
        
        Handles queries like "what did I think about X last week" or
        "my thoughts from 2023" by combining content search with
        temporal filtering.
        
        Args:
            query: Content query
            time_expression: Natural language time expression
            content_weight: Weight for content relevance
            temporal_weight: Weight for temporal relevance
            
        Returns:
            Time-filtered search results
        """
        start_time = time.time()
        
        try:
            # Parse temporal expression
            time_ranges = self._parse_temporal_expression(time_expression)
            
            if not time_ranges:
                return StrategyResult(
                    memory_ids=[],
                    scores=[],
                    strategy_used=SearchStrategy.TEMPORAL_SEARCH,
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    explanation="Could not parse temporal expression"
                )
            
            # Get content-based results first
            content_result = self.retriever.retrieve(query, max_results=50)
            
            # Filter and score by temporal relevance
            temporal_results = []
            for memory in content_result.memories:
                memory_id = memory["memory_id"]
                content_score = memory["score"]
                
                # Get memory timestamp
                memory_time = self._get_memory_timestamp(memory_id)
                if not memory_time:
                    continue
                
                # Check if memory falls within time ranges
                temporal_score = 0.0
                for start_time_range, end_time_range in time_ranges:
                    if start_time_range <= memory_time <= end_time_range:
                        # Score based on how close to center of range
                        range_center = start_time_range + (end_time_range - start_time_range) / 2
                        time_diff = abs((memory_time - range_center).total_seconds())
                        range_duration = (end_time_range - start_time_range).total_seconds()
                        temporal_score = max(1.0 - (time_diff / range_duration), 0.0)
                        break
                
                if temporal_score > 0:
                    combined_score = (content_score * content_weight + 
                                    temporal_score * temporal_weight)
                    temporal_results.append((memory_id, combined_score, temporal_score))
            
            # Sort by combined score
            temporal_results.sort(key=lambda x: x[1], reverse=True)
            memory_ids = [r[0] for r in temporal_results[:10]]
            scores = [r[1] for r in temporal_results[:10]]
            
            confidence = 0.9 if len(memory_ids) > 0 else 0.2
            
            return StrategyResult(
                memory_ids=memory_ids,
                scores=scores,
                strategy_used=SearchStrategy.TEMPORAL_SEARCH,
                confidence=confidence,
                execution_time=time.time() - start_time,
                metadata={
                    "time_ranges": [(str(s), str(e)) for s, e in time_ranges],
                    "temporal_weight": temporal_weight
                },
                explanation=f"Found {len(memory_ids)} memories matching temporal constraint: {time_expression}"
            )
            
        except Exception as e:
            self.logger.warning(f"Temporal search failed: {e}")
            return StrategyResult(
                memory_ids=[],
                scores=[],
                strategy_used=SearchStrategy.TEMPORAL_SEARCH,
                confidence=0.0,
                execution_time=time.time() - start_time,
                explanation=f"Temporal search error: {e}"
            )
    
    def concept_expansion_search(self,
                               query: str,
                               expansion_method: str = "semantic",
                               max_expansions: int = 5,
                               max_results: int = 10) -> StrategyResult:
        """
        Expand query with related concepts before searching.
        
        Identifies key concepts in the query and expands them with
        related terms, synonyms, or conceptually similar ideas to
        cast a wider net during search.
        
        Args:
            query: Original search query
            expansion_method: "semantic", "syntactic", "knowledge_graph"
            max_expansions: Maximum concepts to add
            
        Returns:
            Results from expanded query search
        """
        start_time = time.time()
        
        try:
            # Extract key concepts
            original_concepts = self._extract_concepts(query)
            
            # Expand concepts using specified method
            expanded_concepts = self._expand_concepts(
                original_concepts, method=expansion_method, max_expansions=max_expansions
            )
            
            # Build expanded query
            expanded_query = query
            if expanded_concepts:
                expansion_text = " ".join(expanded_concepts)
                expanded_query = f"{query} {expansion_text}"
            
            # Search with expanded query
            result = self.retriever.retrieve(expanded_query, max_results=max_results)
            
            confidence = 0.7 if len(result.memories) > 0 else 0.3
            
            return StrategyResult(
                memory_ids=[m["memory_id"] for m in result.memories],
                scores=[m["score"] for m in result.memories],
                strategy_used=SearchStrategy.CONCEPT_EXPANSION,
                confidence=confidence,
                execution_time=time.time() - start_time,
                metadata={
                    "original_concepts": original_concepts,
                    "expanded_concepts": expanded_concepts,
                    "expansion_method": expansion_method,
                    "expanded_query": expanded_query
                },
                explanation=f"Expanded query with {len(expanded_concepts)} related concepts using {expansion_method}"
            )
            
        except Exception as e:
            self.logger.warning(f"Concept expansion search failed: {e}")
            return StrategyResult(
                memory_ids=[],
                scores=[],
                strategy_used=SearchStrategy.CONCEPT_EXPANSION,
                confidence=0.0,
                execution_time=time.time() - start_time,
                explanation=f"Concept expansion error: {e}"
            )
    
    def similarity_clustering_search(self,
                                   query: str,
                                   cluster_threshold: float = 0.8,
                                   max_clusters: int = 5) -> StrategyResult:
        """
        Find clusters of similar memories and return representatives.
        
        Groups similar memories together and returns the best
        representative from each cluster, avoiding redundant results.
        
        Args:
            query: Search query
            cluster_threshold: Similarity threshold for clustering
            max_clusters: Maximum clusters to return
            
        Returns:
            Representative memories from each cluster
        """
        start_time = time.time()
        
        try:
            # Get initial results
            initial_result = self.retriever.retrieve(query, max_results=30)
            
            if len(initial_result.memories) < 2:
                return StrategyResult(
                    memory_ids=[m["memory_id"] for m in initial_result.memories],
                    scores=[m["score"] for m in initial_result.memories],
                    strategy_used=SearchStrategy.SIMILARITY_CLUSTERING,
                    confidence=0.5,
                    execution_time=time.time() - start_time,
                    explanation="Too few results for clustering"
                )
            
            # Cluster memories by similarity
            memory_ids = [m["memory_id"] for m in initial_result.memories]
            clusters = self._cluster_memories_by_similarity(memory_ids, cluster_threshold)
            
            # Select best representative from each cluster
            cluster_representatives = []
            for cluster in clusters[:max_clusters]:
                # Find memory with highest score in cluster
                best_memory = None
                best_score = -1
                
                for memory_id in cluster:
                    memory_data = next((m for m in initial_result.memories 
                                      if m["memory_id"] == memory_id), None)
                    if memory_data and memory_data["score"] > best_score:
                        best_memory = memory_id
                        best_score = memory_data["score"]
                
                if best_memory:
                    cluster_representatives.append((best_memory, best_score))
            
            # Sort by score
            cluster_representatives.sort(key=lambda x: x[1], reverse=True)
            memory_ids = [r[0] for r in cluster_representatives]
            scores = [r[1] for r in cluster_representatives]
            
            confidence = 0.8 if len(clusters) > 1 else 0.6
            
            return StrategyResult(
                memory_ids=memory_ids,
                scores=scores,
                strategy_used=SearchStrategy.SIMILARITY_CLUSTERING,
                confidence=confidence,
                execution_time=time.time() - start_time,
                metadata={
                    "num_clusters": len(clusters),
                    "cluster_threshold": cluster_threshold,
                    "total_candidates": len(memory_ids)
                },
                explanation=f"Clustered {len(initial_result.memories)} results into {len(clusters)} groups"
            )
            
        except Exception as e:
            self.logger.warning(f"Similarity clustering search failed: {e}")
            return StrategyResult(
                memory_ids=[],
                scores=[],
                strategy_used=SearchStrategy.SIMILARITY_CLUSTERING,
                confidence=0.0,
                execution_time=time.time() - start_time,
                explanation=f"Clustering error: {e}"
            )
    
    def keyword_fallback_search(self,
                              query: str,
                              use_stemming: bool = True,
                              fuzzy_matching: bool = True) -> StrategyResult:
        """
        Traditional keyword-based search as fallback.
        
        When semantic search fails, falls back to keyword matching
        with stemming and fuzzy matching capabilities.
        
        Args:
            query: Search query
            use_stemming: Whether to use word stemming
            fuzzy_matching: Whether to allow fuzzy matches
            
        Returns:
            Keyword-based search results
        """
        start_time = time.time()
        
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query, use_stemming)
            
            # Search memory contents for keyword matches
            matching_memories = []
            all_memory_ids = self.memory_graph.get_all_memory_ids()
            
            for memory_id in all_memory_ids:
                content = self.memory_graph.get_memory_content(memory_id)
                if not content:
                    continue
                
                # Score based on keyword matches
                score = self._score_keyword_match(content, keywords, fuzzy_matching)
                
                if score > 0:
                    matching_memories.append((memory_id, score))
            
            # Sort by score and limit results
            matching_memories.sort(key=lambda x: x[1], reverse=True)
            memory_ids = [r[0] for r in matching_memories[:10]]
            scores = [r[1] for r in matching_memories[:10]]
            
            confidence = 0.6 if len(memory_ids) > 0 else 0.1
            
            return StrategyResult(
                memory_ids=memory_ids,
                scores=scores,
                strategy_used=SearchStrategy.KEYWORD_FALLBACK,
                confidence=confidence,
                execution_time=time.time() - start_time,
                metadata={
                    "keywords": keywords,
                    "use_stemming": use_stemming,
                    "fuzzy_matching": fuzzy_matching
                },
                explanation=f"Keyword search found {len(memory_ids)} matches for {len(keywords)} keywords"
            )
            
        except Exception as e:
            self.logger.warning(f"Keyword fallback search failed: {e}")
            return StrategyResult(
                memory_ids=[],
                scores=[],
                strategy_used=SearchStrategy.KEYWORD_FALLBACK,
                confidence=0.0,
                execution_time=time.time() - start_time,
                explanation=f"Keyword search error: {e}"
            )
    
    def register_strategy(self, 
                         strategy_type: SearchStrategy,
                         strategy_instance: BaseSearchStrategy) -> None:
        """
        Register a custom search strategy.
        
        Args:
            strategy_type: Type of strategy
            strategy_instance: Strategy implementation
        """
        self.strategies[strategy_type] = strategy_instance
        self.logger.info(f"Registered strategy: {strategy_type}")
    
    def unregister_strategy(self, strategy_type: SearchStrategy) -> bool:
        """
        Remove a search strategy.
        
        Args:
            strategy_type: Strategy to remove
            
        Returns:
            True if strategy was removed
        """
        if strategy_type in self.strategies:
            del self.strategies[strategy_type]
            self.logger.info(f"Unregistered strategy: {strategy_type}")
            return True
        return False
    
    def get_strategy_recommendations(self,
                                   query: str,
                                   context: SearchContext) -> List[Tuple[SearchStrategy, float]]:
        """
        Get ranked strategy recommendations for a query.
        
        Args:
            query: Search query
            context: Search context
            
        Returns:
            List of (strategy, confidence_score) tuples
        """
        recommendations = []
        
        for strategy_type, strategy_instance in self.strategies.items():
            try:
                confidence = strategy_instance.can_handle(query, context)
                if confidence > 0:
                    recommendations.append((strategy_type, confidence))
            except Exception as e:
                self.logger.warning(f"Error getting recommendation for {strategy_type}: {e}")
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    def get_strategy_statistics(self) -> Dict[SearchStrategy, Dict[str, Any]]:
        """
        Get performance statistics for all strategies.
        
        Returns:
            Statistics for each strategy
        """
        return dict(self.strategy_stats)
    
    def _register_default_strategies(self) -> None:
        """Register built-in search strategies."""
        self.strategies[SearchStrategy.HYPOTHETICAL_DOCUMENT] = HypotheticalDocumentStrategy(self)
        self.strategies[SearchStrategy.TEMPORAL_SEARCH] = TemporalSearchStrategy(self)
        self.strategies[SearchStrategy.CONCEPT_EXPANSION] = ConceptExpansionStrategy(self)
        self.strategies[SearchStrategy.KEYWORD_FALLBACK] = KeywordFallbackStrategy(self)
    
    def _extract_temporal_expressions(self, query: str) -> List[str]:
        """
        Extract temporal expressions from query.
        
        Args:
            query: Query to analyze
            
        Returns:
            List of temporal expressions found
        """
        temporal_indicators = []
        query_lower = query.lower()
        
        # Common temporal patterns
        patterns = [
            r'last \w+', r'next \w+', r'yesterday', r'today', r'tomorrow',
            r'\d{4}', r'january|february|march|april|may|june|july|august|september|october|november|december',
            r'morning|afternoon|evening|night', r'week|month|year', r'ago', r'recently'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            temporal_indicators.extend(matches)
        
        return temporal_indicators
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        # Simple entity extraction - could be enhanced with NLP library
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        words = query.split()
        for word in words:
            if len(word) > 1 and word[0].isupper() and word.isalpha():
                entities.append(word)
        
        return entities
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query."""
        # Simple concept extraction - filter out stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = query.lower().split()
        concepts = [word for word in words if word not in stop_words and len(word) > 2]
        
        return concepts
    
    def _detect_question_type(self, query: str) -> Optional[str]:
        """Detect question type from query."""
        query_lower = query.lower()
        
        question_words = {
            'what': 'what',
            'when': 'when', 
            'where': 'where',
            'who': 'who',
            'how': 'how',
            'why': 'why'
        }
        
        for word, qtype in question_words.items():
            if query_lower.startswith(word):
                return qtype
        
        return None
    
    def _classify_query_type(self, query: str, temporal_indicators: List[str], question_type: Optional[str]) -> str:
        """Classify overall query type."""
        if temporal_indicators:
            return "temporal"
        elif question_type == "when":
            return "temporal"
        elif question_type in ["what", "who", "where"]:
            return "factual"
        elif len(query.split()) > 8:
            return "exploratory"
        else:
            return "associative"
    
    def _calculate_complexity_score(self, query: str, entities: List[str], concepts: List[str]) -> float:
        """Calculate query complexity score."""
        word_count = len(query.split())
        entity_count = len(entities)
        concept_count = len(concepts)
        
        # Enhanced complexity scoring
        base_complexity = word_count * 0.05 + entity_count * 0.1 + concept_count * 0.03
        
        # Bonus for long queries
        if word_count > 10:
            base_complexity += 0.3
        elif word_count > 6:
            base_complexity += 0.2
        
        # Bonus for multiple concepts
        if concept_count > 5:
            base_complexity += 0.2
        
        return min(base_complexity, 1.0)
    
    def _calculate_ambiguity_score(self, query: str) -> float:
        """Calculate query ambiguity score."""
        # Simple heuristics for ambiguity
        ambiguous_words = ['thing', 'stuff', 'something', 'anything', 'maybe', 'perhaps', 'kind of']
        
        query_lower = query.lower()
        ambiguity_indicators = sum(1 for word in ambiguous_words if word in query_lower)
        
        return min(ambiguity_indicators / 3.0, 1.0)
    
    def _suggest_strategies(self, query_type: str, temporal_indicators: List[str], 
                          question_type: Optional[str], complexity_score: float, 
                          ambiguity_score: float, context: SearchContext) -> List[SearchStrategy]:
        """Suggest strategies based on query analysis."""
        strategies = []
        
        # Temporal queries
        if temporal_indicators or query_type == "temporal":
            strategies.append(SearchStrategy.TEMPORAL_SEARCH)
        
        # High ambiguity - use hypothetical documents
        if ambiguity_score > 0.3:
            strategies.append(SearchStrategy.HYPOTHETICAL_DOCUMENT)
        
        # Complex queries - use concept expansion
        if complexity_score > 0.4:
            strategies.append(SearchStrategy.CONCEPT_EXPANSION)
        
        # Add clustering for exploratory queries
        if query_type == "exploratory":
            strategies.append(SearchStrategy.SIMILARITY_CLUSTERING)
        
        # Always include keyword fallback
        strategies.append(SearchStrategy.KEYWORD_FALLBACK)
        
        return strategies
    
    def _update_strategy_stats(self, strategy: SearchStrategy, result: StrategyResult) -> None:
        """Update performance statistics for a strategy."""
        stats = self.strategy_stats[strategy]
        
        stats["total_uses"] += 1
        
        # Update averages
        total_uses = stats["total_uses"]
        stats["avg_confidence"] = ((stats["avg_confidence"] * (total_uses - 1) + result.confidence) / total_uses)
        stats["avg_execution_time"] = ((stats["avg_execution_time"] * (total_uses - 1) + result.execution_time) / total_uses)
        
        # Simple success rate based on whether results were found
        success = 1.0 if len(result.memory_ids) > 0 else 0.0
        stats["success_rate"] = ((stats["success_rate"] * (total_uses - 1) + success) / total_uses)
    
    def _generate_hypothetical_memories(self, query: str, num_examples: int = 3) -> List[str]:
        """Generate hypothetical memory examples."""
        # Simple template-based generation (could be enhanced with LLM)
        templates = [
            f"I was thinking about {query} and realized that...",
            f"My thoughts on {query}: ...",
            f"Regarding {query}, I believe that...",
            f"An interesting insight about {query} is..."
        ]
        
        # Return subset of templates
        return templates[:num_examples]
    
    def _expand_concepts(self, concepts: List[str], method: str = "semantic", max_expansions: int = 5) -> List[str]:
        """Expand concepts using various methods."""
        # Simple synonym expansion (could be enhanced with word embeddings)
        expansion_map = {
            "ai": ["artificial intelligence", "machine learning", "neural networks"],
            "machine learning": ["ai", "deep learning", "algorithms"],
            "technology": ["tech", "innovation", "digital"],
            "work": ["job", "career", "professional", "business"],
            "idea": ["concept", "thought", "notion", "insight"]
        }
        
        expanded = []
        for concept in concepts:
            if concept.lower() in expansion_map:
                expanded.extend(expansion_map[concept.lower()][:max_expansions//len(concepts)])
        
        return expanded[:max_expansions]
    
    def _parse_temporal_expression(self, time_expr: str) -> List[Tuple[datetime, datetime]]:
        """Parse temporal expression into time ranges."""
        time_ranges = []
        now = datetime.now()
        
        if "last week" in time_expr.lower():
            end_time = now - timedelta(days=now.weekday())
            start_time = end_time - timedelta(days=7)
            time_ranges.append((start_time, end_time))
        elif "last month" in time_expr.lower():
            end_time = now.replace(day=1) - timedelta(days=1)
            start_time = end_time.replace(day=1)
            time_ranges.append((start_time, end_time))
        elif "yesterday" in time_expr.lower():
            start_time = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
            end_time = start_time + timedelta(days=1)
            time_ranges.append((start_time, end_time))
        
        return time_ranges
    
    def _get_memory_timestamp(self, memory_id: str) -> Optional[datetime]:
        """Get timestamp for a memory."""
        # This would integrate with the memory graph to get timestamps
        # For now, return None as placeholder
        return None
    
    def _cluster_memories_by_similarity(self, memory_ids: List[str], threshold: float = 0.8) -> List[List[str]]:
        """Cluster memories by similarity."""
        # Simple clustering - could be enhanced with proper similarity calculation
        # For now, return single clusters
        clusters = []
        remaining = memory_ids.copy()
        
        while remaining:
            cluster = [remaining.pop(0)]
            clusters.append(cluster)
        
        return clusters
    
    def _extract_keywords(self, query: str, use_stemming: bool = True) -> List[str]:
        """Extract keywords from query."""
        # Comprehensive stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'about', 'above', 'after', 'again', 'against', 'all', 'any', 'as', 'because', 'before',
            'below', 'between', 'both', 'during', 'each', 'few', 'from', 'further', 'if', 'into',
            'more', 'most', 'no', 'nor', 'not', 'now', 'only', 'other', 'out', 'over', 'own',
            'same', 'so', 'some', 'such', 'than', 'then', 'through', 'too', 'under', 'until',
            'up', 'very', 'what', 'when', 'where', 'which', 'while', 'who', 'why', 'how'
        }
        
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Simple stemming (remove common suffixes)
        if use_stemming:
            stemmed = []
            for word in keywords:
                if word.endswith('ing'):
                    word = word[:-3]
                elif word.endswith('ed'):
                    word = word[:-2]
                elif word.endswith('s') and len(word) > 3:
                    word = word[:-1]
                stemmed.append(word)
            keywords = stemmed
        
        return keywords
    
    def _score_keyword_match(self, content: str, keywords: List[str], fuzzy_matching: bool = True) -> float:
        """Score content based on keyword matches."""
        content_lower = content.lower()
        matches = 0
        
        for keyword in keywords:
            if keyword in content_lower:
                matches += 1
            elif fuzzy_matching:
                # Simple fuzzy matching - check if most characters match
                for word in content_lower.split():
                    if len(word) > 3 and len(keyword) > 3:
                        common_chars = sum(1 for c in keyword if c in word)
                        if common_chars / len(keyword) > 0.7:
                            matches += 0.5
                            break
        
        # Normalize score
        return matches / len(keywords) if keywords else 0.0


# Built-in strategy implementations
class HypotheticalDocumentStrategy(BaseSearchStrategy):
    """Strategy for hypothetical document generation and search"""
    
    def __init__(self, search_strategies):
        self.search_strategies = search_strategies
    
    def can_handle(self, query: str, context: SearchContext) -> float:
        """Check if query is abstract or poorly specified"""
        # High confidence for abstract/ambiguous queries
        ambiguity_indicators = ['thing', 'stuff', 'something', 'maybe', 'kind of']
        query_lower = query.lower()
        
        ambiguity_score = sum(1 for word in ambiguity_indicators if word in query_lower)
        
        # Also good for short, vague queries
        if len(query.split()) < 4:
            ambiguity_score += 1
        
        return min(ambiguity_score / 3.0, 1.0)
    
    def execute(self, query: str, context: SearchContext, max_results: int = 10) -> StrategyResult:
        """Execute hypothetical document search"""
        return self.search_strategies.hypothetical_document_search(query, max_results=max_results)


class TemporalSearchStrategy(BaseSearchStrategy):
    """Strategy for temporal/time-based searches"""
    
    def __init__(self, search_strategies):
        self.search_strategies = search_strategies
    
    def can_handle(self, query: str, context: SearchContext) -> float:
        """Check if query contains temporal indicators"""
        temporal_words = ['yesterday', 'today', 'tomorrow', 'last', 'next', 'ago', 'recently', 'week', 'month', 'year']
        query_lower = query.lower()
        
        temporal_score = sum(1 for word in temporal_words if word in query_lower)
        
        # Check for years
        if re.search(r'\b\d{4}\b', query):
            temporal_score += 1
        
        return min(temporal_score / 3.0, 1.0)
    
    def execute(self, query: str, context: SearchContext, max_results: int = 10) -> StrategyResult:
        """Execute temporal search"""
        # Extract temporal expression from query
        temporal_indicators = self.search_strategies._extract_temporal_expressions(query)
        time_expr = " ".join(temporal_indicators) if temporal_indicators else "recent"
        
        return self.search_strategies.temporal_search(query, time_expr)


class ConceptExpansionStrategy(BaseSearchStrategy):
    """Strategy for concept-based query expansion"""
    
    def __init__(self, search_strategies):
        self.search_strategies = search_strategies
    
    def can_handle(self, query: str, context: SearchContext) -> float:
        """Check if query would benefit from concept expansion"""
        # Good for medium-length queries with clear concepts
        word_count = len(query.split())
        
        if 3 <= word_count <= 8:
            return 0.7
        elif word_count > 8:
            return 0.5
        else:
            return 0.3
    
    def execute(self, query: str, context: SearchContext, max_results: int = 10) -> StrategyResult:
        """Execute concept expansion search"""
        return self.search_strategies.concept_expansion_search(query, max_results=max_results)


class KeywordFallbackStrategy(BaseSearchStrategy):
    """Fallback strategy using traditional keyword matching"""
    
    def __init__(self, search_strategies):
        self.search_strategies = search_strategies
    
    def can_handle(self, query: str, context: SearchContext) -> float:
        """Always available as fallback"""
        return 0.3  # Low confidence, but always available
    
    def execute(self, query: str, context: SearchContext, max_results: int = 10) -> StrategyResult:
        """Execute keyword-based search"""
        return self.search_strategies.keyword_fallback_search(query)


# Exception classes
class StrategyError(Exception):
    """Base exception for search strategy operations"""
    pass

class StrategyNotAvailableError(StrategyError):
    """Requested strategy is not available"""
    pass

class QueryAnalysisError(StrategyError):
    """Error during query analysis"""
    pass

class HypotheticalGenerationError(StrategyError):
    """Error generating hypothetical documents"""
    pass