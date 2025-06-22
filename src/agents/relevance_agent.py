"""
Relevance Agent - Evaluates memory relevance for queries and contexts.

This agent serves as the primary gatekeeper for determining which memories
are relevant to a given query or context. It's used by the connection_agent
to make intelligent strengthening decisions and by the retrieval system
to rank and filter search results.
"""

from typing import List, Dict, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import math
from datetime import datetime, timedelta

from src.core.memory_node import MemoryNode
from src.core.memory_graph import MemoryGraph


class RelevanceType(Enum):
    """Different dimensions of relevance."""
    SEMANTIC = "semantic"              # Content/meaning similarity
    CONTEXTUAL = "contextual"          # Similar usage context
    TEMPORAL = "temporal"              # Time-based relevance
    TOPICAL = "topical"               # Topic/domain relevance
    FUNCTIONAL = "functional"          # Serves similar purpose
    ASSOCIATIVE = "associative"        # Learned through usage patterns


class QueryType(Enum):
    """Types of queries the agent handles."""
    SEARCH = "search"                  # Explicit search query
    BROWSE = "browse"                 # Browsing related memories
    CONTEXTUAL = "contextual"         # Context-based retrieval
    SIMILAR = "similar"               # Finding similar memories
    COMPLETION = "completion"          # Auto-completion/suggestion


@dataclass
class RelevanceScore:
    """Comprehensive relevance assessment result."""
    overall: float = 0.0              # 0.0 to 1.0
    confidence: float = 0.0           # Confidence in the assessment
    semantic_score: float = 0.0       # Content/meaning similarity
    context_score: float = 0.0        # Context relevance
    temporal_score: float = 0.0       # Time-based relevance
    topical_score: float = 0.0        # Topic/domain relevance
    reasoning: str = ""               # Human-readable explanation
    must_keep: bool = False           # Flag to bypass all filtering constraints


@dataclass
class QueryContext:
    """Context information for relevance evaluation"""
    query: str
    conversation_history: List[str] = field(default_factory=list)
    user_intent: str = "general"
    domain: str = "general"
    query_type: QueryType = QueryType.SEARCH
    timestamp: datetime = field(default_factory=datetime.now)


class RelevanceAgent:
    """
    Intelligent agent for evaluating memory relevance.
    
    This agent analyzes multiple dimensions of relevance to determine
    how well a memory matches a given query and context. It serves as
    the primary filter for connection strengthening and search ranking.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Relevance Agent.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Configuration weights
        self.semantic_weight = self.config.get('semantic_weight', 0.4)
        self.context_weight = self.config.get('context_weight', 0.3)
        self.temporal_weight = self.config.get('temporal_weight', 0.1)
        self.topical_weight = self.config.get('topical_weight', 0.2)
        
        # Thresholds
        self.relevance_threshold = self.config.get('relevance_threshold', 0.6)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
    
    def evaluate_relevance(
        self, 
        memory: MemoryNode, 
        query: str,
        context: QueryContext
    ) -> RelevanceScore:
        """
        Comprehensive relevance evaluation for a memory.
        
        Args:
            memory: The memory to evaluate
            query: Query string
            context: Complete query context
            
        Returns:
            Detailed relevance assessment
        """
        # Calculate individual dimension scores
        semantic_score = self._calculate_semantic_relevance(memory, query)
        context_score = self._calculate_contextual_relevance(memory, context)
        temporal_score = self._calculate_temporal_relevance(memory, context)
        topical_score = self._calculate_topical_relevance(memory, context)
        
        # Calculate weighted overall score
        overall = (
            semantic_score * self.semantic_weight +
            context_score * self.context_weight +
            temporal_score * self.temporal_weight +
            topical_score * self.topical_weight
        )
        
        # Calculate confidence based on score variance
        scores = [semantic_score, context_score, temporal_score, topical_score]
        variance = sum((score - overall) ** 2 for score in scores) / len(scores)
        confidence = max(0.0, 1.0 - variance)
        
        # Determine if this memory should be flagged as must-keep
        must_keep = self._should_flag_must_keep(
            memory, query, context, semantic_score, context_score, temporal_score, topical_score, overall, confidence
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            semantic_score, context_score, temporal_score, topical_score, overall, must_keep
        )
        
        return RelevanceScore(
            overall=overall,
            confidence=confidence,
            semantic_score=semantic_score,
            context_score=context_score,
            temporal_score=temporal_score,
            topical_score=topical_score,
            reasoning=reasoning,
            must_keep=must_keep
        )
    
    def _calculate_semantic_relevance(self, memory: MemoryNode, query: str) -> float:
        """Calculate semantic similarity between memory content and query."""
        query_lower = query.lower()
        
        # Text matching components
        concept_match = self._text_similarity(memory.concept.lower(), query_lower)
        summary_match = self._text_similarity(memory.summary.lower(), query_lower)
        content_match = self._text_similarity(memory.full_content.lower(), query_lower)
        
        # Keyword and tag matching
        query_words = set(re.findall(r'\w+', query_lower))
        memory_keywords = set(word.lower() for word in memory.keywords)
        memory_tags = set(tag.lower() for tag in memory.tags)
        
        keyword_overlap = len(query_words & memory_keywords) / max(len(query_words), 1)
        tag_overlap = len(query_words & memory_tags) / max(len(query_words), 1)
        
        # Boost for exact matches in keywords/tags
        exact_keyword_boost = 0.5 if any(word in memory_keywords for word in query_words) else 0.0
        exact_tag_boost = 0.3 if any(word in memory_tags for word in query_words) else 0.0
        
        # Weighted combination with boost
        semantic_score = (
            concept_match * 0.3 +
            summary_match * 0.25 +
            content_match * 0.15 +
            keyword_overlap * 0.1 +
            tag_overlap * 0.1 +
            exact_keyword_boost * 0.05 +
            exact_tag_boost * 0.05
        )
        
        return min(1.0, semantic_score)
    
    def _calculate_contextual_relevance(self, memory: MemoryNode, context: QueryContext) -> float:
        """Calculate relevance based on conversation context."""
        if not context.conversation_history:
            return 0.5  # Neutral score when no context
        
        # Check if memory content relates to conversation history
        context_text = " ".join(context.conversation_history).lower()
        memory_text = f"{memory.concept} {memory.summary} {' '.join(memory.keywords)}".lower()
        
        context_similarity = self._text_similarity(memory_text, context_text)
        
        # Boost for specific keyword matches in conversation history
        context_words = set(re.findall(r'\w+', context_text))
        memory_keywords = set(word.lower() for word in memory.keywords)
        memory_tags = set(tag.lower() for tag in memory.tags)
        
        keyword_context_match = len(context_words & memory_keywords) / max(len(memory_keywords), 1)
        tag_context_match = len(context_words & memory_tags) / max(len(memory_tags), 1)
        
        # Domain matching
        domain_match = 0.0
        if context.domain and context.domain.lower() in [tag.lower() for tag in memory.tags]:
            domain_match = 1.0
        elif context.domain and context.domain.lower() in memory.concept.lower():
            domain_match = 0.8
        
        # Intent matching
        intent_match = 0.5  # Default neutral
        if context.user_intent == "learning" and any(
            word in memory.summary.lower() 
            for word in ["guide", "tutorial", "how", "learn", "understand"]
        ):
            intent_match = 0.9
        
        return (
            context_similarity * 0.3 + 
            keyword_context_match * 0.25 +
            tag_context_match * 0.15 +
            domain_match * 0.2 + 
            intent_match * 0.1
        )
    
    def _calculate_temporal_relevance(self, memory: MemoryNode, context: QueryContext) -> float:
        """Calculate time-based relevance factors."""
        # Recency boost for recently accessed memories
        access_boost = min(1.0, memory.access_count / 10.0)
        
        # Higher importance score indicates more relevance
        importance_factor = memory.importance_score
        
        # Combine factors
        temporal_score = (access_boost * 0.6 + importance_factor * 0.4)
        
        return temporal_score
    
    def _calculate_topical_relevance(self, memory: MemoryNode, context: QueryContext) -> float:
        """Calculate topic/domain relevance."""
        query_words = set(re.findall(r'\w+', context.query.lower()))
        
        # Tag relevance
        memory_tags = set(tag.lower() for tag in memory.tags)
        tag_overlap = len(query_words & memory_tags) / max(len(query_words), 1)
        
        # Concept relevance
        concept_words = set(re.findall(r'\w+', memory.concept.lower()))
        concept_overlap = len(query_words & concept_words) / max(len(query_words), 1)
        
        # Domain-specific boost
        domain_boost = 0.0
        if context.domain != "general":
            if context.domain.lower() in [tag.lower() for tag in memory.tags]:
                domain_boost = 0.3
        
        topical_score = tag_overlap * 0.5 + concept_overlap * 0.3 + domain_boost
        
        return min(1.0, topical_score)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate enhanced text similarity based on word overlap and matching."""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Bonus for important word matches (longer words get more weight)
        important_matches = sum(1 + len(word) * 0.1 for word in intersection if len(word) >= 4)
        importance_bonus = min(0.4, important_matches * 0.1)
        
        # Bonus for high match ratio
        match_ratio = len(intersection) / min(len(words1), len(words2))
        ratio_bonus = min(0.3, match_ratio * 0.3)
        
        return min(1.0, jaccard + importance_bonus + ratio_bonus)
    
    def _should_flag_must_keep(
        self, 
        memory: MemoryNode,
        query: str,
        context: QueryContext,
        semantic_score: float,
        context_score: float,
        temporal_score: float,
        topical_score: float,
        overall_score: float,
        confidence: float
    ) -> bool:
        """
        Determine if a memory should be flagged as must-keep.
        
        Must-keep criteria:
        1. Exceptional overall relevance (>= 0.9) with high confidence
        2. Perfect semantic match with strong context relevance
        3. Critical domain-specific knowledge (high importance + perfect topic match)
        4. Recently accessed memory with strong relevance
        5. Memory with specific "critical" or "important" tags
        6. Very high confidence with strong multi-dimensional relevance
        """
        
        # Criterion 1: Exceptional overall relevance with high confidence
        if overall_score >= 0.9 and confidence >= 0.8:
            return True
        
        # Criterion 2: Perfect semantic match with strong context relevance
        if semantic_score >= 0.95 and context_score >= 0.7:
            return True
        
        # Criterion 3: Critical domain-specific knowledge
        if (memory.importance_score >= 0.8 and topical_score >= 0.9) or memory.importance_score >= 0.95:
            return True
        
        # Criterion 4: Recently accessed memory with strong relevance
        if memory.access_count >= 5 and overall_score >= 0.7:
            return True
        
        # Criterion 5: Memory tagged as critical/important
        critical_tags = {'critical', 'important', 'essential', 'key', 'core', 'fundamental'}
        if any(tag.lower() in critical_tags for tag in memory.tags):
            return True
        
        # Criterion 6: Very high confidence with strong multi-dimensional relevance
        strong_dimensions = sum(1 for score in [semantic_score, context_score, topical_score] if score >= 0.8)
        if confidence >= 0.9 and strong_dimensions >= 2 and overall_score >= 0.75:
            return True
        
        # Criterion 7: Exact concept match with good overall relevance
        query_words = set(query.lower().split())
        concept_words = set(memory.concept.lower().split())
        if len(query_words & concept_words) >= 2 and overall_score >= 0.6:
            return True
        
        return False
    
    def _generate_reasoning(
        self, 
        semantic: float, 
        context: float, 
        temporal: float, 
        topical: float, 
        overall: float,
        must_keep: bool = False
    ) -> str:
        """Generate human-readable reasoning for the relevance score."""
        reasoning_parts = []
        
        # Add must-keep flag first if present
        if must_keep:
            reasoning_parts.append("MUST-KEEP")
        
        if semantic >= 0.7:
            reasoning_parts.append("High semantic match")
        elif semantic >= 0.4:
            reasoning_parts.append("Moderate semantic match")
        else:
            reasoning_parts.append("Low semantic match")
        
        if context >= 0.7:
            reasoning_parts.append("strong context relevance")
        elif context >= 0.4:
            reasoning_parts.append("some context relevance")
        
        if temporal >= 0.7:
            reasoning_parts.append("recently accessed/important")
        
        if topical >= 0.7:
            reasoning_parts.append("topically relevant")
        
        # Conclusion based on overall score and must-keep flag
        if must_keep:
            conclusion = "Critical memory (flagged as must-keep)"
        elif overall >= 0.8:
            conclusion = "Highly relevant memory"
        elif overall >= 0.6:
            conclusion = "Moderately relevant memory"
        elif overall >= 0.3:
            conclusion = "Somewhat relevant memory"
        else:
            conclusion = "Low relevance memory"
        
        reasoning = f"{conclusion}. " + ", ".join(reasoning_parts) + "."
        return reasoning
    
    def filter_by_relevance(
        self, 
        memories: List[MemoryNode], 
        query: str,
        context: QueryContext,
        threshold: float = None
    ) -> List[Tuple[MemoryNode, RelevanceScore]]:
        """
        Filter memories by relevance threshold and return with scores.
        
        Args:
            memories: List of memories to evaluate
            query: Query string
            context: Query context
            threshold: Minimum relevance threshold (uses default if None)
            
        Returns:
            List of (memory, relevance_score) tuples that meet threshold
        """
        threshold = threshold or self.relevance_threshold
        relevant_memories = []
        
        for memory in memories:
            score = self.evaluate_relevance(memory, query, context)
            if score.overall >= threshold:
                relevant_memories.append((memory, score))
        
        # Sort by relevance score (highest first)
        relevant_memories.sort(key=lambda x: x[1].overall, reverse=True)
        
        return relevant_memories