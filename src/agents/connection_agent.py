"""
Connection Agent - Manages connection strengthening and creation between memory nodes.

This agent analyzes access patterns, content relationships, and temporal factors
to make intelligent decisions about which memories should be connected and how
strongly those connections should be reinforced.
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from datetime import datetime, timedelta
from collections import defaultdict

from src.core.memory_node import MemoryNode, ConnectionType as CoreConnectionType
from src.core.memory_graph import MemoryGraph


class ConnectionAnalysisType(Enum):
    """Types of connection analysis."""
    SEMANTIC = "semantic"          # Content similarity
    TEMPORAL = "temporal"          # Time-based relationship
    CAUSAL = "causal"             # Cause-effect relationship
    CONTEXTUAL = "contextual"     # Accessed in similar contexts
    SEQUENTIAL = "sequential"     # Accessed in sequence
    ASSOCIATIVE = "associative"   # Learned through co-access


@dataclass
class ConnectionSuggestion:
    """Represents a suggested connection between two memories."""
    source_id: str
    target_id: str
    connection_type: CoreConnectionType
    confidence_score: float
    reasoning: str
    suggested_weight: float
    analysis_type: ConnectionAnalysisType


@dataclass
class AccessEvent:
    """Represents a memory access event for analysis."""
    memory_id: str
    query: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str = "default"
    session_id: str = "default"
    result_rank: int = 0
    interaction_type: str = "search"  # 'search', 'browse', 'related'


@dataclass
class ConnectionStrengthening:
    """Represents a connection strengthening decision."""
    source_id: str
    target_id: str
    current_weight: float
    suggested_weight: float
    weight_change: float
    reasoning: str
    confidence: float


class ConnectionAgent:
    """
    Intelligent agent for managing memory connections.
    
    This agent analyzes usage patterns, content relationships, and temporal factors
    to make intelligent decisions about which memories should be connected and how
    strongly those connections should be reinforced.
    """
    
    def __init__(self, memory_graph: MemoryGraph, config: Optional[Dict] = None):
        """
        Initialize the Connection Agent.
        
        Args:
            memory_graph: The memory graph to manage connections for
            config: Configuration parameters
        """
        self.memory_graph = memory_graph
        self.config = config or {}
        
        # Configuration parameters
        self.semantic_threshold = self.config.get('semantic_threshold', 0.7)
        self.temporal_window = self.config.get('temporal_window_minutes', 60)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.max_connections_per_node = self.config.get('max_connections_per_node', 20)
        
        # Track access patterns for learning
        self.access_history: List[AccessEvent] = []
        self.co_access_patterns: Dict[Tuple[str, str], int] = defaultdict(int)
        self.session_sequences: Dict[str, List[str]] = defaultdict(list)
    
    def analyze_and_suggest_connections(
        self, 
        memory_ids: Optional[List[str]] = None,
        analysis_types: Optional[List[ConnectionAnalysisType]] = None
    ) -> List[ConnectionSuggestion]:
        """
        Analyze memories and suggest new connections.
        
        Args:
            memory_ids: Specific memory IDs to analyze (None for all)
            analysis_types: Types of analysis to perform (None for all)
            
        Returns:
            List of connection suggestions
        """
        if memory_ids is None:
            memory_ids = list(self.memory_graph.nodes.keys())
        
        if analysis_types is None:
            analysis_types = list(ConnectionAnalysisType)
        
        suggestions = []
        
        # Perform each type of analysis
        for analysis_type in analysis_types:
            if analysis_type == ConnectionAnalysisType.SEMANTIC:
                suggestions.extend(self._analyze_semantic_connections(memory_ids))
            elif analysis_type == ConnectionAnalysisType.TEMPORAL:
                suggestions.extend(self._analyze_temporal_connections(memory_ids))
            elif analysis_type == ConnectionAnalysisType.CONTEXTUAL:
                suggestions.extend(self._analyze_contextual_connections(memory_ids))
            elif analysis_type == ConnectionAnalysisType.SEQUENTIAL:
                suggestions.extend(self._analyze_sequential_connections(memory_ids))
            elif analysis_type == ConnectionAnalysisType.ASSOCIATIVE:
                suggestions.extend(self._analyze_associative_connections(memory_ids))
            elif analysis_type == ConnectionAnalysisType.CAUSAL:
                suggestions.extend(self._analyze_causal_connections(memory_ids))
        
        # Remove duplicates and sort by confidence
        unique_suggestions = self._deduplicate_suggestions(suggestions)
        unique_suggestions.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return unique_suggestions
    
    def record_access_event(self, event: AccessEvent):
        """Record a memory access event for learning."""
        self.access_history.append(event)
        
        # Track session sequences
        session_memories = self.session_sequences[event.session_id]
        if event.memory_id not in session_memories[-3:]:  # Avoid immediate duplicates
            session_memories.append(event.memory_id)
        
        # Track co-access patterns within temporal window
        recent_accesses = [
            e for e in self.access_history[-50:]  # Last 50 accesses
            if abs((e.timestamp - event.timestamp).total_seconds()) <= self.temporal_window * 60
        ]
        
        for recent_event in recent_accesses:
            if recent_event.memory_id != event.memory_id:
                pair = tuple(sorted([event.memory_id, recent_event.memory_id]))
                self.co_access_patterns[pair] += 1
    
    def suggest_connection_strengthening(
        self, 
        memory_id: str,
        context: Optional[Dict] = None
    ) -> List[ConnectionStrengthening]:
        """
        Suggest which connections for a memory should be strengthened.
        
        Args:
            memory_id: Memory to analyze connections for
            context: Additional context (query, access patterns, etc.)
            
        Returns:
            List of connection strengthening suggestions
        """
        memory = self.memory_graph.get_node(memory_id)
        if not memory:
            return []
        
        suggestions = []
        
        for target_id, connection in memory.connections.items():
            target_memory = self.memory_graph.get_node(target_id)
            if not target_memory:
                continue
            
            # Calculate suggested weight based on multiple factors
            factors = self._analyze_connection_strength_factors(
                memory, target_memory, connection, context
            )
            
            suggested_weight = self._calculate_suggested_weight(factors)
            weight_change = suggested_weight - connection.weight
            
            # Only suggest if significant change
            if abs(weight_change) >= 0.1:
                confidence = self._calculate_strengthening_confidence(factors)
                reasoning = self._generate_strengthening_reasoning(factors, weight_change)
                
                suggestions.append(ConnectionStrengthening(
                    source_id=memory_id,
                    target_id=target_id,
                    current_weight=connection.weight,
                    suggested_weight=suggested_weight,
                    weight_change=weight_change,
                    reasoning=reasoning,
                    confidence=confidence
                ))
        
        # Sort by absolute weight change (most significant first)
        suggestions.sort(key=lambda x: abs(x.weight_change), reverse=True)
        return suggestions
    
    def apply_connection_suggestions(
        self, 
        suggestions: List[ConnectionSuggestion],
        min_confidence: Optional[float] = None
    ) -> Dict[str, int]:
        """
        Apply connection suggestions to the memory graph.
        
        Args:
            suggestions: List of connection suggestions to apply
            min_confidence: Minimum confidence threshold (uses default if None)
            
        Returns:
            Dictionary with counts of applied suggestions by type
        """
        min_confidence = min_confidence or self.min_confidence
        applied_counts = defaultdict(int)
        
        for suggestion in suggestions:
            if suggestion.confidence_score >= min_confidence:
                success = self.memory_graph.create_connection(
                    source_id=suggestion.source_id,
                    target_id=suggestion.target_id,
                    connection_type=suggestion.connection_type,
                    initial_weight=suggestion.suggested_weight
                )
                
                if success:
                    applied_counts[suggestion.analysis_type.value] += 1
        
        return dict(applied_counts)
    
    def apply_strengthening_suggestions(
        self, 
        suggestions: List[ConnectionStrengthening],
        min_confidence: Optional[float] = None
    ) -> int:
        """
        Apply connection strengthening suggestions.
        
        Args:
            suggestions: List of strengthening suggestions
            min_confidence: Minimum confidence threshold
            
        Returns:
            Number of connections modified
        """
        min_confidence = min_confidence or self.min_confidence
        applied_count = 0
        
        for suggestion in suggestions:
            if suggestion.confidence >= min_confidence:
                source_memory = self.memory_graph.get_node(suggestion.source_id)
                if source_memory and suggestion.target_id in source_memory.connections:
                    connection = source_memory.connections[suggestion.target_id]
                    connection.weight = suggestion.suggested_weight
                    applied_count += 1
        
        return applied_count
    
    # Analysis methods
    
    def _analyze_semantic_connections(self, memory_ids: List[str]) -> List[ConnectionSuggestion]:
        """Analyze semantic relationships between memories."""
        suggestions = []
        
        for i, source_id in enumerate(memory_ids):
            source_memory = self.memory_graph.get_node(source_id)
            if not source_memory:
                continue
            
            for target_id in memory_ids[i+1:]:
                target_memory = self.memory_graph.get_node(target_id)
                if not target_memory or source_id == target_id:
                    continue
                
                # Skip if connection already exists
                if target_id in source_memory.connections:
                    continue
                
                # Calculate semantic similarity
                similarity = self._calculate_semantic_similarity(source_memory, target_memory)
                
                if similarity >= self.semantic_threshold:
                    confidence = min(0.95, similarity)
                    weight = similarity * 0.8  # Scale down for initial weight
                    
                    reasoning = f"High semantic similarity ({similarity:.3f}) between concepts"
                    
                    suggestions.append(ConnectionSuggestion(
                        source_id=source_id,
                        target_id=target_id,
                        connection_type=CoreConnectionType.SIMILARITY,
                        confidence_score=confidence,
                        reasoning=reasoning,
                        suggested_weight=weight,
                        analysis_type=ConnectionAnalysisType.SEMANTIC
                    ))
        
        return suggestions
    
    def _analyze_temporal_connections(self, memory_ids: List[str]) -> List[ConnectionSuggestion]:
        """Analyze temporal relationships between memories."""
        suggestions = []
        
        # Group access events by time
        recent_accesses = sorted(self.access_history[-100:], key=lambda x: x.timestamp)
        
        for i, event1 in enumerate(recent_accesses):
            if event1.memory_id not in memory_ids:
                continue
                
            for event2 in recent_accesses[i+1:i+6]:  # Look at next 5 events
                if event2.memory_id not in memory_ids or event1.memory_id == event2.memory_id:
                    continue
                
                # Check if accessed within temporal window
                time_diff = (event2.timestamp - event1.timestamp).total_seconds() / 60
                if time_diff <= self.temporal_window:
                    
                    # Check if connection already exists
                    source_memory = self.memory_graph.get_node(event1.memory_id)
                    if not source_memory or event2.memory_id in source_memory.connections:
                        continue
                    
                    # Calculate confidence based on temporal proximity
                    confidence = max(0.5, 1.0 - (time_diff / self.temporal_window))
                    weight = confidence * 0.6
                    
                    reasoning = f"Accessed within {time_diff:.1f} minutes of each other"
                    
                    suggestions.append(ConnectionSuggestion(
                        source_id=event1.memory_id,
                        target_id=event2.memory_id,
                        connection_type=CoreConnectionType.TEMPORAL_BEFORE,
                        confidence_score=confidence,
                        reasoning=reasoning,
                        suggested_weight=weight,
                        analysis_type=ConnectionAnalysisType.TEMPORAL
                    ))
        
        return suggestions
    
    def _analyze_contextual_connections(self, memory_ids: List[str]) -> List[ConnectionSuggestion]:
        """Analyze contextual relationships based on similar access contexts."""
        suggestions = []
        
        # Group accesses by similar queries/contexts
        query_groups = defaultdict(list)
        for event in self.access_history[-200:]:
            if event.memory_id in memory_ids:
                # Group by query similarity (simple word overlap)
                query_key = self._get_query_signature(event.query)
                query_groups[query_key].append(event)
        
        for query_key, events in query_groups.items():
            if len(events) < 2:
                continue
            
            # Create connections between memories accessed for similar queries
            for i, event1 in enumerate(events):
                for event2 in events[i+1:]:
                    if event1.memory_id == event2.memory_id:
                        continue
                    
                    source_memory = self.memory_graph.get_node(event1.memory_id)
                    if not source_memory or event2.memory_id in source_memory.connections:
                        continue
                    
                    # Calculate confidence based on query similarity and frequency
                    confidence = min(0.9, 0.5 + len(events) * 0.1)
                    weight = confidence * 0.7
                    
                    reasoning = f"Accessed in similar contexts ({len(events)} similar queries)"
                    
                    suggestions.append(ConnectionSuggestion(
                        source_id=event1.memory_id,
                        target_id=event2.memory_id,
                        connection_type=CoreConnectionType.CONTEXT,
                        confidence_score=confidence,
                        reasoning=reasoning,
                        suggested_weight=weight,
                        analysis_type=ConnectionAnalysisType.CONTEXTUAL
                    ))
        
        return suggestions
    
    def _analyze_sequential_connections(self, memory_ids: List[str]) -> List[ConnectionSuggestion]:
        """Analyze sequential access patterns within sessions."""
        suggestions = []
        
        for session_id, sequence in self.session_sequences.items():
            if len(sequence) < 2:
                continue
            
            for i in range(len(sequence) - 1):
                source_id = sequence[i]
                target_id = sequence[i + 1]
                
                if source_id not in memory_ids or target_id not in memory_ids:
                    continue
                
                source_memory = self.memory_graph.get_node(source_id)
                if not source_memory or target_id in source_memory.connections:
                    continue
                
                # Calculate confidence based on sequence frequency
                sequence_count = sum(
                    1 for seq in self.session_sequences.values()
                    if len(seq) >= i + 2 and seq[i:i+2] == [source_id, target_id]
                )
                
                confidence = min(0.8, 0.4 + sequence_count * 0.1)
                weight = confidence * 0.5
                
                reasoning = f"Sequential access pattern (observed {sequence_count} times)"
                
                suggestions.append(ConnectionSuggestion(
                    source_id=source_id,
                    target_id=target_id,
                    connection_type=CoreConnectionType.TEMPORAL_BEFORE,
                    confidence_score=confidence,
                    reasoning=reasoning,
                    suggested_weight=weight,
                    analysis_type=ConnectionAnalysisType.SEQUENTIAL
                ))
        
        return suggestions
    
    def _analyze_associative_connections(self, memory_ids: List[str]) -> List[ConnectionSuggestion]:
        """Analyze co-access patterns for associative connections."""
        suggestions = []
        
        for (mem1, mem2), count in self.co_access_patterns.items():
            if mem1 not in memory_ids or mem2 not in memory_ids or count < 2:
                continue
            
            source_memory = self.memory_graph.get_node(mem1)
            if not source_memory or mem2 in source_memory.connections:
                continue
            
            # Calculate confidence based on co-access frequency
            confidence = min(0.9, 0.3 + count * 0.1)
            weight = confidence * 0.6
            
            reasoning = f"Co-accessed {count} times within temporal window"
            
            suggestions.append(ConnectionSuggestion(
                source_id=mem1,
                target_id=mem2,
                connection_type=CoreConnectionType.SIMILARITY,
                confidence_score=confidence,
                reasoning=reasoning,
                suggested_weight=weight,
                analysis_type=ConnectionAnalysisType.ASSOCIATIVE
            ))
        
        return suggestions
    
    def _analyze_causal_connections(self, memory_ids: List[str]) -> List[ConnectionSuggestion]:
        """Analyze potential causal relationships between memories."""
        suggestions = []
        
        # Look for causal language patterns
        causal_patterns = [
            r'\bcause[sd]?\b', r'\bbecause\b', r'\bresult[sd]?\b', r'\blead[s]? to\b',
            r'\bdue to\b', r'\btherefore\b', r'\bconsequent\b', r'\beffect\b'
        ]
        
        for source_id in memory_ids:
            source_memory = self.memory_graph.get_node(source_id)
            if not source_memory:
                continue
            
            for target_id in memory_ids:
                if source_id == target_id:
                    continue
                
                target_memory = self.memory_graph.get_node(target_id)
                if not target_memory or target_id in source_memory.connections:
                    continue
                
                # Check for causal language in content
                combined_text = f"{source_memory.full_content} {target_memory.full_content}".lower()
                causal_matches = sum(1 for pattern in causal_patterns if re.search(pattern, combined_text))
                
                if causal_matches > 0:
                    confidence = min(0.7, 0.4 + causal_matches * 0.1)
                    weight = confidence * 0.5
                    
                    reasoning = f"Causal language detected ({causal_matches} patterns)"
                    
                    suggestions.append(ConnectionSuggestion(
                        source_id=source_id,
                        target_id=target_id,
                        connection_type=CoreConnectionType.CAUSE_EFFECT,
                        confidence_score=confidence,
                        reasoning=reasoning,
                        suggested_weight=weight,
                        analysis_type=ConnectionAnalysisType.CAUSAL
                    ))
        
        return suggestions
    
    # Helper methods
    
    def _calculate_semantic_similarity(self, memory1: MemoryNode, memory2: MemoryNode) -> float:
        """Calculate semantic similarity between two memories."""
        # Concept similarity
        concept_sim = self._text_similarity(memory1.concept, memory2.concept)
        
        # Tag overlap
        tags1 = set(tag.lower() for tag in memory1.tags)
        tags2 = set(tag.lower() for tag in memory2.tags)
        tag_overlap = len(tags1 & tags2) / max(len(tags1 | tags2), 1)
        
        # Keyword overlap
        keywords1 = set(word.lower() for word in memory1.keywords)
        keywords2 = set(word.lower() for word in memory2.keywords)
        keyword_overlap = len(keywords1 & keywords2) / max(len(keywords1 | keywords2), 1)
        
        # Content similarity
        content_sim = self._text_similarity(memory1.full_content, memory2.full_content)
        
        # Weighted combination
        similarity = (
            concept_sim * 0.4 +
            tag_overlap * 0.3 +
            keyword_overlap * 0.2 +
            content_sim * 0.1
        )
        
        return similarity
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity based on word overlap."""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _get_query_signature(self, query: str) -> str:
        """Get a signature for query grouping."""
        words = re.findall(r'\w+', query.lower())
        # Use first 3 significant words as signature
        significant_words = [w for w in words if len(w) > 3][:3]
        return " ".join(sorted(significant_words))
    
    def _deduplicate_suggestions(self, suggestions: List[ConnectionSuggestion]) -> List[ConnectionSuggestion]:
        """Remove duplicate suggestions, keeping the highest confidence."""
        seen_pairs = {}
        
        for suggestion in suggestions:
            pair_key = tuple(sorted([suggestion.source_id, suggestion.target_id]))
            
            if pair_key not in seen_pairs or suggestion.confidence_score > seen_pairs[pair_key].confidence_score:
                seen_pairs[pair_key] = suggestion
        
        return list(seen_pairs.values())
    
    def _analyze_connection_strength_factors(
        self, 
        source_memory: MemoryNode, 
        target_memory: MemoryNode, 
        connection, 
        context: Optional[Dict]
    ) -> Dict[str, float]:
        """Analyze factors that affect connection strength."""
        factors = {}
        
        # Current usage count
        factors['usage_frequency'] = min(1.0, connection.usage_count / 10.0)
        
        # Semantic similarity
        factors['semantic_similarity'] = self._calculate_semantic_similarity(source_memory, target_memory)
        
        # Co-access patterns
        pair_key = tuple(sorted([source_memory.id, target_memory.id]))
        factors['co_access_frequency'] = min(1.0, self.co_access_patterns.get(pair_key, 0) / 5.0)
        
        # Recency of access
        recent_accesses = [e for e in self.access_history[-50:] if e.memory_id in [source_memory.id, target_memory.id]]
        factors['access_recency'] = min(1.0, len(recent_accesses) / 10.0)
        
        # Context relevance (if provided)
        if context and 'query' in context:
            query_relevance = max(
                self._text_similarity(context['query'], source_memory.concept),
                self._text_similarity(context['query'], target_memory.concept)
            )
            factors['context_relevance'] = query_relevance
        else:
            factors['context_relevance'] = 0.5
        
        return factors
    
    def _calculate_suggested_weight(self, factors: Dict[str, float]) -> float:
        """Calculate suggested connection weight based on factors."""
        # Weighted combination of factors
        weight = (
            factors['usage_frequency'] * 0.3 +
            factors['semantic_similarity'] * 0.25 +
            factors['co_access_frequency'] * 0.2 +
            factors['access_recency'] * 0.15 +
            factors['context_relevance'] * 0.1
        )
        
        return min(1.0, weight)
    
    def _calculate_strengthening_confidence(self, factors: Dict[str, float]) -> float:
        """Calculate confidence in the strengthening suggestion."""
        # Higher confidence when multiple factors agree
        strong_factors = sum(1 for value in factors.values() if value >= 0.7)
        confidence = 0.5 + strong_factors * 0.15
        
        return min(0.95, confidence)
    
    def _generate_strengthening_reasoning(self, factors: Dict[str, float], weight_change: float) -> str:
        """Generate reasoning for connection strengthening."""
        strong_factors = [name for name, value in factors.items() if value >= 0.7]
        
        if weight_change > 0:
            action = "strengthen"
        else:
            action = "weaken"
        
        if strong_factors:
            reason = f"Should {action} based on: {', '.join(strong_factors)}"
        else:
            reason = f"Should {action} based on overall usage patterns"
        
        return reason
    
    def get_connection_statistics(self) -> Dict[str, any]:
        """Get statistics about connection patterns and suggestions."""
        stats = {
            'total_access_events': len(self.access_history),
            'unique_co_access_pairs': len(self.co_access_patterns),
            'active_sessions': len(self.session_sequences),
            'total_connections': sum(len(node.connections) for node in self.memory_graph.nodes.values()),
            'average_connections_per_node': 0
        }
        
        if self.memory_graph.nodes:
            stats['average_connections_per_node'] = stats['total_connections'] / len(self.memory_graph.nodes)
        
        return stats