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
import logging
import os
from anthropic import Anthropic

from src.core.memory_node import MemoryNode
from src.core.memory_graph import MemoryGraph

logger = logging.getLogger(__name__)


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
    functional_score: float = 0.0     # Functional/purpose relevance
    associative_score: float = 0.0    # AI-assessed associative relevance
    connection_strength_score: float = 0.0  # Existing connection strength
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
    
    def __init__(self, config: Optional[Dict] = None, claude_client=None, memory_graph=None):
        """
        Initialize the Relevance Agent.
        
        Args:
            config: Configuration parameters
            claude_client: Optional Claude API client for associative scoring
            memory_graph: Memory graph for connection strength analysis
        """
        self.config = config or {}
        self.claude_client = claude_client
        self.memory_graph = memory_graph
        
        # Connection learning state
        self.connection_modification_count = 0  # Track how many times connections have been modified
        
        # Configuration weights - dynamically adjusted based on connection learning
        self._base_weights = {
            'semantic': self.config.get('semantic_weight', 0.25),
            'context': self.config.get('context_weight', 0.20), 
            'temporal': self.config.get('temporal_weight', 0.10),
            'topical': self.config.get('topical_weight', 0.15),
            'functional': self.config.get('functional_weight', 0.15),
            'associative': self.config.get('associative_weight', 0.15),
            'connection_strength': self.config.get('connection_strength_weight', 0.0)  # Starts at 0, grows with usage
        }
        
        # Current weights (will be dynamically adjusted)
        self._update_weights()
        
        # Thresholds
        self.relevance_threshold = self.config.get('relevance_threshold', 0.6)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Conversation history configuration
        self.conversation_history_length = self.config.get('conversation_history_length', 3)
        
        # Associative scoring configuration
        self.enable_claude_associative = self.config.get('enable_claude_associative', True)
        self.claude_timeout = self.config.get('claude_timeout', 10.0)
        self.claude_cache = {}  # Simple cache for repeated queries
    
    def _update_weights(self):
        """Update weights dynamically based on connection modification count."""
        # Connection strength weight grows with usage, maxing out at 0.6 (60% of total weight)
        connection_strength_factor = min(0.6, self.connection_modification_count * 0.05)
        
        # Remaining weight distributed among other dimensions
        remaining_weight = 1.0 - connection_strength_factor
        
        # Normalize other weights to fit remaining space
        other_weights_sum = sum(v for k, v in self._base_weights.items() if k != 'connection_strength')
        if other_weights_sum > 0:
            normalization_factor = remaining_weight / other_weights_sum
        else:
            normalization_factor = 1.0
        
        # Set current weights
        self.semantic_weight = self._base_weights['semantic'] * normalization_factor
        self.context_weight = self._base_weights['context'] * normalization_factor
        self.temporal_weight = self._base_weights['temporal'] * normalization_factor
        self.topical_weight = self._base_weights['topical'] * normalization_factor
        self.functional_weight = self._base_weights['functional'] * normalization_factor
        self.associative_weight = self._base_weights['associative'] * normalization_factor
        self.connection_strength_weight = connection_strength_factor
    
    def record_connection_modification(self):
        """Record that a connection has been created or modified."""
        self.connection_modification_count += 1
        self._update_weights()
    
    def evaluate_relevance(
        self, 
        memory: MemoryNode, 
        query: str,
        context: QueryContext,
        reference_memory_ids: Optional[List[str]] = None
    ) -> RelevanceScore:
        """
        Comprehensive relevance evaluation for a memory.
        
        Args:
            memory: The memory to evaluate
            query: Query string
            context: Complete query context
            reference_memory_ids: List of memory IDs to check for connections (e.g., recently accessed memories)
            
        Returns:
            Detailed relevance assessment
        """
        # Calculate individual dimension scores
        semantic_score = self._calculate_semantic_relevance(memory, query)
        context_score = self._calculate_contextual_relevance(memory, context)
        temporal_score = self._calculate_temporal_relevance(memory, context)
        topical_score = self._calculate_topical_relevance(memory, context)
        functional_score = self._calculate_functional_relevance(memory, context)
        associative_score = self._calculate_associative_relevance(memory, query, context)
        connection_strength_score = self._calculate_connection_strength_relevance(memory, reference_memory_ids or [])
        
        # Calculate weighted overall score
        overall = (
            semantic_score * self.semantic_weight +
            context_score * self.context_weight +
            temporal_score * self.temporal_weight +
            topical_score * self.topical_weight +
            functional_score * self.functional_weight +
            associative_score * self.associative_weight +
            connection_strength_score * self.connection_strength_weight
        )
        
        # Calculate confidence based on score variance
        scores = [semantic_score, context_score, temporal_score, topical_score, functional_score, associative_score, connection_strength_score]
        variance = sum((score - overall) ** 2 for score in scores) / len(scores)
        confidence = max(0.0, 1.0 - variance)
        
        # Determine if this memory should be flagged as must-keep
        must_keep = self._should_flag_must_keep(
            memory, query, context, semantic_score, context_score, temporal_score, topical_score, overall, confidence
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            semantic_score, context_score, temporal_score, topical_score, functional_score, associative_score, connection_strength_score, overall, must_keep
        )
        
        return RelevanceScore(
            overall=overall,
            confidence=confidence,
            semantic_score=semantic_score,
            context_score=context_score,
            temporal_score=temporal_score,
            topical_score=topical_score,
            functional_score=functional_score,
            associative_score=associative_score,
            connection_strength_score=connection_strength_score,
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
    
    def _calculate_functional_relevance(self, memory: MemoryNode, context: QueryContext) -> float:
        """Calculate functional/purpose relevance based on query intent and memory function."""
        # Functional relevance keywords by intent
        intent_keywords = {
            "learning": ["guide", "tutorial", "how", "learn", "understand", "explain", "basics", "introduction"],
            "problem_solving": ["solve", "fix", "debug", "error", "issue", "problem", "troubleshoot"],
            "reference": ["reference", "documentation", "api", "specification", "manual"],
            "implementation": ["code", "example", "implement", "build", "create", "develop"],
            "analysis": ["analyze", "compare", "evaluate", "assess", "review", "study"],
            "general": ["information", "about", "what", "describe", "overview"]
        }
        
        user_intent = context.user_intent if context.user_intent in intent_keywords else "general"
        relevant_keywords = intent_keywords[user_intent]
        
        # Check memory content for functional indicators
        memory_text = f"{memory.concept} {memory.summary} {memory.full_content}".lower()
        
        functional_matches = sum(1 for keyword in relevant_keywords if keyword in memory_text)
        functional_score = min(1.0, functional_matches / len(relevant_keywords))
        
        # Boost based on query type alignment
        query_lower = context.query.lower()
        if context.query_type == QueryType.SEARCH and any(word in query_lower for word in ["how", "what", "why"]):
            if user_intent == "learning" and any(word in memory_text for word in ["guide", "tutorial", "how"]):
                functional_score += 0.3
        
        # Boost for action-oriented queries matched with implementation content
        action_words = ["create", "build", "make", "implement", "develop", "code"]
        if any(word in query_lower for word in action_words) and any(word in memory_text for word in action_words):
            functional_score += 0.2
        
        return min(1.0, functional_score)
    
    def _calculate_associative_relevance(self, memory: MemoryNode, query: str, context: QueryContext) -> float:
        """Calculate associative relevance using Claude AI for intelligent assessment."""
        if not self.enable_claude_associative or not self.claude_client:
            # Fallback to simple heuristic if Claude is not available
            logger.info(f"Claude fallback used for memory '{memory.concept}': Claude not available (enabled={self.enable_claude_associative}, client={self.claude_client is not None})")
            return self._calculate_associative_fallback(memory, query, context)
        
        # Create cache key for this evaluation
        cache_key = f"{memory.id}:{hash(query)}:{hash(str(context.conversation_history))}"
        if cache_key in self.claude_cache:
            return self.claude_cache[cache_key]
        
        try:
            # Prepare context for Claude
            conversation_context = ""
            if context.conversation_history:
                recent_messages = context.conversation_history[-self.conversation_history_length:]  # Use configurable length
                conversation_context = f"Recent conversation: {' | '.join(recent_messages)}"
            
            # Construct prompt for Claude
            prompt = f"""Evaluate the relevance of this memory to the current query and conversation context.

Memory:
- Concept: {memory.concept}
- Summary: {memory.summary}
- Content: {memory.full_content[:500]}...
- Tags: {memory.tags}

Current Query: {query}
{conversation_context}
User Intent: {context.user_intent}
Domain: {context.domain}

Consider:
1. How well does this memory relate to the current query?
2. How useful would this memory be in the context of the recent conversation?
3. Are there subtle connections or associations that might not be obvious?
4. Would referencing this memory enhance the response quality?

Rate the associative relevance on a scale of 0.0 to 1.0, where:
- 0.0 = No useful association or relevance
- 0.3 = Minimal association, might provide background context
- 0.5 = Moderate association, could be useful supplementary information  
- 0.7 = Strong association, likely to enhance the response
- 0.9 = Very strong association, highly relevant and valuable

Respond with just a number between 0.0 and 1.0, followed by a brief reason (max 20 words).
Format: "0.X Brief reason here"
"""
            
            # Call Claude API
            response = self._call_claude_api(prompt)
            
            # Parse response
            score = self._parse_claude_response(response)
            
            # Cache the result
            self.claude_cache[cache_key] = score
            
            return score
            
        except Exception as e:
            # Fall back to heuristic on any error
            logger.warning(f"Claude API error for associative scoring of memory '{memory.concept}': {e}. Falling back to heuristic method.")
            return self._calculate_associative_fallback(memory, query, context)
    
    def _calculate_associative_fallback(self, memory: MemoryNode, query: str, context: QueryContext) -> float:
        """Fallback heuristic method for associative relevance when Claude is not available."""
        # Simple associative heuristics
        score = 0.0
        
        # Check for indirect word associations
        query_words = set(re.findall(r'\w+', query.lower()))
        memory_words = set(re.findall(r'\w+', f"{memory.concept} {memory.summary}".lower()))
        
        # Related concept detection
        concept_associations = {
            "async": ["concurrency", "parallel", "threading", "performance"],
            "python": ["programming", "development", "coding", "script"],
            "database": ["data", "storage", "query", "sql"],
            "error": ["exception", "debug", "troubleshoot", "fix"],
            "api": ["interface", "service", "endpoint", "integration"],
        }
        
        for query_word in query_words:
            if query_word in concept_associations:
                related_concepts = concept_associations[query_word]
                if any(concept in memory_words for concept in related_concepts):
                    score += 0.2
        
        # Conversation context associations
        if context.conversation_history:
            context_words = set()
            for msg in context.conversation_history[-self.conversation_history_length:]:  # Use configurable length
                context_words.update(re.findall(r'\w+', msg.lower()))
            
            context_overlap = len(context_words & memory_words) / max(len(context_words), 1)
            score += context_overlap * 0.3
        
        # Intent-based associations
        if context.user_intent == "learning" and any(word in memory.tags for word in ["tutorial", "guide", "basic"]):
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_connection_strength_relevance(self, memory: MemoryNode, reference_memory_ids: List[str]) -> float:
        """
        Calculate relevance based on existing connection strengths to reference memories.
        
        This is the primary method for determining relevance when the system has learned
        from previous interactions and built strong connections.
        
        Args:
            memory: The memory to evaluate
            reference_memory_ids: List of memory IDs to check connections against (e.g., recently accessed)
            
        Returns:
            Connection strength score (0.0 to 1.0)
        """
        if not reference_memory_ids or not self.memory_graph:
            return 0.0
        
        # Calculate maximum connection strength to any reference memory
        max_connection_strength = 0.0
        weighted_connection_sum = 0.0
        total_connections = 0
        
        for ref_memory_id in reference_memory_ids:
            # Check connection from memory to reference
            if ref_memory_id in memory.connections:
                connection = memory.connections[ref_memory_id]
                strength = connection.weight
                max_connection_strength = max(max_connection_strength, strength)
                weighted_connection_sum += strength
                total_connections += 1
            
            # Check reverse connection (reference to memory)
            elif self.memory_graph and ref_memory_id in self.memory_graph.nodes:
                ref_memory = self.memory_graph.nodes[ref_memory_id]
                if memory.id in ref_memory.connections:
                    connection = ref_memory.connections[memory.id]
                    strength = connection.weight
                    max_connection_strength = max(max_connection_strength, strength)
                    weighted_connection_sum += strength
                    total_connections += 1
        
        if total_connections == 0:
            return 0.0
        
        # Calculate score based on both maximum strength and average strength
        average_strength = weighted_connection_sum / total_connections
        
        # Give more weight to maximum strength (strongest single connection)
        # but also consider average to reward memories with multiple strong connections
        connection_strength_score = (max_connection_strength * 0.7) + (average_strength * 0.3)
        
        # Boost score for memories with multiple strong connections
        if total_connections > 1:
            multi_connection_boost = min(0.2, (total_connections - 1) * 0.05)
            connection_strength_score = min(1.0, connection_strength_score + multi_connection_boost)
        
        # Apply connection learning bonus - stronger connections get even more weight
        # as the system learns more about connection patterns
        if self.connection_modification_count > 0:
            learning_factor = min(1.5, 1.0 + (self.connection_modification_count * 0.02))
            connection_strength_score = min(1.0, connection_strength_score * learning_factor)
        
        return connection_strength_score
    
    def _call_claude_api(self, prompt: str) -> str:
        """Make a call to Claude API with timeout and error handling."""
        # This will be implemented based on the specific Claude client available
        # For now, using a placeholder that would integrate with anthropic client
        
        if hasattr(self.claude_client, 'messages') and hasattr(self.claude_client.messages, 'create'):
            # Anthropic client
            try:
                message = self.claude_client.messages.create(
                    model="claude-3-haiku-20240307",  # Use fast model for relevance scoring
                    max_tokens=100,
                    timeout=self.claude_timeout,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text if message.content else "0.0 No response"
            except Exception as e:
                raise Exception(f"Anthropic API error: {e}")
        
        elif callable(self.claude_client):
            # Simple callable client
            try:
                return self.claude_client(prompt, timeout=self.claude_timeout)
            except Exception as e:
                raise Exception(f"Claude client error: {e}")
        
        else:
            raise Exception("Invalid Claude client configuration")
    
    def _parse_claude_response(self, response: str) -> float:
        """Parse Claude's response to extract the relevance score."""
        try:
            # Look for a number at the start of the response
            import re
            match = re.search(r'^(\d*\.?\d+)', response.strip())
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))  # Clamp to 0-1 range
            else:
                # Try to find any decimal number in the response
                numbers = re.findall(r'\b\d*\.?\d+\b', response)
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score))
        except (ValueError, AttributeError):
            pass
        
        # Default fallback
        return 0.5
    
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
        functional: float,
        associative: float,
        connection_strength: float,
        overall: float,
        must_keep: bool = False
    ) -> str:
        """Generate human-readable reasoning for the relevance score."""
        reasoning_parts = []
        
        # Add must-keep flag first if present
        if must_keep:
            reasoning_parts.append("MUST-KEEP")
        
        # Connection strength is now primary - mention it first if significant
        if connection_strength >= 0.7:
            reasoning_parts.append(f"Strong existing connections (weight: {self.connection_strength_weight:.1%})")
        elif connection_strength >= 0.4:
            reasoning_parts.append(f"Moderate existing connections (weight: {self.connection_strength_weight:.1%})")
        elif connection_strength > 0.1:
            reasoning_parts.append(f"Some existing connections (weight: {self.connection_strength_weight:.1%})")
        
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
        
        if functional >= 0.7:
            reasoning_parts.append("functionally aligned")
        elif functional >= 0.4:
            reasoning_parts.append("some functional relevance")
        
        if associative >= 0.7:
            reasoning_parts.append("strong AI-assessed associations")
        elif associative >= 0.4:
            reasoning_parts.append("moderate associative relevance")
        
        # Add connection learning status if relevant
        if self.connection_modification_count > 10:
            reasoning_parts.append(f"learned from {self.connection_modification_count} connection changes")
        
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
        threshold: float = None,
        reference_memory_ids: Optional[List[str]] = None
    ) -> List[Tuple[MemoryNode, RelevanceScore]]:
        """
        Filter memories by relevance threshold and return with scores.
        
        Args:
            memories: List of memories to evaluate
            query: Query string
            context: Query context
            threshold: Minimum relevance threshold (uses default if None)
            reference_memory_ids: List of memory IDs to check for connections
            
        Returns:
            List of (memory, relevance_score) tuples that meet threshold
        """
        threshold = threshold or self.relevance_threshold
        relevant_memories = []
        
        for memory in memories:
            score = self.evaluate_relevance(memory, query, context, reference_memory_ids)
            if score.overall >= threshold:
                relevant_memories.append((memory, score))
        
        # Sort by relevance score (highest first)
        relevant_memories.sort(key=lambda x: x[1].overall, reverse=True)
        
        return relevant_memories

def get_claude_client_from_env():
    api_key = os.getenv('CLAUDE_API_KEY')
    if not api_key:
        raise ValueError('CLAUDE_API_KEY not set in environment')
    return Anthropic(api_key=api_key)