"""
Filter Agent - The Decision Maker for Memory Selection

This agent serves as the final gatekeeper for what memories the user actually sees.
It makes UX-focused decisions about which memories to include in responses,
handling redundancy elimination, diversity, length constraints, and user preferences.
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

from src.core.memory_node import MemoryNode
from src.agents.relevance_agent import RelevanceScore, QueryContext


class FilterReason(Enum):
    """Reasons why memories might be filtered out or included."""
    RELEVANCE_THRESHOLD = "relevance_threshold"
    REDUNDANCY = "redundancy"
    DIVERSITY = "diversity"
    LENGTH_CONSTRAINT = "length_constraint"
    USER_PREFERENCE = "user_preference"
    PLATFORM_CONSTRAINT = "platform_constraint"
    COHERENCE = "coherence"


@dataclass
class UserPreferences:
    """User preferences for memory filtering."""
    max_memories: int = 10           # Increased from 5 to 10
    prefer_recent: bool = True
    avoid_redundancy: bool = True
    relevance_threshold: float = 0.3  # Lowered from 0.5 to 0.3 for more generous filtering
    diversity_factor: float = 0.7  # 0.0 = no diversity, 1.0 = maximum diversity
    preferred_types: List[str] = field(default_factory=list)
    blocked_types: List[str] = field(default_factory=list)


@dataclass
class ResponseContext:
    """Context about the response being generated."""
    response_type: str = "chat"  # chat, search, browse, etc.
    user_context: str = "general"  # mobile, desktop, learning, work, etc.
    conversation_history: List[str] = field(default_factory=list)
    platform: str = "web"  # web, mobile, api
    max_response_length: Optional[int] = None


@dataclass
class FilterResult:
    """Result of the filtering operation."""
    selected_memories: List[MemoryNode]
    relevance_scores: List[RelevanceScore]
    reasoning: str
    filtered_count: int
    filter_details: Dict[str, int] = field(default_factory=dict)


class FilterAgent:
    """
    Intelligent agent for filtering and selecting final memories for user responses.
    
    This agent focuses on USER EXPERIENCE optimization:
    - Eliminates redundancy
    - Ensures diversity
    - Respects length constraints
    - Applies user preferences
    - Maintains response coherence
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Filter Agent.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Similarity thresholds for redundancy detection
        self.redundancy_threshold = self.config.get('redundancy_threshold', 0.7)
        self.diversity_boost = self.config.get('diversity_boost', 0.2)
        
        # Platform-specific constraints - made more generous
        self.platform_limits = {
            'mobile': {'max_memories': 5, 'prefer_short': True},    # Increased from 2 to 5
            'web': {'max_memories': 15, 'prefer_short': False},     # Increased from 5 to 15
            'api': {'max_memories': 25, 'prefer_short': False}      # Increased from 10 to 25
        }
    
    def filter_for_response(
        self,
        candidate_memories: List[MemoryNode],
        relevance_scores: List[RelevanceScore],
        user_preferences: UserPreferences,
        response_context: ResponseContext
    ) -> FilterResult:
        """
        Filter candidate memories for final response inclusion.
        
        Args:
            candidate_memories: List of candidate memories
            relevance_scores: Corresponding relevance scores
            user_preferences: User filtering preferences
            response_context: Context about the response
            
        Returns:
            FilterResult with selected memories and reasoning
        """
        if len(candidate_memories) != len(relevance_scores):
            raise ValueError("Memories and scores lists must have same length")
        
        # Combine memories with their scores
        memory_score_pairs = list(zip(candidate_memories, relevance_scores))
        
        # Apply filtering pipeline
        result = self._apply_filtering_pipeline(
            memory_score_pairs, user_preferences, response_context
        )
        
        return result
    
    def _apply_filtering_pipeline(
        self,
        memory_score_pairs: List[Tuple[MemoryNode, RelevanceScore]],
        user_preferences: UserPreferences,
        response_context: ResponseContext
    ) -> FilterResult:
        """Apply the complete filtering pipeline."""
        original_count = len(memory_score_pairs)
        filter_stats = {}
        reasoning_parts = []
        
        # Step 0: Separate must-keep memories
        must_keep_pairs = [(memory, score) for memory, score in memory_score_pairs if score.must_keep]
        regular_pairs = [(memory, score) for memory, score in memory_score_pairs if not score.must_keep]
        
        if must_keep_pairs:
            reasoning_parts.append(f"Protected {len(must_keep_pairs)} must-keep memories")
        
        # Step 1: Relevance threshold filtering (only for regular memories)
        regular_pairs = self._filter_by_relevance_threshold(
            regular_pairs, user_preferences.relevance_threshold
        )
        filter_stats['relevance_threshold'] = len(memory_score_pairs) - len(must_keep_pairs) - len(regular_pairs)
        if filter_stats['relevance_threshold'] > 0:
            reasoning_parts.append(
                f"Filtered {filter_stats['relevance_threshold']} memories below relevance threshold"
            )
        
        # Step 2: Platform constraints (apply to regular memories only)
        platform_limit = self._get_platform_limit(response_context.platform, user_preferences.max_memories)
        # Reserve space for must-keep memories
        available_slots = max(1, platform_limit - len(must_keep_pairs))
        
        if len(regular_pairs) > available_slots:
            regular_pairs = regular_pairs[:available_slots]
            filter_stats['platform_constraint'] = len(regular_pairs) - available_slots
            reasoning_parts.append(f"Applied {response_context.platform} platform limits")
        
        # Step 3: Redundancy elimination (regular memories only)
        if user_preferences.avoid_redundancy and len(regular_pairs) > 1:
            before_count = len(regular_pairs)
            regular_pairs = self._remove_redundant_memories(regular_pairs)
            filter_stats['redundancy'] = before_count - len(regular_pairs)
            if filter_stats['redundancy'] > 0:
                reasoning_parts.append(f"Removed {filter_stats['redundancy']} redundant memories")
        
        # Step 4: Diversity optimization (regular memories only)
        if user_preferences.diversity_factor > 0.5 and len(regular_pairs) > 1:
            regular_pairs = self._optimize_for_diversity(
                regular_pairs, user_preferences.diversity_factor
            )
            reasoning_parts.append("Optimized for topic diversity")
        
        # Step 5: User preferences (regular memories only)
        regular_pairs = self._apply_user_preferences(
            regular_pairs, user_preferences
        )
        
        # Step 6: Combine must-keep and regular memories
        final_pairs = must_keep_pairs + regular_pairs
        
        # Step 7: Final count limit (only reduce regular memories if needed)
        if len(final_pairs) > user_preferences.max_memories:
            # Keep all must-keep memories, reduce regular memories
            excess = len(final_pairs) - user_preferences.max_memories
            regular_pairs = regular_pairs[:-excess] if excess <= len(regular_pairs) else []
            final_pairs = must_keep_pairs + regular_pairs
            reasoning_parts.append(f"Limited to {user_preferences.max_memories} memories (protected must-keep)")
        
        # Extract final results
        selected_memories = [pair[0] for pair in final_pairs]
        selected_scores = [pair[1] for pair in final_pairs]
        
        # Generate reasoning
        if not reasoning_parts:
            reasoning = f"Selected {len(selected_memories)} relevant memories"
        else:
            reasoning = "; ".join(reasoning_parts) + f". Final selection: {len(selected_memories)} memories"
        
        return FilterResult(
            selected_memories=selected_memories,
            relevance_scores=selected_scores,
            reasoning=reasoning,
            filtered_count=original_count - len(selected_memories),
            filter_details=filter_stats
        )
    
    def _filter_by_relevance_threshold(
        self,
        memory_score_pairs: List[Tuple[MemoryNode, RelevanceScore]],
        threshold: float
    ) -> List[Tuple[MemoryNode, RelevanceScore]]:
        """Filter memories by relevance threshold."""
        return [
            (memory, score) for memory, score in memory_score_pairs
            if score.overall >= threshold
        ]
    
    def _get_platform_limit(self, platform: str, user_max: int) -> int:
        """Get platform-specific memory limit."""
        platform_config = self.platform_limits.get(platform, self.platform_limits['web'])
        return min(user_max, platform_config['max_memories'])
    
    def _remove_redundant_memories(
        self,
        memory_score_pairs: List[Tuple[MemoryNode, RelevanceScore]]
    ) -> List[Tuple[MemoryNode, RelevanceScore]]:
        """Remove redundant memories based on content similarity."""
        if len(memory_score_pairs) <= 1:
            return memory_score_pairs
        
        # Sort by relevance score (keep higher scoring memories)
        sorted_pairs = sorted(memory_score_pairs, key=lambda x: x[1].overall, reverse=True)
        
        selected = []
        for memory, score in sorted_pairs:
            is_redundant = False
            
            for selected_memory, _ in selected:
                similarity = self._calculate_memory_similarity(memory, selected_memory)
                if similarity >= self.redundancy_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected.append((memory, score))
        
        return selected
    
    def _calculate_memory_similarity(self, memory1: MemoryNode, memory2: MemoryNode) -> float:
        """Calculate similarity between two memories for redundancy detection."""
        # Concept similarity
        concept_sim = self._text_similarity(memory1.concept, memory2.concept)
        
        # Tag overlap - more generous scoring
        tags1 = set(tag.lower() for tag in memory1.tags)
        tags2 = set(tag.lower() for tag in memory2.tags)
        if tags1 and tags2:
            tag_intersection = tags1 & tags2
            tag_overlap = len(tag_intersection) / min(len(tags1), len(tags2))
        else:
            tag_overlap = 0.0
        
        # Keyword overlap - more generous scoring
        keywords1 = set(word.lower() for word in memory1.keywords)
        keywords2 = set(word.lower() for word in memory2.keywords)
        if keywords1 and keywords2:
            keyword_intersection = keywords1 & keywords2
            keyword_overlap = len(keyword_intersection) / min(len(keywords1), len(keywords2))
        else:
            keyword_overlap = 0.0
        
        # Summary similarity
        summary_sim = self._text_similarity(memory1.summary, memory2.summary)
        
        # Content similarity for very similar memories
        content_sim = self._text_similarity(memory1.full_content, memory2.full_content)
        
        # Weighted combination - emphasize overlap metrics for redundancy
        similarity = max(
            concept_sim * 0.3 + tag_overlap * 0.3 + keyword_overlap * 0.25 + summary_sim * 0.15,
            content_sim * 0.8,  # High content similarity is strong indicator
            tag_overlap * 0.6 + keyword_overlap * 0.4  # High tag+keyword overlap
        )
        
        return min(1.0, similarity)
    
    def _optimize_for_diversity(
        self,
        memory_score_pairs: List[Tuple[MemoryNode, RelevanceScore]],
        diversity_factor: float
    ) -> List[Tuple[MemoryNode, RelevanceScore]]:
        """Optimize memory selection for topic diversity."""
        if len(memory_score_pairs) <= 2:
            return memory_score_pairs
        
        # Track selected topics
        selected_topics = set()
        diversified = []
        
        # Sort by relevance but apply diversity boost
        for memory, score in memory_score_pairs:
            # Get primary topic (first tag or concept words)
            primary_topic = memory.tags[0].lower() if memory.tags else memory.concept.lower().split()[0]
            
            # Apply diversity penalty if topic already represented
            diversity_penalty = 0.0
            if primary_topic in selected_topics:
                diversity_penalty = diversity_factor * 0.2
            
            # Adjusted score
            adjusted_score = score.overall - diversity_penalty
            
            diversified.append((memory, score, adjusted_score, primary_topic))
            selected_topics.add(primary_topic)
        
        # Sort by adjusted score and return original format
        diversified.sort(key=lambda x: x[2], reverse=True)
        return [(memory, score) for memory, score, _, _ in diversified]
    
    def _apply_user_preferences(
        self,
        memory_score_pairs: List[Tuple[MemoryNode, RelevanceScore]],
        user_preferences: UserPreferences
    ) -> List[Tuple[MemoryNode, RelevanceScore]]:
        """Apply user-specific preferences."""
        filtered = []
        
        for memory, score in memory_score_pairs:
            # Check blocked types
            if any(blocked in memory.tags for blocked in user_preferences.blocked_types):
                continue
            
            # Boost preferred types
            boost = 0.0
            if any(preferred in memory.tags for preferred in user_preferences.preferred_types):
                boost = 0.1
            
            # Boost recent memories if preferred
            if user_preferences.prefer_recent and memory.access_count > 0:
                boost += 0.05
            
            # Create adjusted score
            adjusted_score = RelevanceScore(
                overall=min(1.0, score.overall + boost),
                confidence=score.confidence,
                semantic_score=score.semantic_score,
                context_score=score.context_score,
                temporal_score=score.temporal_score,
                topical_score=score.topical_score,
                reasoning=score.reasoning
            )
            
            filtered.append((memory, adjusted_score))
        
        # Sort by adjusted relevance
        filtered.sort(key=lambda x: x[1].overall, reverse=True)
        return filtered
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on word overlap."""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_filter_explanation(self, filter_result: FilterResult) -> str:
        """Generate detailed explanation of filtering decisions."""
        explanation_parts = [
            f"Filter Agent processed {filter_result.filtered_count + len(filter_result.selected_memories)} candidate memories"
        ]
        
        if filter_result.filter_details:
            for reason, count in filter_result.filter_details.items():
                if count > 0:
                    explanation_parts.append(f"- Filtered {count} for {reason.replace('_', ' ')}")
        
        explanation_parts.append(f"Final selection: {len(filter_result.selected_memories)} memories")
        explanation_parts.append(f"Decision reasoning: {filter_result.reasoning}")
        
        return "\n".join(explanation_parts)