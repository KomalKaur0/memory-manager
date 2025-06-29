"""
AI Memory System Agents

This package contains intelligent agents for memory management:
- RelevanceAgent: Evaluates memory relevance for queries and contexts
- FilterAgent: Makes final decisions about which memories to show users
- ConnectionAgent: Manages memory connections and strengthening
"""

from .relevance_agent import (
    RelevanceAgent,
    RelevanceScore,
    QueryContext,
    RelevanceType,
    QueryType
)

from .filter_agent import (
    FilterAgent,
    FilterResult,
    UserPreferences,
    ResponseContext,
    FilterReason
)

from .connection_agent import (
    ConnectionAgent,
    ConnectionSuggestion,
    ConnectionStrengthening,
    AccessEvent,
    ConnectionAnalysisType
)

__all__ = [
    'RelevanceAgent',
    'RelevanceScore', 
    'QueryContext',
    'RelevanceType',
    'QueryType',
    'FilterAgent',
    'FilterResult',
    'UserPreferences',
    'ResponseContext',
    'FilterReason',
    'ConnectionAgent',
    'ConnectionSuggestion',
    'ConnectionStrengthening',
    'AccessEvent',
    'ConnectionAnalysisType'
]