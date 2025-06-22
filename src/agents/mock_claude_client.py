"""
Mock Claude Client for testing associative relevance scoring.
This provides a fallback when no real Claude API is available.
"""
import re
import random
from typing import Dict, List


class MockClaudeClient:
    """Mock Claude client that provides intelligent-ish responses for testing."""
    
    def __init__(self):
        self.response_patterns = self._build_response_patterns()
    
    def _build_response_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for generating realistic responses."""
        return {
            'high_relevance': [
                "0.8 Directly addresses the query with useful implementation details",
                "0.9 Perfect match for the user's learning needs in this domain",
                "0.7 Strong conceptual overlap with clear practical applications",
                "0.8 Excellent contextual fit given recent conversation topics"
            ],
            'moderate_relevance': [
                "0.5 Provides useful background context for the current discussion",
                "0.6 Related concepts that could enhance understanding",
                "0.4 Some connection to the query but not directly applicable",
                "0.5 Complementary information that supports the main topic"
            ],
            'low_relevance': [
                "0.2 Minimal connection to current query or conversation",
                "0.1 Different domain with little applicable knowledge",
                "0.3 Tangentially related but unlikely to be immediately useful",
                "0.2 Background information with limited direct relevance"
            ]
        }
    
    def __call__(self, prompt: str, timeout: float = 10.0) -> str:
        """Simulate calling Claude with a prompt."""
        return self._generate_response(prompt)
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a contextually appropriate response based on the prompt."""
        # Extract key information from the prompt
        query = self._extract_query(prompt)
        memory_concept = self._extract_memory_concept(prompt)
        conversation_context = self._extract_conversation_context(prompt)
        
        # Determine relevance category based on content analysis
        relevance_category = self._assess_relevance_category(query, memory_concept, conversation_context)
        
        # Select and return appropriate response
        responses = self.response_patterns[relevance_category]
        return random.choice(responses)
    
    def _extract_query(self, prompt: str) -> str:
        """Extract the current query from the prompt."""
        match = re.search(r'Current Query: (.+?)(?:\n|$)', prompt)
        return match.group(1).strip() if match else ""
    
    def _extract_memory_concept(self, prompt: str) -> str:
        """Extract the memory concept from the prompt."""
        match = re.search(r'- Concept: (.+?)(?:\n|$)', prompt)
        return match.group(1).strip() if match else ""
    
    def _extract_conversation_context(self, prompt: str) -> str:
        """Extract conversation context from the prompt."""
        match = re.search(r'Recent conversation: (.+?)(?:\n|$)', prompt)
        return match.group(1).strip() if match else ""
    
    def _assess_relevance_category(self, query: str, memory_concept: str, conversation_context: str) -> str:
        """Assess which relevance category this combination should fall into."""
        query_words = set(re.findall(r'\w+', query.lower()))
        concept_words = set(re.findall(r'\w+', memory_concept.lower()))
        context_words = set(re.findall(r'\w+', conversation_context.lower())) if conversation_context else set()
        
        # Calculate word overlap
        query_concept_overlap = len(query_words & concept_words) / max(len(query_words), 1)
        context_overlap = len(context_words & concept_words) / max(len(context_words), 1) if context_words else 0
        
        # Check for domain alignment
        domains = {
            'programming': ['python', 'code', 'async', 'function', 'programming', 'development'],
            'database': ['database', 'sql', 'query', 'data', 'storage'],
            'web': ['web', 'html', 'css', 'javascript', 'api', 'http'],
            'system': ['system', 'performance', 'memory', 'cpu', 'optimization']
        }
        
        domain_match = False
        for domain_words in domains.values():
            if any(word in query_words for word in domain_words) and any(word in concept_words for word in domain_words):
                domain_match = True
                break
        
        # Determine category
        if query_concept_overlap >= 0.3 or (domain_match and query_concept_overlap >= 0.2):
            return 'high_relevance'
        elif query_concept_overlap >= 0.1 or context_overlap >= 0.2 or domain_match:
            return 'moderate_relevance'
        else:
            return 'low_relevance'


# For compatibility with different client interfaces
class MockAnthropicClient:
    """Mock Anthropic client with messages.create interface."""
    
    def __init__(self):
        self.messages = self
        self.mock_client = MockClaudeClient()
    
    def create(self, model: str, max_tokens: int, timeout: float, messages: List[Dict]) -> 'MockMessage':
        """Simulate the Anthropic messages.create method."""
        # Extract the user message
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Generate response
        response_text = self.mock_client._generate_response(user_message)
        
        return MockMessage(response_text)


class MockMessage:
    """Mock message response object."""
    
    def __init__(self, text: str):
        self.content = [MockContent(text)]


class MockContent:
    """Mock content object."""
    
    def __init__(self, text: str):
        self.text = text