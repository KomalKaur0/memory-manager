"""
Unified Model Client Interface
Provides a consistent interface for different AI models (Claude, Gemini)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from enum import Enum
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Available model providers."""
    CLAUDE = "claude"
    GEMINI = "gemini"
    MOCK = "mock"


@dataclass
class ModelResponse:
    """Standardized response from any model."""
    content: str
    model: str
    provider: ModelProvider
    usage: Optional[Dict] = None
    error: Optional[str] = None


class BaseModelClient(ABC):
    """Abstract base class for model clients."""
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 100) -> ModelResponse:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available/configured."""
        pass
    
    @property
    @abstractmethod
    def provider(self) -> ModelProvider:
        """Get the model provider."""
        pass


class ClaudeClient(BaseModelClient):
    """Claude API client wrapper."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('CLAUDE_API_KEY')
        self._client = None
        
        if self.api_key:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("Anthropic package not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {e}")
    
    def generate_response(self, prompt: str, max_tokens: int = 100) -> ModelResponse:
        """Generate response using Claude."""
        if not self._client:
            return ModelResponse(
                content="",
                model="claude-3-haiku-20240307",
                provider=ModelProvider.CLAUDE,
                error="Claude client not initialized"
            )
        
        try:
            message = self._client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = message.content[0].text if message.content else ""
            usage = {
                "input_tokens": getattr(message.usage, 'input_tokens', 0),
                "output_tokens": getattr(message.usage, 'output_tokens', 0)
            } if hasattr(message, 'usage') else None
            
            return ModelResponse(
                content=content,
                model="claude-3-haiku-20240307",
                provider=ModelProvider.CLAUDE,
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return ModelResponse(
                content="",
                model="claude-3-haiku-20240307",
                provider=ModelProvider.CLAUDE,
                error=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if Claude is available."""
        return self._client is not None
    
    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.CLAUDE


class GeminiClient(BaseModelClient):
    """Gemini API client wrapper."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self._client = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel('gemini-1.5-flash')
            except ImportError:
                logger.error("google-generativeai package not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
    
    def generate_response(self, prompt: str, max_tokens: int = 100) -> ModelResponse:
        """Generate response using Gemini."""
        if not self._client:
            return ModelResponse(
                content="",
                model="gemini-1.5-flash",
                provider=ModelProvider.GEMINI,
                error="Gemini client not initialized"
            )
        
        try:
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.1
                }
            )
            
            content = response.text if response.text else ""
            usage = {
                "input_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "output_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0)
            } if hasattr(response, 'usage_metadata') else None
            
            return ModelResponse(
                content=content,
                model="gemini-1.5-flash",
                provider=ModelProvider.GEMINI,
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return ModelResponse(
                content="",
                model="gemini-1.5-flash",
                provider=ModelProvider.GEMINI,
                error=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return self._client is not None
    
    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.GEMINI


class MockModelClient(BaseModelClient):
    """Mock model client for testing."""
    
    def __init__(self):
        from src.agents.mock_claude_client import MockClaudeClient
        self._mock_client = MockClaudeClient()
    
    def generate_response(self, prompt: str, max_tokens: int = 100) -> ModelResponse:
        """Generate mock response."""
        try:
            content = self._mock_client._generate_response(prompt)
            
            return ModelResponse(
                content=content,
                model="mock-model",
                provider=ModelProvider.MOCK,
                usage={"input_tokens": len(prompt.split()), "output_tokens": len(content.split())}
            )
            
        except Exception as e:
            logger.error(f"Mock client error: {e}")
            return ModelResponse(
                content="0.5 Mock response generation failed",
                model="mock-model",
                provider=ModelProvider.MOCK,
                error=str(e)
            )
    
    def is_available(self) -> bool:
        """Mock is always available."""
        return True
    
    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.MOCK


class UnifiedModelClient:
    """
    Unified client that can switch between different model providers.
    """
    
    def __init__(self, preferred_provider: Union[str, ModelProvider] = ModelProvider.CLAUDE):
        if isinstance(preferred_provider, str):
            preferred_provider = ModelProvider(preferred_provider.lower())
        
        self.preferred_provider = preferred_provider
        self.clients = {
            ModelProvider.CLAUDE: ClaudeClient(),
            ModelProvider.GEMINI: GeminiClient(),
            ModelProvider.MOCK: MockModelClient()
        }
        self.current_client = self._get_available_client()
        
        logger.info(f"Initialized UnifiedModelClient with preferred provider: {preferred_provider.value}, "
                   f"using: {self.current_client.provider.value}")
    
    def _get_available_client(self) -> BaseModelClient:
        """Get the best available client."""
        # Try preferred provider first
        if self.clients[self.preferred_provider].is_available():
            return self.clients[self.preferred_provider]
        
        # Fall back to any available client
        for provider, client in self.clients.items():
            if client.is_available():
                logger.warning(f"Preferred provider {self.preferred_provider.value} not available, "
                              f"falling back to {provider.value}")
                return client
        
        # Final fallback to mock
        logger.warning("No AI providers available, using mock client")
        return self.clients[ModelProvider.MOCK]
    
    def switch_provider(self, provider: Union[str, ModelProvider]) -> bool:
        """
        Switch to a different model provider.
        
        Args:
            provider: The provider to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        if isinstance(provider, str):
            try:
                provider = ModelProvider(provider.lower())
            except ValueError:
                logger.error(f"Invalid provider: {provider}")
                return False
        
        if provider not in self.clients:
            logger.error(f"Provider {provider.value} not supported")
            return False
        
        client = self.clients[provider]
        if not client.is_available():
            logger.error(f"Provider {provider.value} not available")
            return False
        
        old_provider = self.current_client.provider.value
        self.current_client = client
        self.preferred_provider = provider
        
        logger.info(f"Switched from {old_provider} to {provider.value}")
        return True
    
    def generate_response(self, prompt: str, max_tokens: int = 100) -> ModelResponse:
        """Generate response using current client."""
        return self.current_client.generate_response(prompt, max_tokens)
    
    def get_current_provider(self) -> ModelProvider:
        """Get the currently active provider."""
        return self.current_client.provider
    
    def get_available_providers(self) -> List[ModelProvider]:
        """Get list of available providers."""
        return [provider for provider, client in self.clients.items() if client.is_available()]
    
    def is_provider_available(self, provider: Union[str, ModelProvider]) -> bool:
        """Check if a specific provider is available."""
        if isinstance(provider, str):
            try:
                provider = ModelProvider(provider.lower())
            except ValueError:
                return False
        
        return provider in self.clients and self.clients[provider].is_available()


def get_model_client_from_env(preferred_provider: Optional[str] = None) -> UnifiedModelClient:
    """
    Create a UnifiedModelClient with configuration from environment variables.
    
    Args:
        preferred_provider: Preferred model provider ('claude', 'gemini', or 'mock')
        
    Returns:
        Configured UnifiedModelClient instance
    """
    # Get preferred provider from environment or parameter
    env_provider = os.getenv('AI_MODEL_PROVIDER', 'claude').lower()
    provider = preferred_provider or env_provider
    
    try:
        provider_enum = ModelProvider(provider)
    except ValueError:
        logger.warning(f"Invalid provider '{provider}', defaulting to Claude")
        provider_enum = ModelProvider.CLAUDE
    
    return UnifiedModelClient(preferred_provider=provider_enum)