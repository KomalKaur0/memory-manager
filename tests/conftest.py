"""
Shared pytest fixtures and configuration
"""
import os
import pytest
from unittest.mock import patch, MagicMock

# Set test environment variables before any imports
os.environ['VOYAGER_LITE_API_KEY'] = 'test-api-key'
os.environ['CLAUDE_API_KEY'] = 'test-claude-key'

@pytest.fixture(autouse=True)
def mock_environment():
    """Automatically mock environment for all tests"""
    with patch.dict(os.environ, {
        'VOYAGER_LITE_API_KEY': 'test-api-key',
        'CLAUDE_API_KEY': 'test-claude-key'
    }):
        yield

@pytest.fixture
def mock_embedding_config():
    """Mock embedding configuration"""
    return {
        'api_key': 'test-api-key',
        'model_name': 'test-model',
        'batch_size': 32,
        'embedding_dim': 384
    }

@pytest.fixture(autouse=True)
def mock_get_embedding_config(mock_embedding_config):
    """Mock the get_embedding_config_from_env function"""
    with patch('src.retrieval.embedding_search.get_embedding_config_from_env') as mock:
        mock.return_value = mock_embedding_config
        yield mock

@pytest.fixture(autouse=True)
def mock_get_claude_client():
    """Mock the get_claude_client_from_env function"""
    with patch('src.agents.relevance_agent.get_claude_client_from_env') as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock