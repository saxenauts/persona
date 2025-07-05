"""
Integration tests for LLM client system with real providers.
These tests run only when API keys are available.
"""

import pytest
import os
from unittest.mock import patch
from persona.llm.client_factory import (
    get_chat_client, 
    get_embedding_client, 
    reset_clients,
    parse_llm_service
)
from persona.llm.providers.base import ChatMessage


@pytest.mark.skipif(
    not os.getenv("AZURE_API_KEY") or not os.getenv("AZURE_API_BASE"), 
    reason="Azure API credentials not available"
)
class TestAzureIntegration:
    """Integration tests for Azure OpenAI"""
    
    def setup_method(self):
        """Reset clients before each test"""
        reset_clients()
    
    @patch('persona.llm.client_factory.config')
    @pytest.mark.asyncio
    async def test_azure_chat_integration(self, mock_config):
        """Test Azure chat with real API (if credentials available)"""
        # Configure for Azure
        mock_config.MACHINE_LEARNING.LLM_SERVICE = "azure/gpt-4o-mini"
        mock_config.MACHINE_LEARNING.AZURE_API_KEY = os.getenv("AZURE_API_KEY")
        mock_config.MACHINE_LEARNING.AZURE_API_BASE = os.getenv("AZURE_API_BASE")
        mock_config.MACHINE_LEARNING.AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")
        mock_config.MACHINE_LEARNING.AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")
        mock_config.MACHINE_LEARNING.AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        
        client = get_chat_client()
        assert client.get_provider_name() == "azure"
        
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Say 'Hello from Azure!' and nothing else.")
        ]
        
        response = await client.chat(messages, temperature=0.1)
        
        assert response.content is not None
        assert len(response.content) > 0
        assert "azure" in response.model.lower()
        print(f"Azure response: {response.content}")
    
    @patch('persona.llm.client_factory.config')
    @pytest.mark.asyncio
    async def test_azure_embeddings_integration(self, mock_config):
        """Test Azure embeddings with real API (if credentials available)"""
        # Configure for Azure
        mock_config.MACHINE_LEARNING.LLM_SERVICE = "azure/gpt-4o-mini"
        mock_config.MACHINE_LEARNING.AZURE_API_KEY = os.getenv("AZURE_API_KEY")
        mock_config.MACHINE_LEARNING.AZURE_API_BASE = os.getenv("AZURE_API_BASE")
        mock_config.MACHINE_LEARNING.AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")
        mock_config.MACHINE_LEARNING.AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")
        mock_config.MACHINE_LEARNING.AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        
        client = get_embedding_client()
        assert client.get_provider_name() == "azure"
        
        texts = ["Hello world", "This is a test"]
        embeddings = await client.embeddings(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536  # Standard embedding dimension
        assert len(embeddings[1]) == 1536
        assert all(isinstance(x, float) for x in embeddings[0])
        print(f"Azure embeddings generated: {len(embeddings)} vectors of dimension {len(embeddings[0])}")


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), 
    reason="OpenAI API key not available"
)
class TestOpenAIIntegration:
    """Integration tests for OpenAI"""
    
    def setup_method(self):
        """Reset clients before each test"""
        reset_clients()
    
    @patch('persona.llm.client_factory.config')
    @pytest.mark.asyncio
    async def test_openai_chat_integration(self, mock_config):
        """Test OpenAI chat with real API (if credentials available)"""
        # Configure for OpenAI
        mock_config.MACHINE_LEARNING.LLM_SERVICE = "openai/gpt-4o-mini"
        mock_config.MACHINE_LEARNING.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        mock_config.MACHINE_LEARNING.OPENAI_CHAT_MODEL = "gpt-4o-mini"
        mock_config.MACHINE_LEARNING.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        
        client = get_chat_client()
        assert client.get_provider_name() == "openai"
        
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Say 'Hello from OpenAI!' and nothing else.")
        ]
        
        response = await client.chat(messages, temperature=0.1)
        
        assert response.content is not None
        assert len(response.content) > 0
        print(f"OpenAI response: {response.content}")


class TestProviderSwitching:
    """Test switching between providers"""
    
    def setup_method(self):
        """Reset clients before each test"""
        reset_clients()
    
    def test_parse_llm_service_configurations(self):
        """Test parsing different LLM service configurations"""
        configs = [
            ("openai/gpt-4o-mini", "openai", "gpt-4o-mini"),
            ("azure/gpt-4o-mini", "azure", "gpt-4o-mini"),
            ("anthropic/claude-3-5-sonnet-20241022", "anthropic", "claude-3-5-sonnet-20241022"),
            ("gemini/gemini-1.5-flash", "gemini", "gemini-1.5-flash"),
        ]
        
        for service_string, expected_provider, expected_model in configs:
            provider, model = parse_llm_service(service_string)
            assert provider == expected_provider
            assert model == expected_model
    
    @patch('persona.llm.client_factory.config')
    def test_embedding_fallback_logic(self, mock_config):
        """Test that embedding client falls back to OpenAI when provider doesn't support embeddings"""
        # Configure for Anthropic (which doesn't support embeddings)
        mock_config.MACHINE_LEARNING.LLM_SERVICE = "anthropic/claude-3-5-sonnet-20241022"
        mock_config.MACHINE_LEARNING.ANTHROPIC_API_KEY = "test-key"
        mock_config.MACHINE_LEARNING.ANTHROPIC_CHAT_MODEL = "claude-3-5-sonnet-20241022"
        mock_config.MACHINE_LEARNING.OPENAI_API_KEY = "test-openai-key"
        mock_config.MACHINE_LEARNING.OPENAI_CHAT_MODEL = "gpt-4o-mini"
        mock_config.MACHINE_LEARNING.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        
        with patch('persona.llm.providers.anthropic_client.anthropic'), \
             patch('persona.llm.providers.openai_client.openai'):
            
            chat_client = get_chat_client()
            embedding_client = get_embedding_client()
            
            # Chat should use Anthropic
            assert chat_client.get_provider_name() == "anthropic"
            
            # Embeddings should fall back to OpenAI
            assert embedding_client.get_provider_name() == "openai" 