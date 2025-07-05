"""
Unit tests for LLM client system
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from persona.llm.client_factory import (
    parse_llm_service, 
    get_chat_client, 
    get_embedding_client,
    reset_clients,
    create_openai_client,
    create_azure_client
)
from persona.llm.providers.base import ChatMessage, ChatResponse
from persona.llm.providers.openai_client import OpenAIClient
from persona.llm.providers.azure_openai_client import AzureOpenAIClient


class TestClientFactory:
    
    def test_parse_llm_service_valid(self):
        """Test parsing valid LLM service strings"""
        provider, model = parse_llm_service("openai/gpt-4o-mini")
        assert provider == "openai"
        assert model == "gpt-4o-mini"
        
        provider, model = parse_llm_service("azure/gpt-4o-mini")
        assert provider == "azure"
        assert model == "gpt-4o-mini"
        
        provider, model = parse_llm_service("anthropic/claude-3-5-sonnet-20241022")
        assert provider == "anthropic"
        assert model == "claude-3-5-sonnet-20241022"
    
    def test_parse_llm_service_invalid(self):
        """Test parsing invalid LLM service strings"""
        with pytest.raises(ValueError, match="Invalid LLM_SERVICE format"):
            parse_llm_service("invalid-format")
    
    @patch('persona.llm.client_factory.config')
    def test_create_openai_client_success(self, mock_config):
        """Test successful OpenAI client creation"""
        mock_config.MACHINE_LEARNING.OPENAI_API_KEY = "test-key"
        mock_config.MACHINE_LEARNING.OPENAI_CHAT_MODEL = "gpt-4o-mini"
        mock_config.MACHINE_LEARNING.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        
        with patch('persona.llm.providers.openai_client.openai'):
            client = create_openai_client()
            assert isinstance(client, OpenAIClient)
            assert client.chat_model == "gpt-4o-mini"
            assert client.embedding_model == "text-embedding-3-small"
    
    @patch('persona.llm.client_factory.config')
    def test_create_openai_client_missing_key(self, mock_config):
        """Test OpenAI client creation with missing API key"""
        mock_config.MACHINE_LEARNING.OPENAI_API_KEY = ""
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            create_openai_client()
    
    @patch('persona.llm.client_factory.config')
    def test_create_azure_client_success(self, mock_config):
        """Test successful Azure client creation"""
        mock_config.MACHINE_LEARNING.AZURE_API_KEY = "test-key"
        mock_config.MACHINE_LEARNING.AZURE_API_BASE = "https://test.openai.azure.com"
        mock_config.MACHINE_LEARNING.AZURE_API_VERSION = "2024-02-01"
        mock_config.MACHINE_LEARNING.AZURE_CHAT_DEPLOYMENT = "gpt-4o-mini"
        mock_config.MACHINE_LEARNING.AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
        
        with patch('persona.llm.providers.azure_openai_client.openai'):
            client = create_azure_client()
            assert isinstance(client, AzureOpenAIClient)
            assert client.chat_deployment == "gpt-4o-mini"
            assert client.embedding_deployment == "text-embedding-3-small"
    
    @patch('persona.llm.client_factory.config')
    def test_create_azure_client_missing_key(self, mock_config):
        """Test Azure client creation with missing API key"""
        mock_config.MACHINE_LEARNING.AZURE_API_KEY = ""
        mock_config.MACHINE_LEARNING.AZURE_API_BASE = "https://test.openai.azure.com"
        
        with pytest.raises(ValueError, match="AZURE_API_KEY is required"):
            create_azure_client()
    
    @patch('persona.llm.client_factory.config')
    def test_get_chat_client_caching(self, mock_config):
        """Test that chat client is cached properly"""
        mock_config.MACHINE_LEARNING.LLM_SERVICE = "openai/gpt-4o-mini"
        mock_config.MACHINE_LEARNING.OPENAI_API_KEY = "test-key"
        mock_config.MACHINE_LEARNING.OPENAI_CHAT_MODEL = "gpt-4o-mini"
        mock_config.MACHINE_LEARNING.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        
        reset_clients()  # Ensure clean state
        
        with patch('persona.llm.providers.openai_client.openai'):
            client1 = get_chat_client()
            client2 = get_chat_client()
            
            # Should return the same instance due to caching
            assert client1 is client2
    
    @patch('persona.llm.client_factory.config')
    def test_get_embedding_client_fallback(self, mock_config):
        """Test that embedding client falls back to OpenAI when provider doesn't support embeddings"""
        mock_config.MACHINE_LEARNING.LLM_SERVICE = "anthropic/claude-3-5-sonnet-20241022"
        mock_config.MACHINE_LEARNING.ANTHROPIC_API_KEY = "test-key"
        mock_config.MACHINE_LEARNING.ANTHROPIC_CHAT_MODEL = "claude-3-5-sonnet-20241022"
        mock_config.MACHINE_LEARNING.OPENAI_API_KEY = "test-openai-key"
        mock_config.MACHINE_LEARNING.OPENAI_CHAT_MODEL = "gpt-4o-mini"
        mock_config.MACHINE_LEARNING.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        
        reset_clients()  # Ensure clean state
        
        with patch('persona.llm.providers.anthropic_client.anthropic'), \
             patch('persona.llm.providers.openai_client.openai'):
            client = get_embedding_client()
            # Should fall back to OpenAI since Anthropic doesn't support embeddings
            assert isinstance(client, OpenAIClient)


class TestBaseLLMClient:
    
    @pytest.mark.asyncio
    async def test_openai_client_chat(self):
        """Test OpenAI client chat functionality"""
        with patch('persona.llm.providers.openai_client.openai') as mock_openai:
            # Mock the async client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.model = "gpt-4o-mini"
            mock_response.usage = None
            
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create.return_value = mock_response
            mock_openai.AsyncOpenAI.return_value = mock_async_client
            
            client = OpenAIClient(api_key="test-key")
            
            messages = [
                ChatMessage(role="system", content="You are a helpful assistant"),
                ChatMessage(role="user", content="Hello")
            ]
            
            response = await client.chat(messages)
            
            assert isinstance(response, ChatResponse)
            assert response.content == "Test response"
            assert response.model == "gpt-4o-mini"
            
            # Verify the API was called correctly
            mock_async_client.chat.completions.create.assert_called_once()
            call_args = mock_async_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4o-mini"
            assert len(call_args[1]["messages"]) == 2
    
    @pytest.mark.asyncio
    async def test_openai_client_embeddings(self):
        """Test OpenAI client embeddings functionality"""
        with patch('persona.llm.providers.openai_client.openai') as mock_openai:
            # Mock the sync client for embeddings
            mock_response = MagicMock()
            mock_response.data = [MagicMock(), MagicMock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3]
            mock_response.data[1].embedding = [0.4, 0.5, 0.6]
            
            mock_sync_client = MagicMock()
            mock_sync_client.embeddings.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_sync_client
            
            client = OpenAIClient(api_key="test-key")
            
            texts = ["Hello", "World"]
            embeddings = await client.embeddings(texts)
            
            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]
            
            # Verify the API was called correctly
            mock_sync_client.embeddings.create.assert_called_once()
            call_args = mock_sync_client.embeddings.create.call_args
            assert call_args[1]["input"] == ["Hello", "World"]
    
    def test_provider_capabilities(self):
        """Test provider capability reporting"""
        with patch('persona.llm.providers.openai_client.openai'):
            openai_client = OpenAIClient(api_key="test-key")
            assert openai_client.supports_json_mode() is True
            assert openai_client.supports_embeddings() is True
            assert openai_client.get_provider_name() == "openai"
        
        with patch('persona.llm.providers.azure_openai_client.openai'):
            azure_client = AzureOpenAIClient(
                api_key="test-key", 
                api_base="https://test.openai.azure.com"
            )
            assert azure_client.supports_json_mode() is True
            assert azure_client.supports_embeddings() is True
            assert azure_client.get_provider_name() == "azure" 