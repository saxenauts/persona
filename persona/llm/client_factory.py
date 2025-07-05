"""
LLM Client Factory for managing multiple LLM service providers.
"""

from typing import Optional, Tuple
from .providers.base import BaseLLMClient
from .providers.openai_client import OpenAIClient
from .providers.azure_openai_client import AzureOpenAIClient
from .providers.anthropic_client import AnthropicClient
from .providers.gemini_client import GeminiClient
from server.config import config
from server.logging_config import get_logger

logger = get_logger(__name__)

# Global client instances
_chat_client: Optional[BaseLLMClient] = None
_embedding_client: Optional[BaseLLMClient] = None


def parse_llm_service(llm_service: str) -> Tuple[str, str]:
    """
    Parse LLM_SERVICE string into provider and model.
    
    Args:
        llm_service: String in format "provider/model" (e.g., "openai/gpt-4o-mini")
        
    Returns:
        Tuple of (provider, model)
    """
    if "/" not in llm_service:
        raise ValueError(f"Invalid LLM_SERVICE format: {llm_service}. Expected 'provider/model'")
    
    provider, model = llm_service.split("/", 1)
    return provider.lower(), model


def create_openai_client() -> OpenAIClient:
    """Create OpenAI client"""
    if not config.MACHINE_LEARNING.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
    
    return OpenAIClient(
        api_key=config.MACHINE_LEARNING.OPENAI_API_KEY,
        chat_model=config.MACHINE_LEARNING.OPENAI_CHAT_MODEL,
        embedding_model=config.MACHINE_LEARNING.OPENAI_EMBEDDING_MODEL
    )


def create_azure_client() -> AzureOpenAIClient:
    """Create Azure OpenAI client"""
    if not config.MACHINE_LEARNING.AZURE_API_KEY:
        raise ValueError("AZURE_API_KEY is required for Azure provider")
    if not config.MACHINE_LEARNING.AZURE_API_BASE:
        raise ValueError("AZURE_API_BASE is required for Azure provider")
    
    return AzureOpenAIClient(
        api_key=config.MACHINE_LEARNING.AZURE_API_KEY,
        api_base=config.MACHINE_LEARNING.AZURE_API_BASE,
        api_version=config.MACHINE_LEARNING.AZURE_API_VERSION,
        chat_deployment=config.MACHINE_LEARNING.AZURE_CHAT_DEPLOYMENT,
        embedding_deployment=config.MACHINE_LEARNING.AZURE_EMBEDDING_DEPLOYMENT
    )


def create_anthropic_client() -> AnthropicClient:
    """Create Anthropic client"""
    if not config.MACHINE_LEARNING.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
    
    return AnthropicClient(
        api_key=config.MACHINE_LEARNING.ANTHROPIC_API_KEY,
        chat_model=config.MACHINE_LEARNING.ANTHROPIC_CHAT_MODEL
    )


def create_gemini_client() -> GeminiClient:
    """Create Google Gemini client"""
    if not config.MACHINE_LEARNING.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required for Gemini provider")
    
    return GeminiClient(
        api_key=config.MACHINE_LEARNING.GEMINI_API_KEY,
        chat_model=config.MACHINE_LEARNING.GEMINI_CHAT_MODEL
    )


def create_client(provider: str) -> BaseLLMClient:
    """
    Create a client for the specified provider.
    
    Args:
        provider: Provider name (openai, azure, anthropic, gemini)
        
    Returns:
        BaseLLMClient instance
    """
    if provider == "openai":
        return create_openai_client()
    elif provider == "azure":
        return create_azure_client()
    elif provider == "anthropic":
        return create_anthropic_client()
    elif provider == "gemini":
        return create_gemini_client()
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_chat_client() -> BaseLLMClient:
    """
    Get the chat client based on current configuration.
    
    Returns:
        BaseLLMClient instance for chat operations
    """
    global _chat_client
    
    if _chat_client is None:
        provider, model = parse_llm_service(config.MACHINE_LEARNING.LLM_SERVICE)
        _chat_client = create_client(provider)
        logger.info(f"Initialized chat client: {provider}/{model}")
    
    return _chat_client


def get_embedding_client() -> BaseLLMClient:
    """
    Get the embedding client. Falls back to OpenAI if current provider doesn't support embeddings.
    
    Returns:
        BaseLLMClient instance for embedding operations
    """
    global _embedding_client
    
    if _embedding_client is None:
        provider, model = parse_llm_service(config.MACHINE_LEARNING.LLM_SERVICE)
        
        # Try to use the same provider as chat
        try:
            client = create_client(provider)
            if client.supports_embeddings():
                _embedding_client = client
                logger.info(f"Initialized embedding client: {provider}")
            else:
                # Fall back to OpenAI for embeddings
                logger.info(f"Provider {provider} doesn't support embeddings, falling back to OpenAI")
                _embedding_client = create_openai_client()
        except Exception as e:
            logger.warning(f"Failed to create {provider} client for embeddings: {e}")
            logger.info("Falling back to OpenAI for embeddings")
            _embedding_client = create_openai_client()
    
    return _embedding_client


def reset_clients():
    """Reset client instances (useful for testing)"""
    global _chat_client, _embedding_client
    _chat_client = None
    _embedding_client = None 