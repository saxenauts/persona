"""
LLM Client Factory for managing multiple LLM service providers.
"""

from typing import Optional, Tuple
from .providers.base import BaseLLMClient
from .providers.openai_client import OpenAIClient
from .providers.azure_openai_client import AzureOpenAIClient
from .providers.azure_foundry_client import AzureFoundryClient
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
    if not config.MACHINE_LEARNING.OPENAI_CHAT_MODEL:
        raise ValueError("OPENAI_CHAT_MODEL is required for OpenAI provider")
    if not config.MACHINE_LEARNING.OPENAI_EMBEDDING_MODEL:
        raise ValueError("OPENAI_EMBEDDING_MODEL is required for OpenAI provider")
    
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
    if not config.MACHINE_LEARNING.AZURE_API_VERSION:
        raise ValueError("AZURE_API_VERSION is required for Azure provider")
    if not config.MACHINE_LEARNING.AZURE_CHAT_DEPLOYMENT:
        raise ValueError("AZURE_CHAT_DEPLOYMENT is required for Azure provider")
    if not config.MACHINE_LEARNING.AZURE_EMBEDDING_DEPLOYMENT:
        raise ValueError("AZURE_EMBEDDING_DEPLOYMENT is required for Azure provider")
    
    # AZURE_API_BASE can be a comma-separated list of endpoints
    api_base = config.MACHINE_LEARNING.AZURE_API_BASE
    
    return AzureOpenAIClient(
        api_key=config.MACHINE_LEARNING.AZURE_API_KEY,
        api_base=api_base,
        api_version=config.MACHINE_LEARNING.AZURE_API_VERSION,
        chat_deployment=config.MACHINE_LEARNING.AZURE_CHAT_DEPLOYMENT,
        embedding_deployment=config.MACHINE_LEARNING.AZURE_EMBEDDING_DEPLOYMENT
    )


def create_anthropic_client() -> AnthropicClient:
    """Create Anthropic client"""
    if not config.MACHINE_LEARNING.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
    if not config.MACHINE_LEARNING.ANTHROPIC_CHAT_MODEL:
        raise ValueError("ANTHROPIC_CHAT_MODEL is required for Anthropic provider")
    
    return AnthropicClient(
        api_key=config.MACHINE_LEARNING.ANTHROPIC_API_KEY,
        chat_model=config.MACHINE_LEARNING.ANTHROPIC_CHAT_MODEL
    )


def create_gemini_client() -> GeminiClient:
    """Create Google Gemini client"""
    if not config.MACHINE_LEARNING.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required for Gemini provider")
    if not config.MACHINE_LEARNING.GEMINI_CHAT_MODEL:
        raise ValueError("GEMINI_CHAT_MODEL is required for Gemini provider")
    
    return GeminiClient(
        api_key=config.MACHINE_LEARNING.GEMINI_API_KEY,
        chat_model=config.MACHINE_LEARNING.GEMINI_CHAT_MODEL
    )


def create_foundry_client() -> AzureFoundryClient:
    """Create Azure AI Foundry client (new platform)"""
    if not config.MACHINE_LEARNING.AZURE_API_KEY:
        raise ValueError("AZURE_API_KEY is required for Foundry provider")
    if not config.MACHINE_LEARNING.AZURE_API_BASE:
        raise ValueError("AZURE_API_BASE is required for Foundry provider")
    if not config.MACHINE_LEARNING.AZURE_CHAT_DEPLOYMENT:
        raise ValueError("AZURE_CHAT_DEPLOYMENT is required for Foundry provider")
    if not config.MACHINE_LEARNING.AZURE_EMBEDDING_DEPLOYMENT:
        raise ValueError("AZURE_EMBEDDING_DEPLOYMENT is required for Foundry provider")
    
    return AzureFoundryClient(
        api_key=config.MACHINE_LEARNING.AZURE_API_KEY,
        api_base=config.MACHINE_LEARNING.AZURE_API_BASE,
        chat_deployment=config.MACHINE_LEARNING.AZURE_CHAT_DEPLOYMENT,
        embedding_deployment=config.MACHINE_LEARNING.AZURE_EMBEDDING_DEPLOYMENT
    )


def create_client(provider: str) -> BaseLLMClient:
    """
    Create a client for the specified provider.
    
    Args:
        provider: Provider name (openai, azure, foundry, anthropic, gemini)
        
    Returns:
        BaseLLMClient instance
    """
    if provider == "openai":
        return create_openai_client()
    elif provider == "azure":
        return create_azure_client()
    elif provider == "foundry":
        return create_foundry_client()
    elif provider == "anthropic":
        return create_anthropic_client()
    elif provider == "gemini":
        return create_gemini_client()
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported: openai, azure, foundry, anthropic, gemini")


def get_chat_client() -> BaseLLMClient:
    """
    Get the chat client based on current configuration.
    
    Returns:
        BaseLLMClient instance for chat operations
    """
    global _chat_client
    
    if _chat_client is None:
        if not config.MACHINE_LEARNING.LLM_SERVICE:
            raise ValueError("LLM_SERVICE is required but not configured. Set LLM_SERVICE in .env file (e.g., LLM_SERVICE=openai/gpt-4o-mini)")
        
        provider, model = parse_llm_service(config.MACHINE_LEARNING.LLM_SERVICE)
        _chat_client = create_client(provider)
        logger.info(f"Initialized chat client: {provider}/{model}")
    
    return _chat_client


def get_embedding_client() -> BaseLLMClient:
    """
    Get the embedding client based on explicit EMBEDDING_SERVICE configuration.
    No automatic fallbacks - embedding service must be explicitly configured.
    
    Returns:
        BaseLLMClient instance for embedding operations
    """
    global _embedding_client
    
    if _embedding_client is None:
        if not config.MACHINE_LEARNING.EMBEDDING_SERVICE:
            raise ValueError("EMBEDDING_SERVICE is required but not configured. Set EMBEDDING_SERVICE in .env file (e.g., EMBEDDING_SERVICE=openai/text-embedding-3-small)")
        
        provider, model = parse_llm_service(config.MACHINE_LEARNING.EMBEDDING_SERVICE)
        _embedding_client = create_client(provider)
        
        # Validate that the provider actually supports embeddings
        if not _embedding_client.supports_embeddings():
            raise ValueError(f"Provider {provider} does not support embeddings. Use openai or azure for EMBEDDING_SERVICE.")
        
        logger.info(f"Initialized embedding client: {provider}/{model}")
    
    return _embedding_client


def reset_clients():
    """Reset client instances (useful for testing)"""
    global _chat_client, _embedding_client
    _chat_client = None
    _embedding_client = None 