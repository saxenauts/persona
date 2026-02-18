"""
Base interface for LLM service providers.
All LLM clients must implement this interface for consistency.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Standard chat message format"""
    role: str  # "system", "user", "assistant"
    content: str


class ChatResponse(BaseModel):
    """Standard chat response format"""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None


class BaseLLMClient(ABC):
    """Base interface for all LLM service clients"""
    
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', 'default')
        self.embedding_model = kwargs.get('embedding_model', 'default')
    
    @abstractmethod
    async def chat(
        self, 
        messages: List[ChatMessage], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Generate a chat completion response.
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            response_format: Format specification (e.g., {"type": "json_object"})
            **kwargs: Additional provider-specific parameters
            
        Returns:
            ChatResponse object with content and metadata
        """
        pass
    
    @abstractmethod
    async def embeddings(
        self, 
        texts: List[str], 
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider (e.g., 'openai', 'azure', 'anthropic', 'gemini')"""
        pass
    
    @abstractmethod
    def supports_json_mode(self) -> bool:
        """Return whether the provider supports JSON mode responses"""
        pass
    
    @abstractmethod
    def supports_embeddings(self) -> bool:
        """Return whether the provider supports embeddings"""
        pass 