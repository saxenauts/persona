"""
Base interface for LLM service providers.
All LLM clients must implement this interface for consistency.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a tool call requested by the model"""

    id: str
    name: str
    arguments: str  # JSON string of arguments


class ToolResult(BaseModel):
    """Result of executing a tool"""

    tool_call_id: str
    content: str  # JSON string or plain text result


class ChatMessage(BaseModel):
    """Standard chat message format"""

    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool result messages


class ChatResponse(BaseModel):
    """Standard chat response format"""

    content: Optional[str] = None
    model: str
    usage: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[ToolCall]] = None
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "error"] = "end_turn"


class BaseLLMClient(ABC):
    """Base interface for all LLM service clients"""

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "default")
        self.embedding_model = kwargs.get("embedding_model", "default")

    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs,
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
    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
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

    @abstractmethod
    def supports_tools(self) -> bool:
        """Return whether the provider supports function/tool calling"""
        pass

    async def close(self) -> None:
        """Close any underlying client resources."""
        return None
