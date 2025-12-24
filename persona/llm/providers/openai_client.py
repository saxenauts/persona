"""
OpenAI LLM client implementation.
"""

import openai
from typing import List, Dict, Any, Optional
from .base import BaseLLMClient, ChatMessage, ChatResponse
from server.logging_config import get_logger

logger = get_logger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client"""
    
    def __init__(self, api_key: str, chat_model: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small", **kwargs):
        super().__init__(model_name=chat_model, embedding_model=embedding_model, **kwargs)
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        
        # Initialize clients
        self.async_client = openai.AsyncOpenAI(api_key=api_key)
        self.sync_client = openai.OpenAI(api_key=api_key)
    
    async def chat(
        self, 
        messages: List[ChatMessage], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate chat completion using OpenAI API"""
        try:
            # Convert our standard message format to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Prepare request parameters
            request_params = {
                "model": self.chat_model,
                "messages": openai_messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            
            if response_format:
                request_params["response_format"] = response_format
            
            # Add any additional parameters
            request_params.update(kwargs)
            
            # Make the API call
            response = await self.async_client.chat.completions.create(**request_params)
            
            return ChatResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=response.usage.model_dump() if response.usage else None
            )
            
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise
    
    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        if not texts:
            return []
        
        try:
            # Use sync client for embeddings as it's more stable
            response = self.sync_client.embeddings.create(
                input=texts,
                model=self.embedding_model,
                dimensions=1536,
                **kwargs
            )
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI embeddings error: {e}")
            # Return None embeddings to maintain alignment with input
            return [None] * len(texts)
    
    def get_provider_name(self) -> str:
        return "openai"
    
    def supports_json_mode(self) -> bool:
        return True
    
    def supports_embeddings(self) -> bool:
        return True

    async def close(self) -> None:
        try:
            await self.async_client.close()
        except Exception as e:
            logger.debug(f"OpenAI async client close failed: {e}")
        try:
            self.sync_client.close()
        except Exception as e:
            logger.debug(f"OpenAI sync client close failed: {e}")
