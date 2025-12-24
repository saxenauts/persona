"""
Anthropic LLM client implementation.
"""

import anthropic
import json
from typing import List, Dict, Any, Optional
from .base import BaseLLMClient, ChatMessage, ChatResponse
from server.logging_config import get_logger

logger = get_logger(__name__)


class AnthropicClient(BaseLLMClient):
    """Anthropic LLM client"""
    
    def __init__(self, api_key: str, chat_model: str = "claude-3-5-sonnet-20241022", **kwargs):
        super().__init__(model_name=chat_model, **kwargs)
        self.api_key = api_key
        self.chat_model = chat_model
        
        # Initialize client
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def chat(
        self, 
        messages: List[ChatMessage], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate chat completion using Anthropic API"""
        try:
            # Convert messages to Anthropic format
            # Anthropic requires system message to be separate
            system_message = None
            anthropic_messages = []
            
            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    anthropic_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Prepare request parameters
            request_params = {
                "model": self.chat_model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4000,
            }
            
            if system_message:
                request_params["system"] = system_message
            
            # Handle JSON mode by adding instructions to system message
            if response_format and response_format.get("type") == "json_object":
                json_instruction = "Please respond with valid JSON only. Do not include any text outside of the JSON response."
                if system_message:
                    request_params["system"] = f"{system_message}\n\n{json_instruction}"
                else:
                    request_params["system"] = json_instruction
            
            # Add any additional parameters
            request_params.update(kwargs)
            
            # Make the API call
            response = await self.client.messages.create(**request_params)
            
            return ChatResponse(
                content=response.content[0].text,
                model=response.model,
                usage=response.usage.model_dump() if response.usage else None
            )
            
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            raise
    
    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Anthropic doesn't support embeddings - raise NotImplementedError"""
        raise NotImplementedError("Anthropic does not support embeddings. Use OpenAI or Azure for embeddings.")
    
    def get_provider_name(self) -> str:
        return "anthropic"
    
    def supports_json_mode(self) -> bool:
        return True  # We simulate JSON mode with system prompts
    
    def supports_embeddings(self) -> bool:
        return False

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception as e:
            logger.debug(f"Anthropic client close failed: {e}")
