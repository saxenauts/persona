"""
Google Gemini LLM client implementation.
"""

import google.generativeai as genai
import json
from typing import List, Dict, Any, Optional
from .base import BaseLLMClient, ChatMessage, ChatResponse
from server.logging_config import get_logger

logger = get_logger(__name__)


class GeminiClient(BaseLLMClient):
    """Google Gemini LLM client"""
    
    def __init__(self, api_key: str, chat_model: str = "gemini-1.5-flash", **kwargs):
        super().__init__(model_name=chat_model, **kwargs)
        self.api_key = api_key
        self.chat_model = chat_model
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(chat_model)
    
    async def chat(
        self, 
        messages: List[ChatMessage], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate chat completion using Google Gemini API"""
        try:
            # Convert messages to Gemini format
            # Gemini uses a different conversation format
            gemini_messages = []
            system_instruction = None
            
            for msg in messages:
                if msg.role == "system":
                    system_instruction = msg.content
                elif msg.role == "user":
                    gemini_messages.append({
                        "role": "user",
                        "parts": [msg.content]
                    })
                elif msg.role == "assistant":
                    gemini_messages.append({
                        "role": "model",
                        "parts": [msg.content]
                    })
            
            # Handle JSON mode by adding instructions to system
            if response_format and response_format.get("type") == "json_object":
                json_instruction = "Please respond with valid JSON only. Do not include any text outside of the JSON response."
                if system_instruction:
                    system_instruction = f"{system_instruction}\n\n{json_instruction}"
                else:
                    system_instruction = json_instruction
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            # Create model with system instruction if provided
            if system_instruction:
                model = genai.GenerativeModel(
                    self.chat_model,
                    system_instruction=system_instruction
                )
            else:
                model = self.model
            
            # Generate response
            if gemini_messages:
                # Start chat with history
                chat = model.start_chat(history=gemini_messages[:-1])
                response = await chat.send_message_async(
                    gemini_messages[-1]["parts"][0],
                    generation_config=generation_config
                )
            else:
                # Single message
                response = await model.generate_content_async(
                    system_instruction or "",
                    generation_config=generation_config
                )
            
            return ChatResponse(
                content=response.text,
                model=self.chat_model,
                usage=None  # Gemini doesn't provide detailed usage stats in the same format
            )
            
        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            raise
    
    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Google Gemini doesn't support embeddings in the same way - raise NotImplementedError"""
        raise NotImplementedError("Gemini does not support embeddings in the standard format. Use OpenAI or Azure for embeddings.")
    
    def get_provider_name(self) -> str:
        return "gemini"
    
    def supports_json_mode(self) -> bool:
        return True  # We simulate JSON mode with system instructions
    
    def supports_embeddings(self) -> bool:
        return False 