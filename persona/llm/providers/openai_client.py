"""
OpenAI LLM client implementation.
"""

import openai
from typing import List, Dict, Any, Optional
from .base import BaseLLMClient, ChatMessage, ChatResponse, ToolCall
from server.logging_config import get_logger

logger = get_logger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client"""

    def __init__(
        self,
        api_key: str,
        chat_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        **kwargs,
    ):
        super().__init__(
            model_name=chat_model, embedding_model=embedding_model, **kwargs
        )
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
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ChatResponse:
        try:
            openai_messages = []
            for msg in messages:
                oai_msg: Dict[str, Any] = {"role": msg.role}
                if msg.content is not None:
                    oai_msg["content"] = msg.content
                if msg.tool_calls:
                    oai_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in msg.tool_calls
                    ]
                if msg.tool_call_id:
                    oai_msg["tool_call_id"] = msg.tool_call_id
                openai_messages.append(oai_msg)

            request_params: Dict[str, Any] = {
                "model": self.chat_model,
                "messages": openai_messages,
                "temperature": temperature,
            }

            if max_tokens:
                request_params["max_tokens"] = max_tokens

            if response_format:
                request_params["response_format"] = response_format

            if tools:
                request_params["tools"] = tools

            request_params.update(kwargs)

            response = await self.async_client.chat.completions.create(**request_params)

            choice = response.choices[0]
            message = choice.message

            parsed_tool_calls = None
            if message.tool_calls:
                parsed_tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in message.tool_calls
                ]

            stop_reason: str = "end_turn"
            if choice.finish_reason == "tool_calls":
                stop_reason = "tool_use"
            elif choice.finish_reason == "length":
                stop_reason = "max_tokens"

            return ChatResponse(
                content=message.content,
                model=response.model,
                usage=response.usage.model_dump() if response.usage else None,
                tool_calls=parsed_tool_calls,
                stop_reason=stop_reason,
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
                input=texts, model=self.embedding_model, dimensions=1536, **kwargs
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

    def supports_tools(self) -> bool:
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
