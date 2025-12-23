"""
Azure AI Foundry LLM client implementation.

This client supports the new Azure AI Foundry platform which uses the standard
OpenAI SDK with a custom base_url, rather than the Azure-specific SDK.

Supports:
- OpenAI models (gpt-5.2, etc.)
- DeepSeek models (DeepSeek-V3.2)
- Anthropic models via Azure (claude-sonnet-4-5) - handled separately
"""

import openai
import asyncio
import time
import random
import logging
from typing import List, Dict, Any, Optional, Union
from tenacity import (
    retry, 
    wait_exponential, 
    stop_after_attempt, 
    retry_if_exception_type,
    AsyncRetrying,
    before_sleep_log
)
from .base import BaseLLMClient, ChatMessage, ChatResponse
from server.logging_config import get_logger
from persona.llm.rate_limiter import get_rate_limiter_registry, TokenBucketLimiter

logger = get_logger(__name__)

# Default quotas (can be overridden via env vars in future)
DEFAULT_QUOTAS = {
    "gpt-5.2": {"tpm": 10_000_000, "rpm": 100_000},
    "gpt-5-mini": {"tpm": 3_000_000, "rpm": 3_000},
    "text-embedding-3-small": {"tpm": 10_000_000, "rpm": 60_000},
    "DeepSeek-V3.2": {"tpm": 10_000_000, "rpm": 10_000},
    "claude-sonnet-4-5": {"tpm": 4_000_000, "rpm": 4_000},
}


class AzureFoundryClient(BaseLLMClient):
    """
    Azure AI Foundry LLM client using standard OpenAI SDK.
    
    New Azure AI Foundry uses standard OpenAI SDK with custom base_url:
    - Endpoint format: https://<resource>.openai.azure.com/openai/v1/
    - Uses max_completion_tokens instead of max_tokens for newer models
    - Lazy client creation to avoid event loop binding issues
    """
    
    def __init__(
        self, 
        api_key: str, 
        api_base: str,
        chat_deployment: str = "gpt-5.2", 
        embedding_deployment: str = "text-embedding-3-small",
        **kwargs
    ):
        super().__init__(model_name=chat_deployment, embedding_model=embedding_deployment, **kwargs)
        self.api_key = api_key
        
        # Ensure endpoint has /openai/v1/ suffix
        self.api_base = api_base.rstrip('/')
        if not self.api_base.endswith('/openai/v1'):
            self.api_base = f"{self.api_base}/openai/v1/"
        else:
            self.api_base = f"{self.api_base}/"
            
        self.chat_deployment = chat_deployment
        self.embedding_deployment = embedding_deployment
        
        # Lazy client creation - created on first use
        self._async_client: Optional[openai.AsyncOpenAI] = None
        self._client_loop_id: Optional[int] = None
        
        # Rate limiters (initialized lazily)
        self._chat_limiter: Optional[TokenBucketLimiter] = None
        self._embedding_limiter: Optional[TokenBucketLimiter] = None
        
        # Track if model uses new token parameter
        self._uses_completion_tokens = self._check_model_type(chat_deployment)
        
        logger.info(f"Initialized AzureFoundryClient: {self.api_base} with model {chat_deployment}")
    
    def _get_client(self) -> openai.AsyncOpenAI:
        """Get or create async client for current event loop."""
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            loop_id = None
        
        # Create new client if loop changed or doesn't exist
        if self._async_client is None or self._client_loop_id != loop_id:
            self._async_client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            self._client_loop_id = loop_id
            logger.debug(f"Created new AsyncOpenAI client for loop {loop_id}")
        
        return self._async_client
    
    async def _get_chat_limiter(self) -> TokenBucketLimiter:
        """Get rate limiter for chat deployment."""
        if self._chat_limiter is None:
            registry = get_rate_limiter_registry()
            quota = DEFAULT_QUOTAS.get(self.chat_deployment, {"tpm": 1_000_000, "rpm": 10_000})
            self._chat_limiter = await registry.get_or_create(
                name=f"chat:{self.chat_deployment}",
                tpm=quota["tpm"],
                rpm=quota["rpm"]
            )
        return self._chat_limiter
    
    async def _get_embedding_limiter(self) -> TokenBucketLimiter:
        """Get rate limiter for embedding deployment."""
        if self._embedding_limiter is None:
            registry = get_rate_limiter_registry()
            quota = DEFAULT_QUOTAS.get(self.embedding_deployment, {"tpm": 1_000_000, "rpm": 10_000})
            self._embedding_limiter = await registry.get_or_create(
                name=f"embed:{self.embedding_deployment}",
                tpm=quota["tpm"],
                rpm=quota["rpm"]
            )
        return self._embedding_limiter

    def _check_model_type(self, model: str) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens"""
        # GPT-5.x and newer models use max_completion_tokens
        new_models = ['gpt-5', 'o1', 'o3']
        return any(model.lower().startswith(prefix) for prefix in new_models)

    async def chat(
        self, 
        messages: List[ChatMessage], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate chat completion with rate limiting."""
        
        # Estimate tokens for rate limiting (prompt + max_completion)
        prompt_text = " ".join(m.content for m in messages)
        estimated_tokens = len(prompt_text) // 4 + (max_tokens or 1000)
        
        # Acquire rate limit capacity
        limiter = await self._get_chat_limiter()
        await limiter.acquire(estimated_tokens)
        
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)),
            before_sleep=before_sleep_log(logger, logging.INFO),
            reraise=True
        ):
            with attempt:
                try:
                    openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
                    request_params = {
                        "model": self.chat_deployment,
                        "messages": openai_messages,
                        "temperature": temperature,
                        **kwargs
                    }
                    
                    # Handle token parameter based on model type
                    if max_tokens:
                        if self._uses_completion_tokens:
                            request_params["max_completion_tokens"] = max_tokens
                        else:
                            request_params["max_tokens"] = max_tokens
                    
                    if response_format:
                        request_params["response_format"] = response_format

                    client = self._get_client()
                    response = await client.chat.completions.create(**request_params)
                    
                    return ChatResponse(
                        content=response.choices[0].message.content,
                        model=f"foundry/{self.chat_deployment}",
                        usage=response.usage.model_dump() if response.usage else None
                    )

                except openai.BadRequestError as e:
                    if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):
                        logger.warning(f"Azure Content Filter triggered: {e}")
                        return ChatResponse(
                            content="<CONTENT_FILTERED>",
                            model=f"foundry/{self.chat_deployment}",
                            usage={}
                        )
                    raise

    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings with rate limiting."""
        if not texts:
            return []
        
        # Estimate tokens for rate limiting
        estimated_tokens = sum(len(t) // 4 for t in texts)
        
        # Acquire rate limit capacity
        limiter = await self._get_embedding_limiter()
        await limiter.acquire(estimated_tokens)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)),
            reraise=True
        ):
            with attempt:
                try:
                    client = self._get_client()
                    response = await client.embeddings.create(
                        input=texts,
                        model=self.embedding_deployment,
                        **kwargs
                    )
                    return [data.embedding for data in response.data]

                except openai.BadRequestError as e:
                    if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):
                        return [[0.0] * 1536 for _ in texts]
                    raise

    def get_provider_name(self) -> str:
        return "foundry"
    
    def supports_json_mode(self) -> bool:
        return True
    
    def supports_embeddings(self) -> bool:
        return True
