import openai
import asyncio
import time
import random
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

logger = get_logger(__name__)


class AzureEndpointState:
    """State tracking for a single Azure OpenAI endpoint"""
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.remaining_tokens: Optional[int] = None
        self.reset_at: float = 0
        self.last_429_at: float = 0
        self.backoff_until: float = 0
        self.is_healthy: bool = True

    def update_from_headers(self, headers: Dict[str, str]):
        """Update state using x-ratelimit headers"""
        try:
            if "x-ratelimit-remaining-tokens" in headers:
                self.remaining_tokens = int(headers["x-ratelimit-remaining-tokens"])
            
            if "x-ratelimit-reset-tokens" in headers:
                # reset-tokens is often a string like "1s" or "30s" or a timestamp
                reset_str = headers["x-ratelimit-reset-tokens"]
                if reset_str.endswith('ms'):
                    ms = float(reset_str[:-2])
                    self.reset_at = time.time() + (ms / 1000.0)
                elif reset_str.endswith('s'):
                    seconds = float(reset_str[:-1])
                    self.reset_at = time.time() + seconds
                else:
                    # Fallback if it's just a number
                    self.reset_at = time.time() + float(reset_str)
        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"Error parsing rate limit headers: {e}")

    def mark_429(self, retry_after: Optional[float] = None):
        """Mark endpoint as throttled"""
        self.last_429_at = time.time()
        self.is_healthy = False
        wait_time = retry_after or 30.0  # Default 30s backoff
        self.backoff_until = time.time() + wait_time
        logger.warning(f"Endpoint {self.endpoint} hit 429. Backing off for {wait_time}s")

    def check_health(self) -> bool:
        """Check if endpoint is ready to be used"""
        if not self.is_healthy and time.time() > self.backoff_until:
            self.is_healthy = True
        return self.is_healthy


class AzureOpenAIClient(BaseLLMClient):
    """
    Azure OpenAI LLM client with scaling and rate limit mitigation.
    Supports multiple endpoints for load balancing and header-aware pacing.
    """
    
    def __init__(
        self, 
        api_key: str, 
        api_base: Union[str, List[str]], 
        api_version: str = "2024-02-01",
        chat_deployment: str = "gpt-4o-mini", 
        embedding_deployment: str = "text-embedding-3-small",
        **kwargs
    ):
        super().__init__(model_name=chat_deployment, embedding_model=embedding_deployment, **kwargs)
        self.api_key = api_key
        # Support comma-separated string or list for multiple endpoints
        if isinstance(api_base, str):
            self.endpoints = [e.strip() for e in api_base.split(",") if e.strip()]
        else:
            self.endpoints = api_base
            
        self.api_version = api_version
        self.chat_deployment = chat_deployment
        self.embedding_deployment = embedding_deployment
        
        # Tracking state for each endpoint
        self.endpoint_states = {url: AzureEndpointState(url) for url in self.endpoints}
        
        # Initialize clients for each endpoint
        self._async_clients = {
            url: openai.AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=url,
                api_version=api_version
            ) for url in self.endpoints
        }
        
        self.current_endpoint_index = 0
        logger.info(f"Initialized AzureOpenAIClient with {len(self.endpoints)} endpoints.")

    def _get_next_endpoint(self) -> str:
        """Simple round-robin load balancer that skips unhealthy endpoints"""
        for _ in range(len(self.endpoints)):
            url = self.endpoints[self.current_endpoint_index]
            self.current_endpoint_index = (self.current_endpoint_index + 1) % len(self.endpoints)
            
            if self.endpoint_states[url].check_health():
                return url
        
        # If all are unhealthy, pick the one that resets soonest
        return min(self.endpoints, key=lambda u: self.endpoint_states[u].backoff_until)

    async def _pace(self, endpoint_url: str):
        """Introduce small delays if tokens are running low"""
        state = self.endpoint_states[endpoint_url]
        if state.remaining_tokens is not None and state.remaining_tokens < 5000:
            # If less than 5k tokens left, wait a bit
            wait_time = random.uniform(0.5, 2.0)
            logger.info(f"Pacing: {state.remaining_tokens} tokens left on {endpoint_url}. Waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

    async def chat(
        self, 
        messages: List[ChatMessage], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate chat completion with load balancing and rate limit handling"""
        
        # Inner retry logic that handles endpoint switching
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)),
            before_sleep=before_sleep_log(logger, "INFO"),
            reraise=True
        ):
            with attempt:
                endpoint_url = self._get_next_endpoint()
                await self._pace(endpoint_url)
                client = self._async_clients[endpoint_url]
                state = self.endpoint_states[endpoint_url]
                
                try:
                    openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
                    request_params = {
                        "model": self.chat_deployment,
                        "messages": openai_messages,
                        "temperature": temperature,
                        **kwargs
                    }
                    if max_tokens: request_params["max_tokens"] = max_tokens
                    if response_format: request_params["response_format"] = response_format

                    # Use with_raw_response to access headers
                    raw_response = await client.chat.completions.with_raw_response.create(**request_params)
                    response = raw_response.parse()
                    
                    # Update state from headers
                    state.update_from_headers(raw_response.headers)
                    
                    return ChatResponse(
                        content=response.choices[0].message.content,
                        model=f"azure/{self.chat_deployment}@{endpoint_url}",
                        usage=response.usage.model_dump() if response.usage else None
                    )

                except openai.RateLimitError as e:
                    # Extract retry-after if available
                    retry_after = None
                    if hasattr(e, 'response') and 'retry-after' in e.response.headers:
                        retry_after = float(e.response.headers['retry-after'])
                    elif 'retry-after-ms' in str(e): # Sometimes in error message
                         # Rough parsing if needed
                         pass
                    
                    state.mark_429(retry_after)
                    raise # Let tenacity handle the sleep and retry potentially on a different endpoint

                except openai.BadRequestError as e:
                    if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):
                        logger.warning(f"Azure Content Filter triggered: {e}")
                        return ChatResponse(
                            content="<CONTENT_FILTERED>",
                            model=f"azure/{self.chat_deployment}",
                            usage={}
                        )
                    raise

    async def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings with load balancing"""
        if not texts:
            return []

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)),
            reraise=True
        ):
            with attempt:
                endpoint_url = self._get_next_endpoint()
                client = self._async_clients[endpoint_url]
                state = self.endpoint_states[endpoint_url]
                
                try:
                    raw_response = await client.embeddings.with_raw_response.create(
                        input=texts,
                        model=self.embedding_deployment,
                        **kwargs
                    )
                    response = raw_response.parse()
                    state.update_from_headers(raw_response.headers)
                    
                    return [data.embedding for data in response.data]

                except openai.RateLimitError:
                    state.mark_429()
                    raise
                except openai.BadRequestError as e:
                    if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):
                        return [[0.0] * 1536 for _ in texts]
                    raise

    async def chat_batch(self, tasks: List[Dict[str, Any]]) -> str:
        """
        [PLACEHOLDER] Submit a batch for processing.
        In a real implementation, this would use the Azure OpenAI Batch API.
        """
        logger.info(f"Submitting batch of {len(tasks)} tasks.")
        # Implementation would involve async_client.batches.create(...)
        return "batch_id_placeholder"

    def get_provider_name(self) -> str:
        return "azure"
    
    def supports_json_mode(self) -> bool:
        return True
    
    def supports_embeddings(self) -> bool:
        return True

    async def close(self) -> None:
        for client in self._async_clients.values():
            try:
                await client.close()
            except Exception as e:
                logger.debug(f"Azure OpenAI client close failed: {e}")
        self._async_clients = {}
 
