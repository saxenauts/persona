"""
Token-Bucket Rate Limiter for Azure OpenAI / Foundry APIs.

Features:
- Per-second smoothing (not per-minute bursts)
- Dual limits: TPM (tokens) and RPM (requests)
- 429 detection with Retry-After parsing
- Async-safe with asyncio.Lock
- Metrics tracking
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterMetrics:
    """Metrics tracked by the rate limiter."""
    total_requests: int = 0
    total_tokens: int = 0
    total_wait_time_ms: float = 0
    retries_429: int = 0
    

@dataclass
class TokenBucketLimiter:
    """
    Token-bucket rate limiter for a single deployment.
    
    The bucket fills continuously at `tokens_per_second` rate.
    Each request consumes tokens. If bucket is empty, wait.
    
    Args:
        tpm: Tokens per minute limit
        rpm: Requests per minute limit
        name: Deployment name for logging
    """
    tpm: int
    rpm: int
    name: str = "default"
    
    # Internal state
    _tokens: float = field(init=False)
    _requests: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: asyncio.Lock = field(init=False)
    metrics: RateLimiterMetrics = field(init=False)
    
    def __post_init__(self):
        # Start with full buckets
        self._tokens = float(self.tpm)
        self._requests = float(self.rpm)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
        self.metrics = RateLimiterMetrics()
        
        # Per-second rates
        self._tokens_per_sec = self.tpm / 60.0
        self._requests_per_sec = self.rpm / 60.0
        
        logger.info(f"[RateLimiter:{self.name}] Initialized: {self.tpm} TPM, {self.rpm} RPM")
    
    def _refill(self):
        """Refill bucket based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        
        # Add tokens proportional to elapsed time
        self._tokens = min(self.tpm, self._tokens + elapsed * self._tokens_per_sec)
        self._requests = min(self.rpm, self._requests + elapsed * self._requests_per_sec)
    
    async def acquire(self, estimated_tokens: int) -> float:
        """
        Acquire capacity for a request.
        
        Args:
            estimated_tokens: Estimated tokens (prompt + max_completion)
            
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        async with self._lock:
            self._refill()
            
            wait_time = 0.0
            
            # Check if we need to wait for tokens
            if self._tokens < estimated_tokens:
                tokens_needed = estimated_tokens - self._tokens
                wait_time = max(wait_time, tokens_needed / self._tokens_per_sec)
            
            # Check if we need to wait for request slot
            if self._requests < 1:
                requests_needed = 1 - self._requests
                wait_time = max(wait_time, requests_needed / self._requests_per_sec)
            
            if wait_time > 0:
                logger.debug(f"[RateLimiter:{self.name}] Waiting {wait_time:.2f}s for capacity")
                self.metrics.total_wait_time_ms += wait_time * 1000
                await asyncio.sleep(wait_time)
                self._refill()
            
            # Consume capacity
            self._tokens -= estimated_tokens
            self._requests -= 1
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.total_tokens += estimated_tokens
            
            return wait_time
    
    async def handle_429(self, retry_after: Optional[float] = None):
        """
        Handle a 429 response.
        
        Args:
            retry_after: Value from Retry-After header (seconds)
        """
        async with self._lock:
            self.metrics.retries_429 += 1
            
            # Default backoff if no Retry-After
            wait_time = retry_after or 5.0
            
            # Add jitter (10-20% extra)
            import random
            jitter = wait_time * random.uniform(0.1, 0.2)
            total_wait = wait_time + jitter
            
            logger.warning(f"[RateLimiter:{self.name}] 429 received. Waiting {total_wait:.2f}s")
            self.metrics.total_wait_time_ms += total_wait * 1000
            
            await asyncio.sleep(total_wait)
            
            # Refill after wait
            self._refill()
    
    def get_stats(self) -> dict:
        """Get current stats for logging."""
        return {
            "name": self.name,
            "bucket_tokens": round(self._tokens),
            "bucket_requests": round(self._requests, 1),
            "total_requests": self.metrics.total_requests,
            "total_tokens": self.metrics.total_tokens,
            "wait_time_ms": round(self.metrics.total_wait_time_ms),
            "retries_429": self.metrics.retries_429,
        }


class RateLimiterRegistry:
    """
    Registry of rate limiters, one per deployment.
    
    Usage:
        registry = RateLimiterRegistry()
        limiter = registry.get_or_create("gpt-5.2", tpm=10_000_000, rpm=100_000)
        await limiter.acquire(estimated_tokens=500)
    """
    
    def __init__(self):
        self._limiters: dict[str, TokenBucketLimiter] = {}
        self._lock = asyncio.Lock()
    
    async def get_or_create(
        self, 
        name: str, 
        tpm: int, 
        rpm: int
    ) -> TokenBucketLimiter:
        """Get existing limiter or create new one."""
        async with self._lock:
            if name not in self._limiters:
                self._limiters[name] = TokenBucketLimiter(tpm=tpm, rpm=rpm, name=name)
            return self._limiters[name]
    
    def get_all_stats(self) -> list[dict]:
        """Get stats from all limiters."""
        return [limiter.get_stats() for limiter in self._limiters.values()]


# Global registry
_registry: Optional[RateLimiterRegistry] = None


def get_rate_limiter_registry() -> RateLimiterRegistry:
    """Get the global rate limiter registry."""
    global _registry
    if _registry is None:
        _registry = RateLimiterRegistry()
    return _registry
