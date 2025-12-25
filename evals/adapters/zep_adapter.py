import os
import json
import asyncio
import re
import threading
import time
import math
from types import SimpleNamespace
from datetime import datetime

from openai import AsyncAzureOpenAI, AsyncOpenAI, RateLimitError
from neo4j import GraphDatabase, AsyncGraphDatabase

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from graphiti_core.cross_encoder import OpenAIRerankerClient
from graphiti_core.helpers import semaphore_gather
from graphiti_core.prompts import Message
from graphiti_core.search.search_config import (
    SearchConfig,
    EdgeSearchConfig,
    NodeSearchConfig,
    EdgeSearchMethod,
    NodeSearchMethod,
    EdgeReranker,
    NodeReranker,
)

from .base import MemorySystem
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from collections import deque


# =============================================================================
# BUGFIX: graphiti_core 0.24.3 reasoning.effort parameter bug
# =============================================================================
# graphiti_core incorrectly detects gpt-5.x as reasoning models and sends
# the 'reasoning.effort' parameter, which Azure OpenAI rejects with 400 error.
# Only o1-* and o3-* models actually support this parameter.
#
# This patch applies immediately at import time to prevent any ingestion failures.
# =============================================================================


@staticmethod
def _patched_supports_reasoning(model: str) -> bool:
    """Only enable reasoning.effort for actual reasoning models (o1, o3)."""
    return model.startswith(("o1-", "o3-"))


# Patch both OpenAI clients
AzureOpenAILLMClient._supports_reasoning_features = _patched_supports_reasoning
OpenAIClient._supports_reasoning_features = _patched_supports_reasoning

print("[GraphitiAdapter] Applied reasoning.effort bugfix patch for gpt-5.x models")


class CallRateMonitor:
    """Monitor API call rates with rolling window and periodic logging."""

    def __init__(
        self,
        window_seconds: int = 60,
        log_interval: int = 30,
        estimated_tokens_per_call: int = 2000,
    ):
        self.window_seconds = window_seconds
        self.log_interval = log_interval
        self.estimated_tokens_per_call = estimated_tokens_per_call
        self.calls = deque()  # (timestamp, call_type) tuples
        self.lock = threading.Lock()
        self.last_log_time = time.time()
        self.total_calls = 0
        self.quota_tpm = 10_000_000  # gpt-5 quota

    def record_call(self, call_type: str = "llm"):
        """Record an API call and optionally log stats."""
        now = time.time()
        with self.lock:
            self.calls.append((now, call_type))
            self.total_calls += 1

            # Prune old calls outside window
            cutoff = now - self.window_seconds
            while self.calls and self.calls[0][0] < cutoff:
                self.calls.popleft()

            # Log every interval
            if now - self.last_log_time >= self.log_interval:
                self._log_stats(now)
                self.last_log_time = now

    def _log_stats(self, now: float):
        """Log current call rate statistics."""
        calls_in_window = len(self.calls)
        rpm = calls_in_window * (60.0 / self.window_seconds)
        estimated_tpm = rpm * self.estimated_tokens_per_call
        utilization = (estimated_tpm / self.quota_tpm) * 100

        print(
            f"📊 [RateMonitor] Last {self.window_seconds}s: {calls_in_window} calls | "
            f"RPM: {rpm:.1f} | Est. TPM: {estimated_tpm:,.0f}/{self.quota_tpm:,} ({utilization:.1f}%) | "
            f"Total: {self.total_calls}"
        )


class ThreadSafeRateLimiter:
    """A thread-safe rate limiter using threading.Lock (works across event loops)."""

    # Shared monitor across all instances
    _monitor = CallRateMonitor(
        window_seconds=60, log_interval=30, estimated_tokens_per_call=2000
    )

    def __init__(self, requests_per_second: float):
        self.delay = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self.lock = threading.Lock()
        self.last_request = 0.0

    def wait(self):
        """Synchronous wait - call this BEFORE entering async context."""
        with self.lock:
            ThreadSafeRateLimiter._monitor.record_call("llm")
            if self.delay <= 0:
                return
            now = time.monotonic()
            elapsed = now - self.last_request
            wait_time = max(0, self.delay - elapsed)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request = time.monotonic()


class CustomNeo4jDriver(Neo4jDriver):
    """Custom driver to increase connection pool size for high-throughput benchmarking."""

    def __init__(self, uri, user, password, database="neo4j"):
        # Re-implement init to set max_connection_pool_size
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user or "", password or ""),
            max_connection_pool_size=500,  # Increased from default 100
            connection_acquisition_timeout=120.0,  # Wait longer for connection
        )
        self._database = database

        # Schedule indices (same as original)
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.build_indices_and_constraints())
        except RuntimeError:
            pass
        self.aoss_client = None


class GraphitiAdapter(MemorySystem):
    """Graphiti adapter aligned with Zep's LongMemEval pipeline."""

    # Thread-safe global rate limiter
    _rate_limiter = ThreadSafeRateLimiter(
        requests_per_second=float(os.getenv("GRAPHITI_RPS", "0") or 0)
    )

    def _run_async(self, coro):
        """Run a coroutine in a fresh event loop. Always creates new loop to avoid contamination."""
        return asyncio.run(coro)

    def __init__(self):
        # Neo4j Config
        self.neo4j_uri = os.getenv("URI_NEO4J", "bolt://127.0.0.1:7687")
        if "neo4j:7687" in self.neo4j_uri:
            self.neo4j_uri = self.neo4j_uri.replace("neo4j", "127.0.0.1")
        print(f"[GraphitiAdapter] Configured Neo4j at {self.neo4j_uri}...")
        self.neo4j_user = os.getenv("USER_NEO4J", "neo4j")
        self.neo4j_password = os.getenv("PASSWORD_NEO4J", "password")

        provider = (os.getenv("GRAPHITI_PROVIDER") or "openai").strip().lower()
        if provider not in {"openai", "azure"}:
            provider = "openai"
        self.use_azure = provider == "azure"

        if self.use_azure and not (
            os.getenv("AZURE_API_KEY")
            and os.getenv("AZURE_API_BASE")
            and os.getenv("AZURE_CHAT_DEPLOYMENT")
        ):
            raise ValueError(
                "GRAPHITI_PROVIDER=azure requires AZURE_API_KEY, AZURE_API_BASE, "
                "and AZURE_CHAT_DEPLOYMENT."
            )
        azure_model = os.getenv("AZURE_CHAT_DEPLOYMENT")
        azure_embed = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        azure_reranker = os.getenv("AZURE_RERANKER_DEPLOYMENT")

        default_llm_model = azure_model if self.use_azure else "gpt-4o-mini"
        default_embed_model = (
            azure_embed if self.use_azure else "text-embedding-3-small"
        )
        default_reranker_model = (
            (azure_reranker if azure_reranker else "gpt-4.1-nano")
            if self.use_azure
            else "gpt-4.1-nano"
        )

        self.llm_model = os.getenv("GRAPHITI_LLM_MODEL", default_llm_model)
        self.embed_model = os.getenv("GRAPHITI_EMBEDDING_MODEL", default_embed_model)
        self.reranker_model = os.getenv(
            "GRAPHITI_RERANKER_MODEL", default_reranker_model
        )
        self.generator_model = os.getenv("GRAPHITI_GENERATOR_MODEL", self.llm_model)
        self.generator_temperature = float(
            os.getenv("GRAPHITI_GENERATOR_TEMPERATURE", "0") or 0
        )
        default_generator_max = "256" if self.use_azure else "0"
        self.generator_max_tokens = int(
            os.getenv("GRAPHITI_GENERATOR_MAX_TOKENS", default_generator_max) or 0
        )
        self.search_limit = int(os.getenv("GRAPHITI_SEARCH_LIMIT", "20"))
        self.query_max_chars = int(os.getenv("GRAPHITI_QUERY_MAX_CHARS", "255"))
        self.reranker_max_tokens = int(os.getenv("GRAPHITI_RERANKER_MAX_TOKENS", "32"))
        self.reranker_timeout_s = float(
            os.getenv("GRAPHITI_RERANKER_TIMEOUT_S", "120") or 120
        )
        self.ingest_timeout_s = float(
            os.getenv("GRAPHITI_INGEST_TIMEOUT_S", "600") or 600
        )
        self.retrieval_timeout_s = float(
            os.getenv("GRAPHITI_RETRIEVAL_TIMEOUT_S", "0") or 0
        )
        self.reranker_api_version = os.getenv(
            "AZURE_RERANKER_API_VERSION",
            os.getenv("AZURE_API_VERSION", "2024-08-01-preview"),
        )

        self.search_config = self._build_search_config()

        # Stage logging for diagnosis
        self.last_stage_logs = {}
        self._create_log_dir()

    def _create_log_dir(self):
        """Create directory for detailed stage logs."""
        self.log_dir = "evals/results/graphiti_stage_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def _log_stage(self, user_id: str, stage: str, data: dict):
        """Log stage data for later diagnosis."""
        self.last_stage_logs[f"{user_id}_{stage}"] = data
        # Also write to file for persistence
        log_file = f"{self.log_dir}/{user_id}.jsonl"
        with open(log_file, "a") as f:
            f.write(
                json.dumps(
                    {"stage": stage, "timestamp": datetime.now().isoformat(), **data}
                )
                + "\n"
            )

    def _build_search_config(self) -> SearchConfig:
        """Match the Zep LongMemEval search recipe (edges cross-encoder, nodes RRF).

        Edge reranker can be configured via GRAPHITI_EDGE_RERANKER:
        - "cross_encoder" (default): Most accurate, but slow (1 LLM call per edge)
        - "rrf": Fast reciprocal rank fusion (no LLM calls), less accurate
        - "mmr": Maximal marginal relevance (embedding-based)

        For large graphs, consider using "rrf" to avoid timeouts.
        """
        edge_reranker_name = os.getenv(
            "GRAPHITI_EDGE_RERANKER", "cross_encoder"
        ).lower()

        if edge_reranker_name == "rrf":
            edge_reranker = EdgeReranker.rrf
            print(f"[GraphitiAdapter] Using EdgeReranker.rrf (fast mode)")
        elif edge_reranker_name == "mmr":
            edge_reranker = EdgeReranker.mmr
            print(f"[GraphitiAdapter] Using EdgeReranker.mmr")
        else:
            edge_reranker = EdgeReranker.cross_encoder
            print(
                f"[GraphitiAdapter] Using EdgeReranker.cross_encoder (accurate mode, limit={self.search_limit})"
            )

        return SearchConfig(
            edge_config=EdgeSearchConfig(
                search_methods=[
                    EdgeSearchMethod.bm25,
                    EdgeSearchMethod.cosine_similarity,
                    EdgeSearchMethod.bfs,
                ],
                reranker=edge_reranker,
            ),
            node_config=NodeSearchConfig(
                search_methods=[
                    NodeSearchMethod.bm25,
                    NodeSearchMethod.cosine_similarity,
                ],
                reranker=NodeReranker.rrf,
            ),
            limit=self.search_limit,
        )

    async def _get_graphiti(self, timeout: float = 60.0):
        """
        Initialize Graphiti client with loop-safe, fresh async clients.

        Args:
            timeout: Timeout in seconds for LLM clients.
                     Ingestion needs ~300s, Retrieval needs ~60s.
        """
        if self.use_azure:
            # Robust ResponseWrapper to handle non-object responses (strings, errors)
            class ResponseWrapper:
                def __init__(self, response):
                    self.response = response

                @property
                def output_text(self):
                    if isinstance(self.response, str):
                        return "{}"
                    try:
                        return self.response.choices[0].message.content
                    except (AttributeError, IndexError):
                        return "{}"

                @property
                def refusal(self):
                    if isinstance(self.response, str):
                        return "Rate limited or error"
                    try:
                        return self.response.choices[0].message.refusal
                    except Exception:
                        return None

                def model_dump(self):
                    if isinstance(self.response, str):
                        return {"error": self.response}
                    try:
                        return self.response.model_dump()
                    except Exception:
                        return {"raw": str(self.response)}

            async def _patched_create_structured_completion(
                llm_self,
                model: str,
                messages: list,
                temperature: float | None,
                max_tokens: int,
                response_model: type,
                reasoning: str | None = None,
                verbosity: str | None = None,
            ):
                GraphitiAdapter._rate_limiter.wait()

                async for attempt in AsyncRetrying(
                    wait=wait_exponential(multiplier=1, min=2, max=30),
                    stop=stop_after_attempt(3),
                    retry=retry_if_exception_type(
                        (RateLimitError,)
                    ),  # Only retry rate limits, not timeouts
                ):
                    with attempt:
                        call_kwargs = {
                            "model": model,
                            "messages": messages,
                            "temperature": temperature,
                            "response_format": response_model,
                        }
                        if model.startswith(("gpt-5", "o1", "o3")):
                            call_kwargs["max_completion_tokens"] = max_tokens
                            if reasoning:
                                call_kwargs["reasoning_effort"] = reasoning
                        else:
                            # For gpt-4o etc, use max_tokens
                            call_kwargs["max_tokens"] = max_tokens

                        response = await llm_self.client.beta.chat.completions.parse(
                            **call_kwargs
                        )
                        return ResponseWrapper(response)

            async def _patched_create_completion(
                llm_self,
                model: str,
                messages: list,
                temperature: float | None,
                max_tokens: int,
                response_model: type | None = None,
            ):
                GraphitiAdapter._rate_limiter.wait()
                async for attempt in AsyncRetrying(
                    wait=wait_exponential(multiplier=1, min=2, max=30),
                    stop=stop_after_attempt(3),
                    retry=retry_if_exception_type(
                        (RateLimitError,)
                    ),  # Only retry rate limits, not timeouts
                ):
                    with attempt:
                        request_kwargs = {
                            "model": model,
                            "messages": messages,
                            "response_format": {"type": "json_object"},
                        }
                        if model.startswith(("gpt-5", "o1", "o3")):
                            request_kwargs["max_completion_tokens"] = max_tokens
                        else:
                            request_kwargs["max_tokens"] = max_tokens
                        if temperature is not None:
                            request_kwargs["temperature"] = temperature

                        return await llm_self.client.chat.completions.create(
                            **request_kwargs
                        )

            AzureOpenAILLMClient._create_structured_completion = (
                _patched_create_structured_completion
            )
            AzureOpenAILLMClient._create_completion = _patched_create_completion

            generator_client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION", "2024-08-01-preview"),
                azure_endpoint=os.getenv("AZURE_API_BASE"),
                azure_deployment=self.generator_model,
                timeout=timeout,
                max_retries=0,
            )
            print(
                "[GraphitiAdapter] Patched AzureOpenAILLMClient with Graphiti rate limiting."
            )

            azure_chat_client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION", "2024-08-01-preview"),
                azure_endpoint=os.getenv("AZURE_API_BASE"),
                azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
                timeout=timeout,  # Dynamic timeout based on operation
                max_retries=0,  # Disable SDK retries - we control retry at app level
            )
            # DEBUG: Verify timeout is set correctly
            actual_timeout = getattr(azure_chat_client, "timeout", "UNKNOWN")
            print(
                f"[GraphitiAdapter] DEBUG: Azure chat client timeout = {actual_timeout}"
            )

            llm_config = LLMConfig(
                api_key=os.getenv("AZURE_API_KEY"),
                model=self.llm_model,
                small_model=self.llm_model,
                temperature=float(os.getenv("GRAPHITI_LLM_TEMPERATURE", "0") or 0),
            )
            llm_client = AzureOpenAILLMClient(
                azure_client=azure_chat_client,
                config=llm_config,
            )
            # FORCE PATCH INSTANCE METHODS
            # This ensures we override whatever the class defines
            import types

            llm_client._create_structured_completion = types.MethodType(
                _patched_create_structured_completion, llm_client
            )
            llm_client._create_completion = types.MethodType(
                _patched_create_completion, llm_client
            )

            azure_emb_client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_API_KEY"),
                api_version=os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15"),
                azure_endpoint=os.getenv("AZURE_API_BASE"),
                azure_deployment=self.embed_model,
                timeout=timeout,  # Dynamic timeout
                max_retries=0,  # Disable SDK retries
            )
            # DEBUG: Verify embedder timeout
            emb_timeout = getattr(azure_emb_client, "timeout", "UNKNOWN")
            print(
                f"[GraphitiAdapter] DEBUG: Azure embedder client timeout = {emb_timeout}"
            )

            embedder = AzureOpenAIEmbedderClient(
                azure_client=azure_emb_client,
                model=self.embed_model,
            )
        else:
            generator_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                timeout=timeout,
                max_retries=0,
            )
            llm_config = LLMConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                model=self.llm_model,
                small_model=self.llm_model,
                temperature=float(os.getenv("GRAPHITI_LLM_TEMPERATURE", "0") or 0),
            )
            llm_client = OpenAIClient(config=llm_config)

            embedder_config = OpenAIEmbedderConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                embedding_model=self.embed_model,
            )
            embedder = OpenAIEmbedder(config=embedder_config)

        if self.use_azure:
            reranker_async_client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_API_KEY"),
                api_version=self.reranker_api_version,
                azure_endpoint=os.getenv("AZURE_API_BASE"),
                azure_deployment=self.reranker_model,
                timeout=timeout,  # Dynamic timeout
                max_retries=0,  # Disable SDK retries
            )
            reranker_config = LLMConfig(
                api_key=os.getenv("AZURE_API_KEY"),
                model=self.reranker_model,
                temperature=0,
                max_tokens=1,
            )
            reranker_client = OpenAIRerankerClient(
                config=reranker_config,
                client=reranker_async_client,
            )
        else:
            reranker_config = LLMConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                model=self.reranker_model,
                temperature=0,
                max_tokens=1,
            )
            reranker_client = OpenAIRerankerClient(config=reranker_config)

        if self.use_azure:
            reranker_max_tokens = self.reranker_max_tokens
            reranker_timeout_s = self.reranker_timeout_s

            async def _rank_azure(self, query: str, passages: list[str]):
                openai_messages_list = [
                    [
                        Message(
                            role="system",
                            content="You are an expert tasked with determining whether the passage is relevant to the query",
                        ),
                        Message(
                            role="user",
                            content=f"""
                           Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
                           <PASSAGE>
                           {passage}
                           </PASSAGE>
                           <QUERY>
                           {query}
                           </QUERY>
                           """,
                        ),
                    ]
                    for passage in passages
                ]

                def _build_request(openai_messages):
                    request = {
                        "model": self.config.model,
                        "messages": openai_messages,
                        "temperature": 0,
                    }
                    if self.config.model.startswith(("gpt-5", "o1", "o3")):
                        request["max_completion_tokens"] = reranker_max_tokens
                    else:
                        request["max_tokens"] = reranker_max_tokens
                    return request

                async def _call(openai_messages):
                    request = _build_request(openai_messages)
                    if reranker_timeout_s > 0:
                        try:
                            return await asyncio.wait_for(
                                self.client.chat.completions.create(**request),
                                timeout=reranker_timeout_s,
                            )
                        except Exception:
                            return None
                    return await self.client.chat.completions.create(**request)

                responses = await semaphore_gather(
                    *[
                        _call(openai_messages)
                        for openai_messages in openai_messages_list
                    ]
                )

                scores = []
                for response in responses:
                    if response is None:
                        scores.append(0.0)
                        continue
                    content = (
                        (response.choices[0].message.content or "").strip().lower()
                    )
                    scores.append(1.0 if content.startswith("true") else 0.0)

                results = [
                    (passage, score)
                    for passage, score in zip(passages, scores, strict=True)
                ]
                results.sort(reverse=True, key=lambda x: x[1])
                return results

            reranker_client.rank = _rank_azure.__get__(
                reranker_client, OpenAIRerankerClient
            )
        elif self.reranker_model.startswith(("gpt-5", "o1", "o3")):

            async def _rank_with_max_completion(self, query: str, passages: list[str]):
                openai_messages_list = [
                    [
                        Message(
                            role="system",
                            content="You are an expert tasked with determining whether the passage is relevant to the query",
                        ),
                        Message(
                            role="user",
                            content=f"""
                           Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
                           <PASSAGE>
                           {passage}
                           </PASSAGE>
                           <QUERY>
                           {query}
                           </QUERY>
                           """,
                        ),
                    ]
                    for passage in passages
                ]

                responses = await semaphore_gather(
                    *[
                        self.client.chat.completions.create(
                            model=self.config.model,
                            messages=openai_messages,
                            temperature=0,
                            max_completion_tokens=1,
                            logit_bias={"6432": 1, "7983": 1},
                            logprobs=True,
                            top_logprobs=2,
                        )
                        for openai_messages in openai_messages_list
                    ]
                )

                scores = []
                for response in responses:
                    logprobs = response.choices[0].logprobs
                    if logprobs and logprobs.content:
                        top_logprobs = logprobs.content[0].top_logprobs
                        if top_logprobs:
                            norm_logprobs = math.exp(top_logprobs[0].logprob)
                            token = top_logprobs[0].token.strip().split(" ")[0].lower()
                            scores.append(
                                norm_logprobs if token == "true" else 1 - norm_logprobs
                            )
                            continue
                    content = (
                        (response.choices[0].message.content or "").strip().lower()
                    )
                    scores.append(1.0 if content.startswith("true") else 0.0)

                results = [
                    (passage, score)
                    for passage, score in zip(passages, scores, strict=True)
                ]
                results.sort(reverse=True, key=lambda x: x[1])
                return results

            reranker_client.rank = _rank_with_max_completion.__get__(
                reranker_client, OpenAIRerankerClient
            )

        # Use Custom Driver with large connection pool
        custom_driver = CustomNeo4jDriver(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
        )

        graphiti_client = Graphiti(
            graph_driver=custom_driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=reranker_client,
        )

        # Return all async clients for explicit cleanup
        async_clients = [generator_client]
        if self.use_azure:
            async_clients.extend(
                [azure_chat_client, azure_emb_client, reranker_async_client]
            )
        else:
            # For OpenAI, we need to extract from wrapper clients if possible,
            # but usually they manage their own.
            pass

        return graphiti_client, generator_client, async_clients

    async def _close_all(self):
        """No-op - clients are created fresh per call and cleaned up by asyncio.run()."""
        pass

    def add_session(self, user_id: str, session_data: str, date: str):
        self.add_sessions(user_id, [{"content": session_data, "date": date}])

    def add_sessions(self, user_id: str, sessions: list):
        """Ingest sessions into Graphiti (sequential by default for correctness)."""
        safe_user_id = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
        ingest_stats: dict = {}
        start_time = time.time()

        async def _ingest_all():
            nonlocal ingest_stats
            # HTTP client timeout should be >= per-episode timeout to avoid premature termination
            # Use separate env var to avoid confusion with per-episode timeout (GRAPHITI_INGEST_TIMEOUT_S)
            ingest_client_timeout = float(
                os.getenv("GRAPHITI_HTTP_CLIENT_TIMEOUT_S", "900") or 900
            )
            client, _, async_clients = await self._get_graphiti(
                timeout=ingest_client_timeout
            )
            try:
                max_concurrent = int(os.getenv("GRAPHITI_INGEST_CONCURRENCY", "1") or 1)
                max_concurrent = max(1, max_concurrent)

                # Log ingestion configuration for debugging
                print(
                    f"    [GraphitiAdapter] Ingestion config: per_episode={self.ingest_timeout_s}s, http_client={ingest_client_timeout}s, concurrency={max_concurrent}"
                )
                semaphore = asyncio.Semaphore(max_concurrent)
                results = []
                failed_sessions = []

                async def _ingest_one(idx, s):
                    async with semaphore:
                        # Sync rate limit still applies globally across threads
                        self._rate_limiter.wait()

                        try:
                            episode_date = datetime.strptime(s["date"], "%Y-%m-%d")
                        except:
                            episode_date = datetime.utcnow()

                        try:
                            add_episode = client.add_episode(
                                name=f"Session on {s['date']}",
                                episode_body=s["content"],
                                source=EpisodeType.text,
                                source_description="chat history + " + s["date"],
                                reference_time=episode_date,
                                group_id=safe_user_id,
                            )
                            if self.ingest_timeout_s > 0:
                                result = await asyncio.wait_for(
                                    add_episode, timeout=self.ingest_timeout_s
                                )
                            else:
                                result = await add_episode
                            results.append(result)
                        except asyncio.TimeoutError:
                            print(
                                f"    [GraphitiAdapter] ⏱️ TIMEOUT for session {idx}: exceeded {self.ingest_timeout_s}s per-episode limit"
                            )
                            failed_sessions.append(
                                {
                                    "index": idx,
                                    "error": f"Timeout after {self.ingest_timeout_s}s",
                                }
                            )
                        except Exception as exc:
                            print(
                                f"    [GraphitiAdapter] ❌ Ingestion error for session {idx}: {exc}"
                            )
                            failed_sessions.append({"index": idx, "error": str(exc)})
                        # Simple progress marker in logs
                        if (idx + 1) % 5 == 0 or (idx + 1) == len(sessions):
                            print(
                                f"    [GraphitiAdapter] Progress: {idx + 1}/{len(sessions)} sessions for {safe_user_id}..."
                            )

                if max_concurrent == 1:
                    for i, s in enumerate(sessions):
                        await _ingest_one(i, s)
                else:
                    tasks = [_ingest_one(i, s) for i, s in enumerate(sessions)]
                    await asyncio.gather(*tasks)
            finally:
                for ac in async_clients:
                    try:
                        await ac.close()
                    except:
                        pass
                # Close Neo4j driver
                try:
                    client.graph_driver.close()
                except:
                    pass

            created_nodes = sum(len(r.nodes) for r in results)
            created_edges = sum(len(r.edges) for r in results)
            ingest_stats = {
                "memories_created": created_nodes,
                "links_created": created_edges,
                "memories_created_by_type": {
                    "episode": len(results),
                    "node": created_nodes,
                    "edge": created_edges,
                },
                "errors": failed_sessions,
                "timings_ms": {
                    "total": (time.time() - start_time) * 1000,
                },
            }

            # Final summary
            success_count = len(results)
            failed_count = len(failed_sessions)
            status_icon = "✅" if failed_count == 0 else "⚠️"
            print(
                f"    [GraphitiAdapter] {status_icon} Ingested {success_count}/{len(sessions)} sessions "
                f"for {safe_user_id}"
            )
            self._log_stage(
                safe_user_id,
                "stage1_ingestion_complete",
                {
                    "total_sessions": len(sessions),
                    "success_count": success_count,
                    "failed_count": failed_count,
                },
            )

        self._run_async(_ingest_all())
        self.last_ingest_stats = ingest_stats

    def _format_edge_date_range(self, edge) -> str:
        # Handle valid_at/invalid_at which might be None or datetime strings
        valid = edge.valid_at if edge.valid_at else "date unknown"
        invalid = edge.invalid_at if edge.invalid_at else "present"
        return f"{valid} - {invalid}"

    def _compose_search_context(self, edges: list, nodes: list) -> str:
        TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges. If the fact is about an event, the event takes place during this time.
# format: FACT (Date range: from - to)
<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
"""
        facts = [
            f"  - {edge.fact} ({self._format_edge_date_range(edge)})" for edge in edges
        ]
        entities = [f"  - {node.name}: {node.summary}" for node in nodes]
        return TEMPLATE.format(facts="\n".join(facts), entities="\n".join(entities))

    def _is_personamem_query(self, query: str) -> bool:
        return "Answer with only the letter" in query and "Options:" in query

    def _extract_personamem_question(self, query: str) -> str:
        text = query
        if "Question:" in text:
            text = text.split("Question:", 1)[1]
        if "Options:" in text:
            text = text.split("Options:", 1)[0]
        return text.strip()

    def query(self, user_id: str, query: str) -> str:
        safe_user_id = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)

        async def _run_rag():
            # Use shorter timeout for retrieval/generation
            retrieval_client_timeout = float(
                os.getenv("GRAPHITI_RETRIEVAL_TIMEOUT_S", "60") or 60
            )
            client, generator_client, async_clients = await self._get_graphiti(
                timeout=retrieval_client_timeout
            )
            try:
                is_personamem = self._is_personamem_query(query)
                search_query = query
                if is_personamem:
                    search_query = self._extract_personamem_question(query)
                if self.query_max_chars > 0:
                    search_query = search_query[: self.query_max_chars]
                start_retrieval = time.time()
                retrieval_error = None
                try:
                    search_task = client.search_(
                        query=search_query,
                        group_ids=[safe_user_id],
                        config=self.search_config,
                    )
                    if self.retrieval_timeout_s > 0:
                        results = await asyncio.wait_for(
                            search_task, timeout=self.retrieval_timeout_s
                        )
                    else:
                        results = await search_task
                except asyncio.TimeoutError:
                    retrieval_error = "timeout"
                    results = SimpleNamespace(edges=[], nodes=[])
                retrieval_duration_ms = (time.time() - start_retrieval) * 1000

                edges = results.edges
                nodes = results.nodes

                context = self._compose_search_context(edges, nodes)
                context_preview = context[:2000]

                if is_personamem:
                    system_prompt = (
                        "You are a helpful expert assistant answering multiple-choice questions "
                        "based on the provided context. Respond with only the option letter (a/b/c/d)."
                    )
                    user_prompt = f"""
Your task is to answer the multiple-choice question using the context. If the answer is not in the context, choose the most likely option.
    <CONTEXT>
    {context}
    </CONTEXT>
    <QUESTION>
    {query}
    </QUESTION>

Answer with only the letter (a/b/c/d).
"""
                else:
                    system_prompt = (
                        "You are a helpful expert assistant answering questions from lme_experiment users "
                        "based on the provided context."
                    )
                    user_prompt = f"""
Your task is to briefly answer the question. You are given the following context from the previous conversation. If you don't know how to answer the question, abstain from answering.
    <CONTEXT>
    {context}
    </CONTEXT>
    <QUESTION>
    {query}
    </QUESTION>

Answer:
"""

                self._rate_limiter.wait()
                start_generation = time.time()
                temperature = self.generator_temperature
                if self.generator_model.startswith(("gpt-5", "o1", "o3")):
                    temperature = None
                request = {
                    "model": self.generator_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                }
                if self.generator_max_tokens > 0:
                    if self.generator_model.startswith(("gpt-5", "o1", "o3")):
                        request["max_completion_tokens"] = self.generator_max_tokens
                    else:
                        request["max_tokens"] = self.generator_max_tokens

                response = await generator_client.chat.completions.create(**request)
                generation_duration_ms = (time.time() - start_generation) * 1000
                answer = response.choices[0].message.content or ""
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0

                self.last_query_stats = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "model": self.generator_model,
                    "temperature": self.generator_temperature,
                    "retrieval_ms": retrieval_duration_ms,
                    "generation_ms": generation_duration_ms,
                    "retrieval": {
                        "error": retrieval_error,
                        "context_preview": context_preview,
                        "vector_search": {
                            "top_k": self.search_limit,
                            "seeds": [],
                            "duration_ms": retrieval_duration_ms,
                        },
                        "graph_traversal": {
                            "max_hops": 0,
                            "nodes_visited": len(nodes),
                            "relationships_traversed": len(edges),
                            "final_ranked_nodes": [
                                getattr(node, "uuid", None)
                                for node in nodes
                                if getattr(node, "uuid", None)
                            ],
                            "duration_ms": 0,
                        },
                    },
                }

                self._log_stage(
                    safe_user_id,
                    "stage4_generation",
                    {
                        "query": query,
                        "answer": answer,
                        "edges_count": len(edges),
                        "nodes_count": len(nodes),
                        "retrieval_ms": retrieval_duration_ms,
                        "generation_ms": generation_duration_ms,
                    },
                )

                return answer
            except Exception as e:
                print(f"[GraphitiAdapter] RAG error for {safe_user_id}: {e}")
                self._log_stage(
                    safe_user_id,
                    "stage3_retrieval",
                    {"query": query, "status": "failed", "error": str(e)},
                )
                return f"Error: {e}"
            finally:
                for ac in async_clients:
                    try:
                        await ac.close()
                    except:
                        pass
                # Close Neo4j driver
                try:
                    client.graph_driver.close()
                except:
                    pass

        return self._run_async(_run_rag())

    def reset(self, user_id: str):
        # Clear graph for user.
        try:
            # Use a fresh sync driver for reset
            driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            with driver.session() as session:
                session.run(
                    "MATCH (n) WHERE n.group_id = $uid DETACH DELETE n", uid=user_id
                )
            driver.close()
        except Exception as e:
            print(f"[GraphitiAdapter] Reset error: {e}")


# Backwards-compatible alias
ZepAdapter = GraphitiAdapter
