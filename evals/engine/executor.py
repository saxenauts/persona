"""Async executor with rate limiting for high-throughput evaluation."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping, Optional, Sequence

from ..core.models import TestCase, QueryResult, EvalResult, MetricResult, Session
from ..core.interfaces import MemorySystemAdapter, Metric, AdapterCapabilities


@dataclass
class TokenBucket:
    """
    Token bucket rate limiter for async operations.

    Design: Smooth rate limiting that allows bursting while maintaining
    average rate. Better than strict per-second limits for API calls.

    Usage:
        limiter = TokenBucket(rate_per_sec=10.0, burst=20.0)
        await limiter.acquire()  # blocks if no tokens available
    """

    rate_per_sec: float  # Tokens added per second
    burst: float  # Maximum tokens (bucket capacity)

    _tokens: float = field(default=0.0, init=False)
    _last_update: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def acquire(self, tokens: float = 1.0) -> None:
        """
        Acquire tokens, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire (default 1)
        """
        async with self._lock:
            now = time.monotonic()

            # Initialize on first use
            if self._last_update == 0.0:
                self._last_update = now
                self._tokens = self.burst

            # Add tokens based on elapsed time
            elapsed = now - self._last_update
            self._last_update = now
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate_per_sec)

            # If we have enough tokens, consume and return
            if self._tokens >= tokens:
                self._tokens -= tokens
                return

            # Calculate wait time for needed tokens
            needed = tokens - self._tokens
            wait_seconds = needed / max(1e-9, self.rate_per_sec)

        # Wait outside the lock
        await asyncio.sleep(wait_seconds)
        await self.acquire(tokens)

    @property
    def available(self) -> float:
        """Current available tokens (approximate, for monitoring)."""
        return self._tokens


async def with_retry(
    coro_factory: Callable[[], Any],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
) -> Any:
    """
    Retry async operation with exponential backoff.

    Design: Only retries on rate limit errors (429). Other errors propagate.
    Uses jitter to prevent thundering herd.

    Args:
        coro_factory: Callable that returns a coroutine (not an awaited result!)
        max_attempts: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        jitter: Random jitter factor (0-1)

    Returns:
        Result of successful coroutine execution

    Raises:
        Last exception if all retries fail
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check if this is a rate limit error
            is_rate_limit = any(
                indicator in error_str
                for indicator in (
                    "429",
                    "rate limit",
                    "ratelimit",
                    "too many requests",
                    "quota",
                )
            )

            # Don't retry non-rate-limit errors
            if not is_rate_limit:
                raise

            # Don't retry on last attempt
            if attempt == max_attempts - 1:
                raise

            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2**attempt), max_delay)
            delay += random.random() * jitter * delay

            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Retry logic error")


@dataclass
class ExecutorConfig:
    """Configuration for the async executor."""

    concurrency: int = 10  # Max concurrent operations
    rate_limit_rpm: Optional[int] = None  # Requests per minute (None = unlimited)
    retry_max_attempts: int = 5
    retry_base_delay: float = 1.0
    timeout_seconds: float = 300.0  # Per-operation timeout


class AsyncExecutor:
    """
    Async executor for parallel evaluation with rate limiting.

    Design Principles:
    - Semaphore controls concurrency (prevents overwhelming backends)
    - Per-provider rate limiters (different APIs have different limits)
    - Retry with backoff for transient errors
    - Graceful degradation on failures

    Usage:
        executor = AsyncExecutor(
            concurrency=10,
            rate_limiters={"openai": TokenBucket(rate_per_sec=3.0, burst=10.0)}
        )

        results = await executor.run_cases(adapter, test_cases, metrics)
    """

    def __init__(
        self,
        config: Optional[ExecutorConfig] = None,
        rate_limiters: Optional[Mapping[str, TokenBucket]] = None,
    ):
        self.config = config or ExecutorConfig()
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        self._rate_limiters = dict(rate_limiters or {})

        # Create default rate limiter if rpm specified
        if self.config.rate_limit_rpm and "default" not in self._rate_limiters:
            self._rate_limiters["default"] = TokenBucket(
                rate_per_sec=self.config.rate_limit_rpm / 60.0,
                burst=min(10.0, self.config.rate_limit_rpm / 6.0),
            )

    def get_limiter(self, name: str) -> Optional[TokenBucket]:
        """Get rate limiter for a provider/adapter."""
        return self._rate_limiters.get(name) or self._rate_limiters.get("default")

    def _format_query_with_options(self, test_case: TestCase) -> str:
        query = test_case.query
        if test_case.options and test_case.correct_option:
            options_text = "\n".join(
                f"({k}) {v}" for k, v in sorted(test_case.options.items())
            )
            query = f"{query}\n\nWhich response is most appropriate based on what you know about me?\n{options_text}\n\nRespond with just the letter (a, b, c, or d)."
        return query

    async def execute_query(
        self,
        adapter: MemorySystemAdapter,
        test_case: TestCase,
    ) -> QueryResult:
        limiter = self.get_limiter(adapter.name)
        formatted_query = self._format_query_with_options(test_case)

        async with self._semaphore:
            if limiter:
                await limiter.acquire()

            async def do_query():
                if adapter.capabilities.supports_async:
                    return await adapter.aquery(test_case.user_id, formatted_query)
                else:
                    return await asyncio.to_thread(
                        adapter.query, test_case.user_id, formatted_query
                    )

            try:
                return await asyncio.wait_for(
                    with_retry(
                        do_query,
                        max_attempts=self.config.retry_max_attempts,
                        base_delay=self.config.retry_base_delay,
                    ),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                return QueryResult(
                    answer="",
                    error=f"Timeout after {self.config.timeout_seconds}s",
                )
            except Exception as e:
                return QueryResult(
                    answer="",
                    error=str(e),
                )

    async def execute_ingestion(
        self,
        adapter: MemorySystemAdapter,
        user_id: str,
        sessions: Sequence[Session],
    ) -> Optional[str]:
        """
        Execute session ingestion with rate limiting.

        Returns:
            None on success, error message on failure
        """
        limiter = self.get_limiter(adapter.name)

        async with self._semaphore:
            if limiter:
                await limiter.acquire()

            async def do_ingest():
                if adapter.capabilities.supports_async:
                    await adapter.aadd_sessions(user_id, sessions)
                else:
                    await asyncio.to_thread(adapter.add_sessions, user_id, sessions)

            try:
                await asyncio.wait_for(
                    with_retry(do_ingest),
                    timeout=self.config.timeout_seconds,
                )
                return None
            except Exception as e:
                return str(e)

    async def evaluate_metrics(
        self,
        test_case: TestCase,
        query_result: QueryResult,
        metrics: Sequence[Metric],
        resources: Mapping[str, Any],
        adapter_caps: AdapterCapabilities,
    ) -> Sequence[MetricResult]:
        """
        Evaluate all applicable metrics for a test case.

        Design:
        - Skips metrics that require capabilities the adapter doesn't have
        - Runs metrics in parallel (they're typically LLM calls)
        - Catches individual metric failures gracefully
        """
        results = []

        async def run_metric(metric: Metric) -> Optional[MetricResult]:
            # Check if adapter supports required capabilities
            required = metric.required_capabilities()
            if (
                required.supports_retrieval_items
                and not adapter_caps.supports_retrieval_items
            ):
                return MetricResult(
                    metric=metric.name,
                    kind=metric.kind,
                    score_type=metric.score_type,
                    score=0.0,
                    passed=False,
                    reason=f"Skipped: adapter doesn't support retrieval items",
                )

            try:
                return await metric.evaluate(
                    test_case, query_result, resources=resources
                )
            except Exception as e:
                return MetricResult(
                    metric=metric.name,
                    kind=metric.kind,
                    score_type=metric.score_type,
                    score=0.0,
                    passed=False,
                    reason=f"Error: {e}",
                )

        # Run metrics in parallel
        tasks = [run_metric(m) for m in metrics]
        metric_results = await asyncio.gather(*tasks)

        return [r for r in metric_results if r is not None]

    async def run_test_case(
        self,
        adapter: MemorySystemAdapter,
        test_case: TestCase,
        metrics: Sequence[Metric],
        resources: Mapping[str, Any],
        run_id: str,
    ) -> EvalResult:
        """
        Run full evaluation pipeline for a single test case.

        Pipeline: reset → ingest → query → evaluate metrics → return result
        """
        started = datetime.utcnow()

        # Reset adapter state
        try:
            if adapter.capabilities.supports_async:
                await adapter.areset(test_case.user_id)
            else:
                await asyncio.to_thread(adapter.reset, test_case.user_id)
        except Exception as e:
            finished = datetime.utcnow()
            return EvalResult(
                run_id=run_id,
                adapter=adapter.name,
                benchmark=test_case.benchmark,
                test_case_id=test_case.id,
                question_type=test_case.question_type,
                query_result=QueryResult(answer="", error=f"Reset failed: {e}"),
                passed=False,
                started_at=started,
                finished_at=finished,
            )

        # Ingest sessions
        if test_case.sessions:
            ingest_error = await self.execute_ingestion(
                adapter, test_case.user_id, test_case.sessions
            )
            if ingest_error:
                finished = datetime.utcnow()
                return EvalResult(
                    run_id=run_id,
                    adapter=adapter.name,
                    benchmark=test_case.benchmark,
                    test_case_id=test_case.id,
                    question_type=test_case.question_type,
                    query_result=QueryResult(
                        answer="", error=f"Ingestion failed: {ingest_error}"
                    ),
                    passed=False,
                    started_at=started,
                    finished_at=finished,
                )

        # Execute query
        query_result = await self.execute_query(adapter, test_case)

        # Evaluate metrics
        metric_results = await self.evaluate_metrics(
            test_case, query_result, metrics, resources, adapter.capabilities
        )

        # Determine overall pass/fail
        passed = all(m.passed for m in metric_results) if metric_results else False

        finished = datetime.utcnow()

        return EvalResult(
            run_id=run_id,
            adapter=adapter.name,
            benchmark=test_case.benchmark,
            test_case_id=test_case.id,
            question_type=test_case.question_type,
            query_result=query_result,
            metric_results=tuple(metric_results),
            passed=passed,
            started_at=started,
            finished_at=finished,
        )

    async def run_cases(
        self,
        adapter: MemorySystemAdapter,
        test_cases: Sequence[TestCase],
        metrics: Sequence[Metric],
        resources: Mapping[str, Any],
        run_id: str,
        *,
        on_progress: Optional[Callable[[int, int, EvalResult], None]] = None,
    ) -> Sequence[EvalResult]:
        """
        Run evaluation on multiple test cases in parallel.

        Args:
            adapter: Memory system to evaluate
            test_cases: Test cases to run
            metrics: Metrics to evaluate
            resources: Shared resources (LLM clients, etc.)
            run_id: Unique run identifier
            on_progress: Optional callback(completed, total, result) for progress

        Returns:
            Sequence of EvalResults
        """
        results: list[EvalResult] = []
        total = len(test_cases)
        completed = 0

        async def run_one(tc: TestCase) -> EvalResult:
            nonlocal completed
            result = await self.run_test_case(adapter, tc, metrics, resources, run_id)
            completed += 1
            if on_progress:
                on_progress(completed, total, result)
            return result

        tasks: list[asyncio.Task[EvalResult]] = []
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(run_one(tc)) for tc in test_cases]
            results = [t.result() for t in tasks]
        except ExceptionGroup as eg:
            for exc in eg.exceptions:
                print(f"Task failed: {exc}")
            results = [
                t.result() for t in tasks if not t.cancelled() and t.exception() is None
            ]

        return results

    async def run_query_only(
        self,
        adapter: MemorySystemAdapter,
        test_case: TestCase,
        metrics: Sequence[Metric],
        resources: Mapping[str, Any],
        run_id: str,
    ) -> EvalResult:
        """
        Run query and metrics only (skip reset/ingest).

        Used when sessions are pre-ingested via run_cases_grouped().
        """
        started = datetime.utcnow()

        # Execute query
        query_result = await self.execute_query(adapter, test_case)

        # Evaluate metrics
        metric_results = await self.evaluate_metrics(
            test_case, query_result, metrics, resources, adapter.capabilities
        )

        # Determine overall pass/fail
        passed = all(m.passed for m in metric_results) if metric_results else False

        finished = datetime.utcnow()

        return EvalResult(
            run_id=run_id,
            adapter=adapter.name,
            benchmark=test_case.benchmark,
            test_case_id=test_case.id,
            question_type=test_case.question_type,
            query_result=query_result,
            metric_results=tuple(metric_results),
            passed=passed,
            started_at=started,
            finished_at=finished,
        )

    async def run_cases_grouped(
        self,
        adapter: MemorySystemAdapter,
        test_cases: Sequence[TestCase],
        metrics: Sequence[Metric],
        resources: Mapping[str, Any],
        run_id: str,
        *,
        on_progress: Optional[Callable[[int, int, EvalResult], None]] = None,
    ) -> Sequence[EvalResult]:
        """
        Run evaluation with pre-ingestion optimization.

        Groups test cases by user_id, ingests each user's sessions ONCE,
        then runs all queries for that user against pre-ingested data.

        Optimization: Reduces ingestion from O(n) to O(unique_users).
        For PersonaMem with ~16 questions per shared context, this is ~16x faster.

        Args:
            adapter: Memory system to evaluate
            test_cases: Test cases to run
            metrics: Metrics to evaluate
            resources: Shared resources (LLM clients, etc.)
            run_id: Unique run identifier
            on_progress: Optional callback(completed, total, result) for progress

        Returns:
            Sequence of EvalResults
        """
        from collections import defaultdict

        # Group test cases by user_id
        by_user: dict[str, list[TestCase]] = defaultdict(list)
        for tc in test_cases:
            by_user[tc.user_id].append(tc)

        total = len(test_cases)
        completed = 0
        all_results: list[EvalResult] = []

        # Process each user group sequentially (but queries within group run parallel)
        for user_id, user_cases in by_user.items():
            # Get sessions from first test case that has them
            sessions = None
            for tc in user_cases:
                if tc.sessions:
                    sessions = tc.sessions
                    break

            # Reset adapter for this user
            try:
                if adapter.capabilities.supports_async:
                    await adapter.areset(user_id)
                else:
                    await asyncio.to_thread(adapter.reset, user_id)
            except Exception as e:
                # Mark all cases for this user as failed
                for tc in user_cases:
                    err_result = EvalResult(
                        run_id=run_id,
                        adapter=adapter.name,
                        benchmark=tc.benchmark,
                        test_case_id=tc.id,
                        question_type=tc.question_type,
                        query_result=QueryResult(answer="", error=f"Reset failed: {e}"),
                        passed=False,
                        started_at=datetime.utcnow(),
                        finished_at=datetime.utcnow(),
                    )
                    all_results.append(err_result)
                    completed += 1
                    if on_progress:
                        on_progress(completed, total, err_result)
                continue

            # Ingest sessions ONCE for this user
            if sessions:
                ingest_error = await self.execute_ingestion(adapter, user_id, sessions)
                if ingest_error:
                    # Mark all cases for this user as failed
                    for tc in user_cases:
                        err_result = EvalResult(
                            run_id=run_id,
                            adapter=adapter.name,
                            benchmark=tc.benchmark,
                            test_case_id=tc.id,
                            question_type=tc.question_type,
                            query_result=QueryResult(
                                answer="", error=f"Ingestion failed: {ingest_error}"
                            ),
                            passed=False,
                            started_at=datetime.utcnow(),
                            finished_at=datetime.utcnow(),
                        )
                        all_results.append(err_result)
                        completed += 1
                        if on_progress:
                            on_progress(completed, total, err_result)
                    continue

            # Run all queries for this user in parallel (no more ingestion)
            async def run_query(tc: TestCase) -> EvalResult:
                nonlocal completed
                result = await self.run_query_only(
                    adapter, tc, metrics, resources, run_id
                )
                completed += 1
                if on_progress:
                    on_progress(completed, total, result)
                return result

            tasks: list[asyncio.Task[EvalResult]] = []
            try:
                async with asyncio.TaskGroup() as tg:
                    tasks = [tg.create_task(run_query(tc)) for tc in user_cases]
                user_results = [t.result() for t in tasks]
            except ExceptionGroup as eg:
                for exc in eg.exceptions:
                    print(f"Query task failed: {exc}")
                user_results = [
                    t.result()
                    for t in tasks
                    if not t.cancelled() and t.exception() is None
                ]

            all_results.extend(user_results)

        return all_results
