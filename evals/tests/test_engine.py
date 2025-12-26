"""
Tests for Engine and Executor integration.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from evals.core.models import (
    TestCase,
    QueryResult,
    Session,
    MetricResult,
    EvalResult,
)
from evals.core.interfaces import AdapterCapabilities
from evals.engine import (
    Engine,
    EngineConfig,
    RunResult,
    AsyncExecutor,
    ExecutorConfig,
    TokenBucket,
    create_default_engine,
)
from evals.metrics import BinaryExactMatch, OptionExtractor


# === Mock Adapter ===


class MockAdapter:
    """Mock adapter for testing."""

    name = "mock_adapter"
    capabilities = AdapterCapabilities(
        supports_async=False,
        supports_bulk_ingest=True,
        supports_retrieval_items=False,
        supports_context_text=False,
    )

    def __init__(self, answers: dict = None):
        """
        Args:
            answers: Dict mapping query strings to answers
        """
        self._answers = answers or {}
        self._reset_count = 0
        self._ingest_count = 0
        self._query_count = 0

    def reset(self, user_id: str) -> None:
        self._reset_count += 1

    def add_sessions(self, user_id: str, sessions) -> None:
        self._ingest_count += len(sessions)

    def query(self, user_id: str, query: str, *, trace: bool = True) -> QueryResult:
        self._query_count += 1
        answer = self._answers.get(query, "default answer")
        return QueryResult(answer=answer)

    async def areset(self, user_id: str) -> None:
        self.reset(user_id)

    async def aadd_sessions(self, user_id: str, sessions) -> None:
        self.add_sessions(user_id, sessions)

    async def aquery(
        self, user_id: str, query: str, *, trace: bool = True
    ) -> QueryResult:
        return self.query(user_id, query, trace=trace)


class MockBenchmark:
    """Mock benchmark for testing."""

    name = "mock_benchmark"
    version = "1.0"

    def __init__(self, cases: list = None):
        self._cases = cases or []

    def load(self, *, variant=None):
        return self._cases

    def default_metrics(self):
        return ["binary_exact_match"]

    def sample(self, sizes, *, seed=None, variant=None):
        return self._cases[: sum(sizes.values())]


# === TokenBucket Tests ===


class TestTokenBucket:
    """Tests for TokenBucket rate limiter."""

    @pytest.mark.asyncio
    async def test_immediate_acquire_with_burst(self):
        """Should acquire immediately when burst capacity available."""
        bucket = TokenBucket(rate_per_sec=10.0, burst=5.0)

        start = asyncio.get_event_loop().time()
        await bucket.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.1  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Should enforce rate limiting after burst exhausted."""
        bucket = TokenBucket(rate_per_sec=100.0, burst=2.0)

        # Exhaust burst
        await bucket.acquire()
        await bucket.acquire()

        # Third acquire should take some time
        start = asyncio.get_event_loop().time()
        await bucket.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        # Should take at least 1/100 second to get next token
        assert elapsed >= 0.005  # Allow some tolerance


# === AsyncExecutor Tests ===


class TestAsyncExecutor:
    """Tests for AsyncExecutor."""

    def test_executor_config_defaults(self):
        """ExecutorConfig should have sensible defaults."""
        config = ExecutorConfig()

        assert config.concurrency == 10
        assert config.rate_limit_rpm is None
        assert config.retry_max_attempts == 5

    @pytest.mark.asyncio
    async def test_execute_query(self):
        """Should execute query through adapter."""
        executor = AsyncExecutor(config=ExecutorConfig(concurrency=5))
        adapter = MockAdapter(answers={"What is 2+2?": "4"})

        tc = TestCase(
            id="test_1",
            benchmark="test",
            user_id="user_1",
            query="What is 2+2?",
            sessions=(),
        )

        result = await executor.execute_query(adapter, tc)

        assert result.answer == "4"
        assert adapter._query_count == 1

    @pytest.mark.asyncio
    async def test_run_test_case_full_pipeline(self):
        """Should run full pipeline: reset → ingest → query → evaluate."""
        executor = AsyncExecutor(config=ExecutorConfig(concurrency=5))
        adapter = MockAdapter(answers={"What is my color?": "blue"})

        tc = TestCase(
            id="test_1",
            benchmark="test",
            user_id="user_1",
            query="What is my color?",
            sessions=(Session(content="I like blue"),),
            reference_answer="blue",
        )

        metrics = [BinaryExactMatch()]

        result = await executor.run_test_case(
            adapter=adapter,
            test_case=tc,
            metrics=metrics,
            resources={},
            run_id="test_run",
        )

        assert isinstance(result, EvalResult)
        assert result.run_id == "test_run"
        assert result.test_case_id == "test_1"
        assert adapter._reset_count == 1
        assert adapter._ingest_count == 1
        assert adapter._query_count == 1

    @pytest.mark.asyncio
    async def test_run_cases_parallel(self):
        """Should run multiple cases in parallel."""
        executor = AsyncExecutor(config=ExecutorConfig(concurrency=3))
        adapter = MockAdapter(answers={"q1": "a1", "q2": "a2", "q3": "a3"})

        cases = [
            TestCase(
                id="tc_1",
                benchmark="test",
                user_id="u1",
                query="q1",
                sessions=(),
                reference_answer="a1",
            ),
            TestCase(
                id="tc_2",
                benchmark="test",
                user_id="u2",
                query="q2",
                sessions=(),
                reference_answer="a2",
            ),
            TestCase(
                id="tc_3",
                benchmark="test",
                user_id="u3",
                query="q3",
                sessions=(),
                reference_answer="a3",
            ),
        ]

        metrics = [BinaryExactMatch()]

        results = await executor.run_cases(
            adapter=adapter,
            test_cases=cases,
            metrics=metrics,
            resources={},
            run_id="parallel_run",
        )

        assert len(results) == 3
        assert adapter._query_count == 3


# === Engine Tests ===


class TestEngine:
    """Tests for Engine."""

    def test_engine_config_defaults(self):
        """EngineConfig should have sensible defaults."""
        config = EngineConfig()

        assert config.concurrency == 10
        assert config.verbose is False
        assert config.sample_sizes is None

    def test_create_default_engine(self):
        """create_default_engine should register all built-in metrics."""
        engine = create_default_engine(concurrency=5)

        assert engine.get_metric("binary_exact_match") is not None
        assert engine.get_metric("option_extractor") is not None
        assert engine.get_metric("llm_binary_judge") is not None
        assert engine.get_metric("abstention_accuracy") is not None

    def test_register_metric(self):
        """Should register custom metrics."""
        engine = Engine()
        metric = BinaryExactMatch()

        engine.register_metric(metric)

        assert engine.get_metric("binary_exact_match") is metric

    @pytest.mark.asyncio
    async def test_run_evaluation(self):
        """Should run full evaluation with benchmark."""
        engine = create_default_engine(concurrency=2, verbose=False)

        adapter = MockAdapter(
            answers={
                "What color?": "blue",
                "What food?": "pizza",
            }
        )

        cases = [
            TestCase(
                id="tc_1",
                benchmark="mock",
                user_id="u1",
                query="What color?",
                sessions=(),
                reference_answer="blue",
            ),
            TestCase(
                id="tc_2",
                benchmark="mock",
                user_id="u2",
                query="What food?",
                sessions=(),
                reference_answer="pizza",
            ),
        ]

        benchmark = MockBenchmark(cases=cases)

        result = await engine.run(
            benchmark=benchmark,
            adapter=adapter,
            metric_names=["binary_exact_match"],
            resources={},
        )

        assert isinstance(result, RunResult)
        assert result.total_cases == 2
        assert result.passed_cases == 2
        assert result.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_run_with_failures(self):
        """Should handle failing test cases."""
        engine = create_default_engine(concurrency=2)

        adapter = MockAdapter(
            answers={
                "What color?": "blue",
                "What food?": "sushi",  # Wrong answer
            }
        )

        cases = [
            TestCase(
                id="tc_1",
                benchmark="mock",
                user_id="u1",
                query="What color?",
                sessions=(),
                reference_answer="blue",
            ),
            TestCase(
                id="tc_2",
                benchmark="mock",
                user_id="u2",
                query="What food?",
                sessions=(),
                reference_answer="pizza",  # Expected pizza, got sushi
            ),
        ]

        benchmark = MockBenchmark(cases=cases)

        result = await engine.run(
            benchmark=benchmark,
            adapter=adapter,
            metric_names=["binary_exact_match"],
            resources={},
        )

        assert result.total_cases == 2
        assert result.passed_cases == 1
        assert result.failed_cases == 1
        assert result.pass_rate == 0.5

    @pytest.mark.asyncio
    async def test_run_single(self):
        """Should run single test case for debugging."""
        engine = create_default_engine()
        adapter = MockAdapter(answers={"test query": "test answer"})

        tc = TestCase(
            id="single_tc",
            benchmark="test",
            user_id="user_1",
            query="test query",
            sessions=(),
            reference_answer="test answer",
        )

        result = await engine.run_single(
            adapter=adapter,
            test_case=tc,
            metric_names=["binary_exact_match"],
        )

        assert isinstance(result, EvalResult)
        assert result.passed is True


class TestRunResult:
    """Tests for RunResult aggregation."""

    def test_summary(self):
        """Should produce human-readable summary."""
        result = RunResult(
            run_id="test_run",
            run_spec=MagicMock(),
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            finished_at=datetime(2025, 1, 1, 12, 0, 30),
            results=[],
            total_cases=100,
            passed_cases=85,
            failed_cases=10,
            error_cases=5,
            by_question_type={
                "type_a": {"total": 50, "passed": 45, "pass_rate": 0.9},
                "type_b": {"total": 50, "passed": 40, "pass_rate": 0.8},
            },
        )

        summary = result.summary()

        assert "test_run" in summary
        assert "85" in summary
        assert "85.0%" in summary

    def test_pass_rate_zero_cases(self):
        """Should handle zero cases gracefully."""
        result = RunResult(
            run_id="empty_run",
            run_spec=MagicMock(),
            started_at=datetime.now(),
            finished_at=datetime.now(),
            results=[],
            total_cases=0,
        )

        assert result.pass_rate == 0.0
