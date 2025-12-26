"""
Tests for core models: TestCase, QueryResult, Session, etc.
"""

import pytest
from datetime import datetime

from evals.core.models import (
    Session,
    TestCase,
    RetrievedItem,
    Usage,
    Latency,
    QueryResult,
    MetricResult,
    EvalResult,
    RunSpec,
)


class TestSession:
    """Tests for Session dataclass."""

    def test_minimal_session(self):
        """Session with just content."""
        s = Session(content="Hello, I like pizza")
        assert s.content == "Hello, I like pizza"
        assert s.date is None
        assert s.metadata == {}

    def test_full_session(self):
        """Session with all fields."""
        s = Session(
            content="User said something",
            date="2025-01-15",
            metadata={"source": "chat", "turn_count": 5},
        )
        assert s.content == "User said something"
        assert s.date == "2025-01-15"
        assert s.metadata["source"] == "chat"

    def test_session_is_immutable(self):
        """Session should be frozen (immutable)."""
        s = Session(content="test")
        with pytest.raises(Exception):  # FrozenInstanceError
            s.content = "changed"


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_minimal_test_case(self):
        """TestCase with required fields only."""
        tc = TestCase(
            id="tc_001",
            benchmark="test",
            user_id="user_123",
            query="What is my favorite color?",
            sessions=(),
        )
        assert tc.id == "tc_001"
        assert tc.benchmark == "test"
        assert tc.reference_answer is None
        assert tc.options is None

    def test_full_test_case(self):
        """TestCase with all fields."""
        tc = TestCase(
            id="tc_002",
            benchmark="personamem",
            user_id="user_456",
            query="What is the capital of France?",
            sessions=(Session(content="Paris is beautiful", date="2025-01-01"),),
            reference_answer="Paris",
            question_type="single-session-user",
            tags=("geography", "factual"),
            options={"a": "London", "b": "Paris", "c": "Berlin", "d": "Madrid"},
            correct_option="b",
            metadata={"difficulty": "easy"},
        )
        assert tc.reference_answer == "Paris"
        assert tc.question_type == "single-session-user"
        assert tc.correct_option == "b"
        assert "geography" in tc.tags

    def test_test_case_with_multiple_sessions(self):
        """TestCase with multiple sessions."""
        sessions = (
            Session(content="Day 1 conversation", date="2025-01-01"),
            Session(content="Day 2 conversation", date="2025-01-02"),
            Session(content="Day 3 conversation", date="2025-01-03"),
        )
        tc = TestCase(
            id="tc_003",
            benchmark="longmemeval",
            user_id="user_789",
            query="What happened on day 2?",
            sessions=sessions,
        )
        assert len(tc.sessions) == 3
        assert tc.sessions[1].date == "2025-01-02"


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_minimal_query_result(self):
        """QueryResult with just answer."""
        qr = QueryResult(answer="The answer is 42")
        assert qr.answer == "The answer is 42"
        assert qr.retrieved == ()
        assert qr.error is None

    def test_query_result_with_error(self):
        """QueryResult representing a failed query."""
        qr = QueryResult(answer="", error="Connection timeout")
        assert qr.answer == ""
        assert qr.error == "Connection timeout"

    def test_query_result_with_retrieval(self):
        """QueryResult with retrieved items."""
        items = (
            RetrievedItem(id="node_1", text="Relevant fact 1", score=0.95, rank=1),
            RetrievedItem(id="node_2", text="Relevant fact 2", score=0.87, rank=2),
        )
        qr = QueryResult(
            answer="Based on your preferences...",
            retrieved=items,
            context_text="Context: Relevant fact 1. Relevant fact 2.",
        )
        assert len(qr.retrieved) == 2
        assert qr.retrieved[0].score == 0.95
        assert qr.context_text is not None

    def test_query_result_with_usage(self):
        """QueryResult with token usage tracking."""
        qr = QueryResult(
            answer="Response",
            usage=Usage(
                provider="openai",
                model="gpt-4o-mini",
                prompt_tokens=150,
                completion_tokens=50,
                total_tokens=200,
            ),
        )
        assert qr.usage.model == "gpt-4o-mini"
        assert qr.usage.total_tokens == 200

    def test_query_result_with_latency(self):
        """QueryResult with timing breakdown."""
        qr = QueryResult(
            answer="Response",
            latency=Latency(
                retrieval_ms=120.5,
                generation_ms=350.2,
                total_ms=470.7,
            ),
        )
        assert qr.latency.retrieval_ms == 120.5
        assert qr.latency.total_ms == 470.7


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_binary_metric_pass(self):
        """Binary metric that passed."""
        mr = MetricResult(
            metric="exact_match",
            kind="generation",
            score_type="binary",
            score=1.0,
            passed=True,
            reason="Exact string match",
        )
        assert mr.passed is True
        assert mr.score == 1.0

    def test_binary_metric_fail(self):
        """Binary metric that failed."""
        mr = MetricResult(
            metric="exact_match",
            kind="generation",
            score_type="binary",
            score=0.0,
            passed=False,
            reason="Strings do not match",
        )
        assert mr.passed is False
        assert mr.score == 0.0

    def test_continuous_metric_with_threshold(self):
        """Continuous metric with threshold."""
        mr = MetricResult(
            metric="context_precision",
            kind="retrieval",
            score_type="continuous",
            score=0.85,
            passed=True,
            threshold=0.7,
            reason="Score 0.850 ≥ threshold 0.7",
        )
        assert mr.score_type == "continuous"
        assert mr.threshold == 0.7
        assert mr.passed is True


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_eval_result_passed(self):
        """EvalResult where all metrics passed."""
        er = EvalResult(
            run_id="run_abc123",
            adapter="persona",
            benchmark="personamem",
            test_case_id="tc_001",
            question_type="single-session-user",
            query_result=QueryResult(answer="blue"),
            metric_results=(
                MetricResult(
                    metric="option_extractor",
                    kind="generation",
                    score_type="binary",
                    score=1.0,
                    passed=True,
                    reason="Correct option",
                ),
            ),
            passed=True,
        )
        assert er.passed is True
        assert er.adapter == "persona"

    def test_eval_result_failed(self):
        """EvalResult where a metric failed."""
        er = EvalResult(
            run_id="run_xyz789",
            adapter="graphiti",
            benchmark="longmemeval",
            test_case_id="tc_002",
            query_result=QueryResult(answer="wrong answer"),
            metric_results=(
                MetricResult(
                    metric="llm_binary_judge",
                    kind="generation",
                    score_type="binary",
                    score=0.0,
                    passed=False,
                    reason="Judge said NO",
                ),
            ),
            passed=False,
        )
        assert er.passed is False


class TestRunSpec:
    """Tests for RunSpec dataclass."""

    def test_run_spec(self):
        """RunSpec captures all parameters for reproducibility."""
        rs = RunSpec(
            run_id="run_001",
            created_at=datetime(2025, 1, 15, 12, 0, 0),
            adapters=("persona", "graphiti"),
            benchmarks=("personamem",),
            metric_names=("option_extractor", "context_precision"),
            concurrency=20,
            rate_limit_rpm=60,
            random_seed=42,
        )
        assert rs.run_id == "run_001"
        assert rs.concurrency == 20
        assert rs.random_seed == 42
        assert "persona" in rs.adapters
