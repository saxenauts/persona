"""
Tests for metrics: exact_match, retrieval, llm_judge.
"""

import pytest
import asyncio

from evals.core.models import TestCase, QueryResult, Session, MetricResult
from evals.metrics import (
    BinaryExactMatch,
    OptionExtractor,
    ContainsAnswer,
    ContextPrecision,
    ContextRecall,
    LLMBinaryJudge,
    AbstentionAccuracy,
    SemanticSimilarity,
    AllOf,
    AnyOf,
    ThresholdGate,
)
from evals.core.models import RetrievedItem


# === Fixtures ===


@pytest.fixture
def simple_test_case():
    """Basic test case for testing metrics."""
    return TestCase(
        id="test_001",
        benchmark="test",
        user_id="user_1",
        query="What is my favorite color?",
        sessions=(Session(content="I love blue"),),
        reference_answer="blue",
        question_type="single-session-user",
    )


@pytest.fixture
def multiple_choice_test_case():
    """Test case with multiple choice options."""
    return TestCase(
        id="test_002",
        benchmark="personamem",
        user_id="user_2",
        query="What is my favorite color?",
        sessions=(Session(content="I love blue"),),
        reference_answer="b",
        options={"a": "Red", "b": "Blue", "c": "Green", "d": "Yellow"},
        correct_option="b",
        question_type="recall_user_shared_facts",
    )


@pytest.fixture
def abstention_test_case():
    """Test case for abstention/unanswerable questions."""
    return TestCase(
        id="test_abs_001",
        benchmark="longmemeval",
        user_id="user_3",
        query="What is my cat's name?",
        sessions=(Session(content="I have a dog named Max"),),
        reference_answer="Information not provided",
        question_type="single-session-user",
        tags=("abstention",),
        metadata={"is_abstention": True},
    )


# === BinaryExactMatch Tests ===


class TestBinaryExactMatch:
    """Tests for BinaryExactMatch metric."""

    @pytest.mark.asyncio
    async def test_exact_match_pass(self, simple_test_case):
        """Exact match should pass when strings match."""
        metric = BinaryExactMatch()
        qr = QueryResult(answer="blue")

        result = await metric.evaluate(simple_test_case, qr, resources={})

        assert result.passed is True
        assert result.score == 1.0
        assert result.metric == "binary_exact_match"

    @pytest.mark.asyncio
    async def test_exact_match_fail(self, simple_test_case):
        """Exact match should fail when strings don't match."""
        metric = BinaryExactMatch()
        qr = QueryResult(answer="red")

        result = await metric.evaluate(simple_test_case, qr, resources={})

        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_exact_match_case_insensitive(self, simple_test_case):
        """Exact match should be case insensitive by default."""
        metric = BinaryExactMatch()
        qr = QueryResult(answer="BLUE")

        result = await metric.evaluate(simple_test_case, qr, resources={})

        assert result.passed is True


# === OptionExtractor Tests ===


class TestOptionExtractor:
    """Tests for OptionExtractor metric."""

    @pytest.mark.asyncio
    async def test_option_extractor_letter_only(self, multiple_choice_test_case):
        """Should extract option letter from response."""
        metric = OptionExtractor()
        qr = QueryResult(answer="b")

        result = await metric.evaluate(multiple_choice_test_case, qr, resources={})

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_option_extractor_letter_with_text(self, multiple_choice_test_case):
        """Should extract option letter from longer response."""
        metric = OptionExtractor()
        qr = QueryResult(answer="The answer is (b) Blue")

        result = await metric.evaluate(multiple_choice_test_case, qr, resources={})

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_option_extractor_wrong_option(self, multiple_choice_test_case):
        """Should fail when wrong option is selected."""
        metric = OptionExtractor()
        qr = QueryResult(answer="a) Red")

        result = await metric.evaluate(multiple_choice_test_case, qr, resources={})

        assert result.passed is False


# === ContainsAnswer Tests ===


class TestContainsAnswer:
    """Tests for ContainsAnswer metric."""

    @pytest.mark.asyncio
    async def test_contains_answer_pass(self, simple_test_case):
        """Should pass when answer contains reference."""
        metric = ContainsAnswer()
        qr = QueryResult(
            answer="Your favorite color is blue, based on what you told me."
        )

        result = await metric.evaluate(simple_test_case, qr, resources={})

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_contains_answer_fail(self, simple_test_case):
        """Should fail when answer doesn't contain reference."""
        metric = ContainsAnswer()
        qr = QueryResult(answer="I don't know your favorite color.")

        result = await metric.evaluate(simple_test_case, qr, resources={})

        assert result.passed is False


# === AbstentionAccuracy Tests ===


class TestAbstentionAccuracy:
    """Tests for AbstentionAccuracy metric."""

    @pytest.mark.asyncio
    async def test_abstention_correctly_detected(self, abstention_test_case):
        """Should pass when model correctly abstains."""
        metric = AbstentionAccuracy()
        qr = QueryResult(answer="I don't have information about your cat's name.")

        result = await metric.evaluate(abstention_test_case, qr, resources={})

        assert result.passed is True
        assert result.artifacts["model_abstained"] is True

    @pytest.mark.asyncio
    async def test_abstention_missed(self, abstention_test_case):
        """Should fail when model incorrectly answers unanswerable question."""
        metric = AbstentionAccuracy()
        qr = QueryResult(answer="Your cat's name is Whiskers.")

        result = await metric.evaluate(abstention_test_case, qr, resources={})

        assert result.passed is False
        assert result.artifacts["model_abstained"] is False

    @pytest.mark.asyncio
    async def test_non_abstention_question_skipped(self, simple_test_case):
        """Should skip (pass) non-abstention questions."""
        metric = AbstentionAccuracy()
        qr = QueryResult(answer="blue")

        result = await metric.evaluate(simple_test_case, qr, resources={})

        assert result.passed is True
        assert result.artifacts.get("skipped") is True


# === Composition Tests ===


class TestMetricComposition:
    """Tests for AllOf, AnyOf, ThresholdGate."""

    def test_all_of_pass(self):
        """AllOf should pass when all metrics pass."""
        gate = AllOf("must_pass_all", ["metric_a", "metric_b"])
        results = [
            MetricResult(
                metric="metric_a",
                kind="generation",
                score_type="binary",
                score=1.0,
                passed=True,
            ),
            MetricResult(
                metric="metric_b",
                kind="generation",
                score_type="binary",
                score=1.0,
                passed=True,
            ),
        ]

        combined = gate.gate(results)

        assert combined.passed is True
        assert combined.metric == "must_pass_all"

    def test_all_of_fail(self):
        """AllOf should fail when any metric fails."""
        gate = AllOf("must_pass_all", ["metric_a", "metric_b"])
        results = [
            MetricResult(
                metric="metric_a",
                kind="generation",
                score_type="binary",
                score=1.0,
                passed=True,
            ),
            MetricResult(
                metric="metric_b",
                kind="generation",
                score_type="binary",
                score=0.0,
                passed=False,
            ),
        ]

        combined = gate.gate(results)

        assert combined.passed is False

    def test_any_of_pass(self):
        """AnyOf should pass when any metric passes."""
        gate = AnyOf("must_pass_one", ["metric_a", "metric_b"])
        results = [
            MetricResult(
                metric="metric_a",
                kind="generation",
                score_type="binary",
                score=0.0,
                passed=False,
            ),
            MetricResult(
                metric="metric_b",
                kind="generation",
                score_type="binary",
                score=1.0,
                passed=True,
            ),
        ]

        combined = gate.gate(results)

        assert combined.passed is True

    def test_any_of_fail(self):
        """AnyOf should fail when all metrics fail."""
        gate = AnyOf("must_pass_one", ["metric_a", "metric_b"])
        results = [
            MetricResult(
                metric="metric_a",
                kind="generation",
                score_type="binary",
                score=0.0,
                passed=False,
            ),
            MetricResult(
                metric="metric_b",
                kind="generation",
                score_type="binary",
                score=0.0,
                passed=False,
            ),
        ]

        combined = gate.gate(results)

        assert combined.passed is False

    def test_threshold_gate(self):
        """ThresholdGate should convert continuous to binary."""
        gate = ThresholdGate("precision_pass", "context_precision", threshold=0.7)

        # Above threshold
        result_high = MetricResult(
            metric="context_precision",
            kind="retrieval",
            score_type="continuous",
            score=0.85,
            passed=True,
        )
        binary_high = gate.apply(result_high)
        assert binary_high.passed is True
        assert binary_high.score_type == "binary"

        # Below threshold
        result_low = MetricResult(
            metric="context_precision",
            kind="retrieval",
            score_type="continuous",
            score=0.65,
            passed=True,
        )
        binary_low = gate.apply(result_low)
        assert binary_low.passed is False


# === ContextPrecision Tests ===


class TestContextPrecision:
    """Tests for ContextPrecision metric."""

    @pytest.fixture
    def test_case_with_reference(self):
        """Test case with reference doc IDs."""
        return TestCase(
            id="retrieval_test",
            benchmark="test",
            user_id="user_1",
            query="What did I eat yesterday?",
            sessions=(),
            reference={"relevant_item_ids": ["doc_1", "doc_2", "doc_3"]},
        )

    @pytest.mark.asyncio
    async def test_precision_perfect(self, test_case_with_reference):
        """Should return 1.0 when all retrieved items are relevant."""
        metric = ContextPrecision()
        qr = QueryResult(
            answer="You ate pizza",
            retrieved=(
                RetrievedItem(id="doc_1", text="Pizza for dinner", rank=1),
                RetrievedItem(id="doc_2", text="Salad for lunch", rank=2),
            ),
        )

        result = await metric.evaluate(test_case_with_reference, qr, resources={})

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_precision_partial(self, test_case_with_reference):
        """Should return partial score when some items are irrelevant."""
        metric = ContextPrecision()
        qr = QueryResult(
            answer="You ate pizza",
            retrieved=(
                RetrievedItem(id="doc_1", text="Pizza for dinner", rank=1),
                RetrievedItem(id="doc_99", text="Irrelevant", rank=2),  # Not relevant
            ),
        )

        result = await metric.evaluate(test_case_with_reference, qr, resources={})

        assert result.score == 0.5  # 1 out of 2 relevant


# === LLMBinaryJudge Tests (Mock) ===


class TestLLMBinaryJudge:
    """Tests for LLMBinaryJudge metric with mocked LLM."""

    @pytest.mark.asyncio
    async def test_judge_yes_response(self, simple_test_case):
        """Should pass when judge says YES."""
        metric = LLMBinaryJudge()
        qr = QueryResult(answer="Your favorite color is blue")

        # Mock LLM that always returns YES
        async def mock_llm(prompt):
            return "YES"

        result = await metric.evaluate(
            simple_test_case, qr, resources={"llm": mock_llm}
        )

        assert result.passed is True
        assert result.artifacts["judge_response"] == "YES"

    @pytest.mark.asyncio
    async def test_judge_no_response(self, simple_test_case):
        """Should fail when judge says NO."""
        metric = LLMBinaryJudge()
        qr = QueryResult(answer="I don't know")

        async def mock_llm(prompt):
            return "NO"

        result = await metric.evaluate(
            simple_test_case, qr, resources={"llm": mock_llm}
        )

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_judge_no_llm_provided(self, simple_test_case):
        """Should fail gracefully when no LLM provided."""
        metric = LLMBinaryJudge()
        qr = QueryResult(answer="blue")

        result = await metric.evaluate(simple_test_case, qr, resources={})

        assert result.passed is False
        assert "No LLM client" in result.reason
