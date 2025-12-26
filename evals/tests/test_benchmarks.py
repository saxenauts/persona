"""
Tests for benchmarks: PersonaMem, LongMemEval, registry.
"""

import pytest
from pathlib import Path

from evals.benchmarks import (
    get_registry,
    BenchmarkRegistry,
    PersonaMemBenchmark,
    LongMemEvalBenchmark,
)
from evals.core.models import TestCase, Session


# === Registry Tests ===


class TestBenchmarkRegistry:
    """Tests for BenchmarkRegistry."""

    def test_get_registry_singleton(self):
        """get_registry should return singleton instance."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_builtin_benchmarks_registered(self):
        """Built-in benchmarks should be auto-registered."""
        registry = get_registry()
        benchmarks = registry.list_benchmarks()

        assert "personamem" in benchmarks
        assert "longmemeval" in benchmarks

    def test_get_benchmark(self):
        """Should get benchmark instance by name."""
        registry = get_registry()

        pm = registry.get("personamem")
        assert pm is not None
        assert pm.name == "personamem"

        lm = registry.get("longmemeval")
        assert lm is not None
        assert lm.name == "longmemeval"

    def test_get_unknown_benchmark(self):
        """Should return None for unknown benchmark."""
        registry = get_registry()
        assert registry.get("nonexistent") is None

    def test_has_benchmark(self):
        """has() should check if benchmark exists."""
        registry = get_registry()
        assert registry.has("personamem") is True
        assert registry.has("nonexistent") is False


# === PersonaMemBenchmark Tests ===


class TestPersonaMemBenchmark:
    """Tests for PersonaMemBenchmark."""

    def test_benchmark_properties(self):
        """Benchmark should have correct properties."""
        pm = PersonaMemBenchmark()

        assert pm.name == "personamem"
        assert pm.version == "1.0"
        assert len(pm.QUESTION_TYPES) == 5

    def test_default_metrics(self):
        """Should return appropriate default metrics."""
        pm = PersonaMemBenchmark()
        metrics = pm.default_metrics()

        assert len(metrics) > 0
        assert "option_extractor" in metrics

    @pytest.mark.skipif(
        not Path("evals/data/personamem").exists(),
        reason="PersonaMem data not available",
    )
    def test_load_cases(self):
        """Should load test cases from data files."""
        pm = PersonaMemBenchmark()
        cases = pm.load(variant="32k")

        assert len(cases) > 0
        assert all(isinstance(c, TestCase) for c in cases)

    @pytest.mark.skipif(
        not Path("evals/data/personamem").exists(),
        reason="PersonaMem data not available",
    )
    def test_sample_stratified(self):
        """Should sample stratified by question type."""
        pm = PersonaMemBenchmark()

        sizes = {
            "recall_user_shared_facts": 2,
            "provide_preference_aligned_recommendations": 2,
        }
        sampled = pm.sample(sizes, seed=42, variant="32k")

        # Should have approximately the requested distribution
        assert len(sampled) <= 4


# === LongMemEvalBenchmark Tests ===


class TestLongMemEvalBenchmark:
    """Tests for LongMemEvalBenchmark."""

    def test_benchmark_properties(self):
        """Benchmark should have correct properties."""
        lm = LongMemEvalBenchmark()

        assert lm.name == "longmemeval"
        assert lm.version == "1.0"
        assert len(lm.QUESTION_TYPES) == 6

    def test_question_types(self):
        """Should have all LongMemEval question types."""
        lm = LongMemEvalBenchmark()
        expected_types = {
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
            "multi-session",
            "temporal-reasoning",
            "knowledge-update",
        }
        assert set(lm.QUESTION_TYPES) == expected_types

    def test_default_metrics(self):
        """Should return appropriate default metrics."""
        lm = LongMemEvalBenchmark()
        metrics = lm.default_metrics()

        assert "llm_binary_judge" in metrics
        assert "abstention_accuracy" in metrics

    @pytest.mark.skipif(
        not Path("evals/data/longmemeval_oracle.json").exists(),
        reason="LongMemEval data not available",
    )
    def test_load_cases(self):
        """Should load test cases from data file."""
        lm = LongMemEvalBenchmark()
        cases = lm.load()

        assert len(cases) > 0
        assert all(isinstance(c, TestCase) for c in cases)

    @pytest.mark.skipif(
        not Path("evals/data/longmemeval_oracle.json").exists(),
        reason="LongMemEval data not available",
    )
    def test_load_abstention_variant(self):
        """Should filter to abstention cases only."""
        lm = LongMemEvalBenchmark()
        all_cases = lm.load()
        abstention_cases = lm.load(variant="abstention")

        # Abstention should be a subset
        assert len(abstention_cases) <= len(all_cases)

        # All abstention cases should be marked
        for tc in abstention_cases:
            assert tc.metadata.get("is_abstention") is True or "abstention" in tc.tags

    @pytest.mark.skipif(
        not Path("evals/data/longmemeval_oracle.json").exists(),
        reason="LongMemEval data not available",
    )
    def test_test_case_structure(self):
        """Test cases should have proper structure."""
        lm = LongMemEvalBenchmark()
        cases = lm.load()

        if cases:
            tc = cases[0]

            # Should have required fields
            assert tc.id
            assert tc.benchmark == "longmemeval"
            assert tc.user_id
            assert tc.query

            # Should have sessions with dates (temporal context)
            if tc.sessions:
                assert all(isinstance(s, Session) for s in tc.sessions)

    @pytest.mark.skipif(
        not Path("evals/data/longmemeval_oracle.json").exists(),
        reason="LongMemEval data not available",
    )
    def test_sample_stratified(self):
        """Should sample stratified by question type."""
        lm = LongMemEvalBenchmark()

        sizes = {
            "single-session-user": 2,
            "multi-session": 2,
        }
        sampled = lm.sample(sizes, seed=42)

        assert len(sampled) <= 4

    @pytest.mark.skipif(
        not Path("evals/data/longmemeval_oracle.json").exists(),
        reason="LongMemEval data not available",
    )
    def test_get_type_distribution(self):
        """Should return question type distribution."""
        lm = LongMemEvalBenchmark()
        dist = lm.get_type_distribution()

        assert isinstance(dist, dict)
        assert len(dist) > 0
        assert all(isinstance(v, int) for v in dist.values())
