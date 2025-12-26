"""
Base metric classes and composition helpers.

Design Principles (from Hamel/Eugene research):
- Metrics are single-purpose and composable
- Binary pass/fail preferred over numeric scales
- Continuous scores (0-1) convert to binary via threshold
- Always include reasoning for debuggability
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..core.models import TestCase, QueryResult, MetricResult, MetricKind, ScoreType
from ..core.interfaces import AdapterCapabilities


class BaseMetric(ABC):
    """
    Abstract base class for metrics.

    Subclasses must implement:
    - name, kind, score_type properties
    - evaluate() async method

    Example:
        class MyMetric(BaseMetric):
            name = "my_metric"
            kind = "end_to_end"
            score_type = "binary"

            async def evaluate(self, test_case, query_result, *, resources):
                passed = check_something(test_case, query_result)
                return MetricResult(
                    metric=self.name,
                    kind=self.kind,
                    score_type=self.score_type,
                    score=1.0 if passed else 0.0,
                    passed=passed,
                    reason="explanation",
                )
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric identifier."""
        ...

    @property
    @abstractmethod
    def kind(self) -> MetricKind:
        """Category: retrieval, generation, end_to_end, etc."""
        ...

    @property
    @abstractmethod
    def score_type(self) -> ScoreType:
        """binary or continuous."""
        ...

    def required_capabilities(self) -> AdapterCapabilities:
        """
        Override to specify adapter requirements.

        Default: no special requirements.
        """
        return AdapterCapabilities()

    @abstractmethod
    async def evaluate(
        self,
        test_case: TestCase,
        query_result: QueryResult,
        *,
        resources: Mapping[str, Any],
    ) -> MetricResult:
        """
        Evaluate the test case and return a result.

        Args:
            test_case: Input with query, reference answer, labels
            query_result: Adapter output with answer, retrieved items
            resources: Shared resources (LLM clients, etc.)
        """
        ...


# === Composition Helpers ===


@dataclass
class AllOf:
    """
    Composite metric that passes only if ALL sub-metrics pass.

    Usage:
        combined = AllOf("must_pass_all", ["exact_match", "context_precision"])
        result = combined.gate(metric_results)
    """

    name: str
    metric_names: Sequence[str]

    def gate(self, results: Sequence[MetricResult]) -> MetricResult:
        """Check if all named metrics passed."""
        by_name = {r.metric: r for r in results}
        missing = [m for m in self.metric_names if m not in by_name]

        if missing:
            return MetricResult(
                metric=self.name,
                kind="end_to_end",
                score_type="binary",
                score=0.0,
                passed=False,
                reason=f"Missing required metrics: {missing}",
            )

        passed = all(by_name[m].passed for m in self.metric_names)
        failed_names = [m for m in self.metric_names if not by_name[m].passed]

        return MetricResult(
            metric=self.name,
            kind="end_to_end",
            score_type="binary",
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="" if passed else f"Failed metrics: {failed_names}",
        )


@dataclass
class AnyOf:
    """
    Composite metric that passes if ANY sub-metric passes.

    Usage:
        combined = AnyOf("must_pass_one", ["option_a", "option_b"])
        result = combined.gate(metric_results)
    """

    name: str
    metric_names: Sequence[str]

    def gate(self, results: Sequence[MetricResult]) -> MetricResult:
        """Check if any named metric passed."""
        by_name = {r.metric: r for r in results}

        found = [m for m in self.metric_names if m in by_name]
        if not found:
            return MetricResult(
                metric=self.name,
                kind="end_to_end",
                score_type="binary",
                score=0.0,
                passed=False,
                reason=f"No metrics found from: {self.metric_names}",
            )

        passed = any(by_name[m].passed for m in found)
        passing_names = [m for m in found if by_name[m].passed]

        return MetricResult(
            metric=self.name,
            kind="end_to_end",
            score_type="binary",
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=f"Passed: {passing_names}" if passed else "All metrics failed",
        )


@dataclass
class ThresholdGate:
    """
    Convert a continuous metric result to binary pass/fail.

    Usage:
        gate = ThresholdGate("precision_pass", "context_precision", threshold=0.7)
        binary_result = gate.apply(continuous_result)
    """

    name: str
    source_metric: str
    threshold: float

    def apply(self, result: MetricResult) -> MetricResult:
        """Apply threshold to convert continuous to binary."""
        if result.metric != self.source_metric:
            raise ValueError(
                f"Expected metric {self.source_metric}, got {result.metric}"
            )

        passed = result.score >= self.threshold

        return MetricResult(
            metric=self.name,
            kind=result.kind,
            score_type="binary",
            score=1.0 if passed else 0.0,
            passed=passed,
            threshold=self.threshold,
            reason=f"Score {result.score:.3f} {'≥' if passed else '<'} threshold {self.threshold}",
            artifacts={"original_result": result},
        )
