"""
Retrieval quality metrics.

Metrics for evaluating the retrieval component of memory systems:
- ContextPrecision: What fraction of retrieved items are relevant?
- ContextRecall: What fraction of relevant items were retrieved?

Design: These require labeled test cases with relevant_item_ids
or the adapter to support retrieval item output.
"""

from __future__ import annotations

from typing import Any, Mapping, Set

from ..core.models import TestCase, QueryResult, MetricResult
from ..core.interfaces import AdapterCapabilities
from .base import BaseMetric


class ContextPrecision(BaseMetric):
    """
    Retrieval precision: what fraction of retrieved items are relevant?

    Precision = |retrieved ∩ relevant| / |retrieved|

    Requires:
    - test_case.reference["relevant_item_ids"] - list of relevant item IDs
    - adapter that returns retrieved items with IDs

    Design: Returns continuous score (0-1) that can be thresholded.
    """

    name = "context_precision"
    kind = "retrieval"
    score_type = "continuous"

    def __init__(self, *, top_k: int = 10, threshold: float = 0.5):
        """
        Args:
            top_k: Only consider top k retrieved items
            threshold: Score threshold for pass/fail
        """
        self.top_k = top_k
        self.threshold = threshold

    def required_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(supports_retrieval_items=True)

    async def evaluate(
        self,
        test_case: TestCase,
        query_result: QueryResult,
        *,
        resources: Mapping[str, Any],
    ) -> MetricResult:
        # Get labeled relevant items
        relevant_ids: Set[str] = set(test_case.reference.get("relevant_item_ids", []))

        if not relevant_ids:
            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reason="No relevant_item_ids labels in test case",
            )

        # Get retrieved items
        retrieved = query_result.retrieved[: self.top_k]
        if not retrieved:
            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reason="No items retrieved",
            )

        retrieved_ids = {r.id for r in retrieved}

        # Calculate precision
        hits = len(retrieved_ids & relevant_ids)
        precision = hits / len(retrieved_ids)
        passed = precision >= self.threshold

        return MetricResult(
            metric=self.name,
            kind=self.kind,
            score_type=self.score_type,
            score=precision,
            passed=passed,
            threshold=self.threshold,
            reason=f"Precision: {hits}/{len(retrieved_ids)} = {precision:.3f}",
            artifacts={
                "retrieved_ids": list(retrieved_ids),
                "relevant_ids": list(relevant_ids),
                "hits": hits,
                "top_k": self.top_k,
            },
        )


class ContextRecall(BaseMetric):
    """
    Retrieval recall: what fraction of relevant items were retrieved?

    Recall = |retrieved ∩ relevant| / |relevant|

    Requires:
    - test_case.reference["relevant_item_ids"] - list of relevant item IDs
    - adapter that returns retrieved items with IDs

    Design: Returns continuous score (0-1) that can be thresholded.
    """

    name = "context_recall"
    kind = "retrieval"
    score_type = "continuous"

    def __init__(self, *, top_k: int = 20, threshold: float = 0.5):
        """
        Args:
            top_k: Only consider top k retrieved items
            threshold: Score threshold for pass/fail
        """
        self.top_k = top_k
        self.threshold = threshold

    def required_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(supports_retrieval_items=True)

    async def evaluate(
        self,
        test_case: TestCase,
        query_result: QueryResult,
        *,
        resources: Mapping[str, Any],
    ) -> MetricResult:
        # Get labeled relevant items
        relevant_ids: Set[str] = set(test_case.reference.get("relevant_item_ids", []))

        if not relevant_ids:
            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reason="No relevant_item_ids labels in test case",
            )

        # Get retrieved items
        retrieved = query_result.retrieved[: self.top_k]
        retrieved_ids = {r.id for r in retrieved} if retrieved else set()

        # Calculate recall
        hits = len(retrieved_ids & relevant_ids)
        recall = hits / len(relevant_ids)
        passed = recall >= self.threshold

        return MetricResult(
            metric=self.name,
            kind=self.kind,
            score_type=self.score_type,
            score=recall,
            passed=passed,
            threshold=self.threshold,
            reason=f"Recall: {hits}/{len(relevant_ids)} = {recall:.3f}",
            artifacts={
                "retrieved_ids": list(retrieved_ids),
                "relevant_ids": list(relevant_ids),
                "hits": hits,
                "top_k": self.top_k,
            },
        )


class ContextRelevance(BaseMetric):
    """
    Checks if retrieved context contains information needed to answer.

    Unlike precision/recall, this doesn't require labeled relevant_item_ids.
    Instead, it checks if the gold answer or key terms appear in the context.

    Design: Simple heuristic metric, useful when labeled data isn't available.
    """

    name = "context_relevance"
    kind = "retrieval"
    score_type = "binary"

    def required_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(supports_context_text=True)

    async def evaluate(
        self,
        test_case: TestCase,
        query_result: QueryResult,
        *,
        resources: Mapping[str, Any],
    ) -> MetricResult:
        # Get context text
        context = query_result.context_text or ""
        if not context and query_result.retrieved:
            # Build context from retrieved items
            context = " ".join(r.text for r in query_result.retrieved if r.text)

        if not context:
            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=0.0,
                passed=False,
                reason="No context available",
            )

        context_lower = context.lower()

        # Check if gold answer appears in context
        gold = test_case.reference_answer
        if gold and gold.strip():
            if gold.lower() in context_lower:
                return MetricResult(
                    metric=self.name,
                    kind=self.kind,
                    score_type=self.score_type,
                    score=1.0,
                    passed=True,
                    reason="Gold answer found in context",
                )

        # For multiple choice, check if correct option text is in context
        if test_case.correct_option and test_case.options:
            correct_text = test_case.options.get(test_case.correct_option, "")
            if correct_text and correct_text.lower() in context_lower:
                return MetricResult(
                    metric=self.name,
                    kind=self.kind,
                    score_type=self.score_type,
                    score=1.0,
                    passed=True,
                    reason="Correct option text found in context",
                )

        return MetricResult(
            metric=self.name,
            kind=self.kind,
            score_type=self.score_type,
            score=0.0,
            passed=False,
            reason="Relevant information not found in context",
            artifacts={"context_preview": context[:500]},
        )
