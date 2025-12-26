"""
Exact match and option extraction metrics.

For PersonaMem-style multiple choice questions and LongMemEval-style
free-form answer matching.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from ..core.models import TestCase, QueryResult, MetricResult
from ..core.interfaces import AdapterCapabilities
from .base import BaseMetric


class BinaryExactMatch(BaseMetric):
    """
    Binary metric: does the answer exactly match the reference?

    Design: Simple, fast, deterministic. Good baseline metric.
    Case-insensitive by default.
    """

    name = "binary_exact_match"
    kind = "end_to_end"
    score_type = "binary"

    def __init__(self, *, case_sensitive: bool = False, strip: bool = True):
        self.case_sensitive = case_sensitive
        self.strip = strip

    def required_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities()

    async def evaluate(
        self,
        test_case: TestCase,
        query_result: QueryResult,
        *,
        resources: Mapping[str, Any],
    ) -> MetricResult:
        gold = test_case.reference_answer or ""
        pred = query_result.answer or ""

        if self.strip:
            gold = gold.strip()
            pred = pred.strip()

        if not self.case_sensitive:
            gold = gold.lower()
            pred = pred.lower()

        passed = bool(gold) and gold == pred

        return MetricResult(
            metric=self.name,
            kind=self.kind,
            score_type=self.score_type,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="Exact match" if passed else f"Expected: {gold!r}, Got: {pred!r}",
        )


class ContainsAnswer(BaseMetric):
    """
    Binary metric: does the answer contain the reference answer?

    Design: More lenient than exact match. Good for free-form answers
    where the model may add extra context.
    """

    name = "contains_answer"
    kind = "end_to_end"
    score_type = "binary"

    def __init__(self, *, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def required_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities()

    async def evaluate(
        self,
        test_case: TestCase,
        query_result: QueryResult,
        *,
        resources: Mapping[str, Any],
    ) -> MetricResult:
        gold = test_case.reference_answer or ""
        pred = query_result.answer or ""

        if not self.case_sensitive:
            gold = gold.lower()
            pred = pred.lower()

        gold = gold.strip()
        pred = pred.strip()

        passed = bool(gold) and gold in pred

        return MetricResult(
            metric=self.name,
            kind=self.kind,
            score_type=self.score_type,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="Answer contains reference"
            if passed
            else f"Reference '{gold}' not found in answer",
        )


class OptionExtractor(BaseMetric):
    """
    Metric for multiple choice questions (PersonaMem style).

    Extracts the chosen option (a/b/c/d) from the answer and compares
    to the correct option.

    Design: Handles various answer formats:
    - "a"
    - "(a)"
    - "The answer is a"
    - "Option A: ..."
    """

    name = "option_extractor"
    kind = "end_to_end"
    score_type = "binary"

    # Patterns for extracting option letters
    OPTION_PATTERNS = [
        r"^([a-d])$",  # Just the letter
        r"^\(([a-d])\)$",  # Parenthesized
        r"^([a-d])\.",  # Letter with period
        r"^([a-d])\)",  # Letter with closing paren
        r"^option\s*([a-d])",  # "Option A"
        r"^answer[:\s]+([a-d])",  # "Answer: a"
        r"^the answer is[:\s]+([a-d])",  # "The answer is a"
        r"\(([a-d])\)",  # Parenthesized anywhere
    ]

    def required_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities()

    def _extract_option(self, text: str) -> Optional[str]:
        """Extract option letter from text."""
        text = text.strip().lower()

        # Try each pattern
        for pattern in self.OPTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        # Try first word if it's a single letter
        tokens = text.split()
        if tokens:
            first = tokens[0].strip("().,:;")
            if first in "abcd":
                return first

        return None

    async def evaluate(
        self,
        test_case: TestCase,
        query_result: QueryResult,
        *,
        resources: Mapping[str, Any],
    ) -> MetricResult:
        # Get correct option
        correct = test_case.correct_option
        if not correct:
            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=0.0,
                passed=False,
                reason="No correct_option specified in test case",
            )

        correct = correct.lower().strip()

        # Extract chosen option from answer
        answer = query_result.answer or ""
        extracted = self._extract_option(answer)

        if not extracted:
            # Try matching option text instead
            if test_case.options:
                correct_text = test_case.options.get(correct, "")
                if correct_text and correct_text.lower() in answer.lower():
                    return MetricResult(
                        metric=self.name,
                        kind=self.kind,
                        score_type=self.score_type,
                        score=1.0,
                        passed=True,
                        reason=f"Answer contains correct option text",
                        artifacts={
                            "correct_option": correct,
                            "matched_text": correct_text,
                        },
                    )

            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=0.0,
                passed=False,
                reason=f"Could not extract option from: {answer[:100]}",
            )

        passed = extracted == correct

        return MetricResult(
            metric=self.name,
            kind=self.kind,
            score_type=self.score_type,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=f"Extracted: {extracted}, Expected: {correct}",
            artifacts={"extracted_option": extracted, "correct_option": correct},
        )
