"""
LongMemEval benchmark implementation.

Converts existing LongMemEval loader to the new Benchmark interface.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..core.models import TestCase, Session


class LongMemEvalBenchmark:
    """
    LongMemEval Oracle benchmark for long-term memory evaluation.

    Features:
    - 6 question types: single-session-user, single-session-assistant,
      single-session-preference, multi-session, temporal-reasoning, knowledge-update
    - Abstention questions (expected answer: decline/unknown)
    - Haystack sessions with temporal context
    - Free-form answers (evaluated via LLM-as-judge)
    """

    name = "longmemeval"
    version = "1.0"

    DEFAULT_DATA_PATH = Path("evals/data/longmemeval_oracle.json")

    QUESTION_TYPES = [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
    ]

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else self.DEFAULT_DATA_PATH

    def load(self, *, variant: Optional[str] = None) -> Sequence[TestCase]:
        """
        Load all test cases from LongMemEval Oracle.

        Args:
            variant: Optional filter - "abstention" for abstention-only,
                     or a question type name

        Returns:
            Sequence of TestCase objects
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"LongMemEval Oracle data not found at {self.data_path}"
            )

        with open(self.data_path, "r") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict):
            if "data" in data:
                questions_data = data["data"]
            else:
                questions_data = list(data.values())[0] if data else []
        else:
            questions_data = data

        test_cases = []
        for item in questions_data:
            test_case = self._parse_question(item)

            # Apply variant filter
            if variant:
                if variant == "abstention" and not test_case.metadata.get(
                    "is_abstention"
                ):
                    continue
                elif (
                    variant in self.QUESTION_TYPES
                    and test_case.question_type != variant
                ):
                    continue

            test_cases.append(test_case)

        return test_cases

    def _parse_question(self, item: dict[str, Any]) -> TestCase:
        """Parse a raw question item into a TestCase object."""
        question_id = item.get("question_id", "")
        is_abstention = "_abs" in question_id

        # Convert haystack sessions to Session objects
        sessions = self._parse_haystack_sessions(
            item.get("haystack_sessions", []),
            item.get("haystack_dates", []),
            item.get("haystack_session_ids", []),
        )

        return TestCase(
            id=question_id,
            benchmark=self.name,
            user_id=f"user_{question_id.split('_')[0]}",
            query=item.get("question", ""),
            sessions=tuple(sessions),
            reference_answer=item.get("answer", ""),
            question_type=item.get("question_type", "unknown"),
            tags=("abstention",) if is_abstention else (),
            metadata={
                "question_date": item.get("question_date", ""),
                "is_abstention": is_abstention,
                "haystack_dates": item.get("haystack_dates", []),
                "haystack_session_ids": item.get("haystack_session_ids", []),
            },
            # LongMemEval is free-form, not multiple choice
            options=None,
            correct_option=None,
        )

    def _parse_haystack_sessions(
        self,
        haystack_sessions: list[list[dict[str, str]]],
        haystack_dates: list[str],
        haystack_session_ids: list[str],
    ) -> list[Session]:
        """Convert haystack sessions to Session objects."""
        sessions = []

        for i, session_turns in enumerate(haystack_sessions):
            # Build content from turns
            parts = []
            for turn in session_turns:
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "")
                parts.append(f"{role}: {content}")

            content = "\n".join(parts)

            # Get date and session_id if available
            date = haystack_dates[i] if i < len(haystack_dates) else None
            session_id = (
                haystack_session_ids[i]
                if i < len(haystack_session_ids)
                else f"session_{i}"
            )

            sessions.append(
                Session(
                    content=content,
                    date=date,
                    metadata={"session_id": session_id},
                )
            )

        return sessions

    def default_metrics(self) -> Sequence[str]:
        """
        Default metrics for LongMemEval.

        LongMemEval requires LLM-as-judge because answers are free-form.
        """
        return ["llm_binary_judge", "abstention_accuracy"]

    def sample(
        self,
        sizes: Mapping[str, int],
        *,
        seed: Optional[int] = None,
        variant: Optional[str] = None,
    ) -> Sequence[TestCase]:
        """
        Stratified sampling by question type.

        Args:
            sizes: {question_type: count} mapping
            seed: Random seed for reproducibility
            variant: Benchmark variant filter
        """
        all_cases = self.load(variant=variant)

        # Group by question type
        by_type: dict[str, list[TestCase]] = {}
        for tc in all_cases:
            qtype = tc.question_type or "unknown"
            by_type.setdefault(qtype, []).append(tc)

        # Sample from each type
        rng = random.Random(seed)
        sampled = []

        for qtype, count in sizes.items():
            available = by_type.get(qtype, [])
            if not available:
                continue
            n = min(count, len(available))
            sampled.extend(rng.sample(available, n))

        return sampled

    def get_abstention_cases(self) -> Sequence[TestCase]:
        """Get only abstention test cases."""
        return self.load(variant="abstention")

    def get_type_distribution(self) -> dict[str, int]:
        """Get distribution of questions by type."""
        cases = self.load()
        distribution: dict[str, int] = {}

        for tc in cases:
            qtype = tc.question_type or "unknown"
            distribution[qtype] = distribution.get(qtype, 0) + 1

        return distribution
