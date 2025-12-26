"""
PersonaMem benchmark implementation.

Converts existing PersonaMem loader to the new Benchmark interface.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..core.models import TestCase, Session


class PersonaMemBenchmark:
    """
    PersonaMem benchmark for personal memory evaluation.

    Features:
    - 589 questions across 5 question types
    - Multiple choice format (a/b/c/d)
    - Variants: 32k, 128k context sizes
    - Shared context sessions
    """

    name = "personamem"
    version = "1.0"

    DEFAULT_DATA_DIR = Path("evals/data/personamem")

    QUESTION_TYPES = [
        "recall_user_shared_facts",
        "provide_preference_aligned_recommendations",
        "suggest_new_ideas",
        "recalling_the_reasons_behind_previous_updates",
        "generalizing_to_new_scenarios",
    ]

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else self.DEFAULT_DATA_DIR

    def load(self, *, variant: Optional[str] = None) -> Sequence[TestCase]:
        """
        Load all test cases from PersonaMem.

        Args:
            variant: "32k" or "128k" (default: "32k")
        """
        variant = variant or "32k"

        questions_path = self.data_dir / f"questions_{variant}_{variant}.json"
        contexts_path = self.data_dir / f"shared_contexts_{variant}.jsonl"

        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_path}")

        # Load questions
        with open(questions_path) as f:
            questions = json.load(f)

        # Load shared contexts
        shared_contexts = {}
        if contexts_path.exists():
            with open(contexts_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        shared_contexts.update(data)

        # Convert to TestCase objects
        test_cases = []
        for i, q in enumerate(questions):
            question_id = f"personamem_{variant}_{i}"

            # Get sessions from shared context
            context_id = q.get("shared_context_id", "")
            sessions = []
            if context_id and context_id in shared_contexts:
                turns = shared_contexts[context_id]
                # Convert turns to session content
                content = self._turns_to_content(turns)
                sessions = [Session(content=content, date="unknown")]

            # Build test case
            test_case = TestCase(
                id=question_id,
                benchmark=self.name,
                user_id=f"user_{context_id or i}",
                query=q.get("question", ""),
                sessions=tuple(sessions),
                reference_answer=q.get("correct_answer", ""),
                question_type=q.get("category", "unknown"),
                options=q.get("options", {}),
                correct_option=q.get("correct_answer", "").lower()
                if q.get("correct_answer") in "abcdABCD"
                else None,
                metadata={
                    "shared_context_id": context_id,
                    "variant": variant,
                    "original_index": i,
                },
            )
            test_cases.append(test_case)

        return test_cases

    def _turns_to_content(self, turns: list) -> str:
        """Convert conversation turns to session content."""
        parts = []
        for turn in turns:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def default_metrics(self) -> Sequence[str]:
        """Default metrics for PersonaMem."""
        return ["option_extractor", "context_relevance"]

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
            variant: Benchmark variant
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
