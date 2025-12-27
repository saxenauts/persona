"""
Analyze PersonaMem and LongMemEval question datasets.
Understand task types, reasoning requirements, and memory access patterns.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class QuestionTypeStats:
    name: str
    count: int
    avg_context_tokens: float
    example_question: str
    example_answer: str


class PersonaMemAnalyzer:
    def __init__(self, questions_path: str):
        self.questions_path = Path(questions_path)
        self.questions: List[dict] = []
        self._load()

    def _load(self):
        with open(self.questions_path) as f:
            self.questions = json.load(f)

    def get_type_distribution(self) -> Dict[str, QuestionTypeStats]:
        by_type: Dict[str, List[dict]] = defaultdict(list)

        for q in self.questions:
            qtype = q.get("question_type", "unknown")
            by_type[qtype].append(q)

        stats = {}
        for qtype, qs in by_type.items():
            ctx_tokens = [q.get("context_length_in_tokens", 0) for q in qs]
            example = qs[0]

            stats[qtype] = QuestionTypeStats(
                name=qtype,
                count=len(qs),
                avg_context_tokens=sum(ctx_tokens) / len(ctx_tokens)
                if ctx_tokens
                else 0,
                example_question=example.get("user_question_or_message", "")[:200],
                example_answer=example.get("correct_answer", ""),
            )

        return stats

    def get_topic_distribution(self) -> Dict[str, int]:
        topics = defaultdict(int)
        for q in self.questions:
            topic = q.get("topic", "unknown")
            topics[topic] += 1
        return dict(topics)

    def get_context_length_stats(self) -> dict:
        tokens = [q.get("context_length_in_tokens", 0) for q in self.questions]

        if not tokens:
            return {}

        import statistics

        return {
            "min": min(tokens),
            "max": max(tokens),
            "mean": round(statistics.mean(tokens)),
            "median": round(statistics.median(tokens)),
            "total_questions": len(tokens),
        }

    def get_distance_to_ref_stats(self) -> dict:
        distances = []
        for q in self.questions:
            dist = q.get("distance_to_ref_in_tokens", 0)
            if dist:
                distances.append(dist)

        if not distances:
            return {}

        import statistics

        return {
            "min": min(distances),
            "max": max(distances),
            "mean": round(statistics.mean(distances)),
            "median": round(statistics.median(distances)),
            "questions_with_distance": len(distances),
        }

    def analyze_question_type(self, qtype: str) -> dict:
        qs = [q for q in self.questions if q.get("question_type") == qtype]

        if not qs:
            return {"error": f"No questions of type {qtype}"}

        examples = []
        for q in qs[:3]:
            examples.append(
                {
                    "question": q.get("user_question_or_message", ""),
                    "answer": q.get("correct_answer", ""),
                    "context_tokens": q.get("context_length_in_tokens", 0),
                    "distance_to_ref_tokens": q.get("distance_to_ref_in_tokens", 0),
                }
            )

        return {
            "type": qtype,
            "count": len(qs),
            "examples": examples,
            "reasoning_required": self._infer_reasoning_type(qtype),
        }

    def _infer_reasoning_type(self, qtype: str) -> str:
        reasoning_map = {
            "track_full_preference_evolution": "temporal_tracking",
            "recall_user_shared_facts": "exact_recall",
            "recalling_the_reasons_behind_previous_updates": "causal_reasoning",
            "suggest_new_ideas": "synthesis_generation",
            "generalizing_to_new_scenarios": "inference_generalization",
            "provide_preference_aligned_recommendations": "preference_matching",
            "recalling_facts_mentioned_by_the_user": "verbatim_recall",
        }
        return reasoning_map.get(qtype, "unknown")

    def get_memory_requirements(self) -> dict:
        requirements = {
            "exact_recall": [],
            "temporal_tracking": [],
            "synthesis": [],
            "inference": [],
        }

        for qtype, stats in self.get_type_distribution().items():
            reasoning = self._infer_reasoning_type(qtype)

            if reasoning in ["exact_recall", "verbatim_recall"]:
                requirements["exact_recall"].append(qtype)
            elif reasoning == "temporal_tracking":
                requirements["temporal_tracking"].append(qtype)
            elif reasoning in ["synthesis_generation", "preference_matching"]:
                requirements["synthesis"].append(qtype)
            elif reasoning in ["inference_generalization", "causal_reasoning"]:
                requirements["inference"].append(qtype)

        return requirements

    def summary(self) -> dict:
        type_stats = self.get_type_distribution()

        return {
            "total_questions": len(self.questions),
            "question_types": {
                name: {"count": s.count, "avg_tokens": round(s.avg_context_tokens)}
                for name, s in type_stats.items()
            },
            "context_stats": self.get_context_length_stats(),
            "memory_requirements": self.get_memory_requirements(),
            "topics": self.get_topic_distribution(),
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python question_analyzer.py <questions.json>")
        sys.exit(1)

    analyzer = PersonaMemAnalyzer(sys.argv[1])
    print(json.dumps(analyzer.summary(), indent=2))
