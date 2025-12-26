"""Automatic failure classification based on heuristics and patterns."""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import re


@dataclass
class FailureClassification:
    category: str
    confidence: float
    reasoning: str


class FailureClassifier:
    CATEGORIES = {
        "cross_session_gap": "Information across sessions not linked",
        "temporal_blindness": "Date ordering not captured",
        "entity_resolution": "Same entity not recognized across mentions",
        "insufficient_context": "Retrieved context too short or sparse",
        "semantic_mismatch": "Query doesn't match node embeddings",
        "ingestion_failure": "Data not indexed properly",
        "hallucination": "Answer contains made-up information",
        "wrong_aggregation": "Counting/summarization error",
        "abstention_failure": "Failed to say 'I don't know' when appropriate",
        "format_error": "Answer format doesn't match expected",
        "api_error": "API error (rate limit, timeout, etc.)",
    }

    def classify(
        self,
        question_text: str,
        question_type: str,
        gold_answer: str,
        generated_answer: str,
        retrieved_count: int,
        session_count: int,
        nodes_created: int,
    ) -> FailureClassification:
        classifications: List[Tuple[str, float, str]] = []

        q_lower = question_text.lower()
        gen_lower = (generated_answer or "").lower()
        gold_lower = gold_answer.lower()

        # Check for API errors first (rate limits, timeouts)
        api_error_patterns = [
            "error:",
            "rate limit",
            "timeout",
            "429",
            "500",
            "502",
            "503",
        ]
        if any(pat in gen_lower for pat in api_error_patterns):
            return FailureClassification(
                category="api_error",
                confidence=1.0,
                reasoning="API error detected in response",
            )

        if "multi-session" in question_type or session_count > 3:
            classifications.append(
                (
                    "cross_session_gap",
                    0.7,
                    f"Multi-session question with {session_count} sessions",
                )
            )

        temporal_keywords = [
            "when",
            "before",
            "after",
            "first",
            "last",
            "date",
            "time",
            "year",
            "month",
            "how long",
        ]
        if (
            any(kw in q_lower for kw in temporal_keywords)
            or "temporal" in question_type
        ):
            classifications.append(
                ("temporal_blindness", 0.6, "Question requires temporal reasoning")
            )

        if retrieved_count < 3:
            classifications.append(
                ("insufficient_context", 0.8, f"Only {retrieved_count} nodes retrieved")
            )

        count_keywords = ["how many", "count", "number of", "total"]
        if any(kw in q_lower for kw in count_keywords):
            try:
                gold_match = re.search(r"\d+", gold_answer)
                gen_match = re.search(r"\d+", generated_answer)
                gold_num = int(gold_match.group()) if gold_match else None
                gen_num = int(gen_match.group()) if gen_match else None
                if gold_num is not None and gen_num is not None and gold_num != gen_num:
                    classifications.append(
                        (
                            "wrong_aggregation",
                            0.9,
                            f"Expected {gold_num}, got {gen_num}",
                        )
                    )
            except (ValueError, AttributeError):
                pass

        if nodes_created == 0 or (session_count > 0 and nodes_created < session_count):
            classifications.append(
                (
                    "ingestion_failure",
                    0.85,
                    f"Only {nodes_created} nodes from {session_count} sessions",
                )
            )

        if gold_lower in ["a", "b", "c", "d"] and gen_lower not in ["a", "b", "c", "d"]:
            if not any(
                opt in gen_lower
                for opt in ["a)", "b)", "c)", "d)", "(a)", "(b)", "(c)", "(d)"]
            ):
                classifications.append(
                    (
                        "format_error",
                        0.95,
                        "Multiple choice answer not in expected format",
                    )
                )

        hallucination_phrases = [
            "as mentioned",
            "you told me",
            "based on our conversation",
            "according to",
            "you said",
            "as we discussed",
        ]
        if any(phrase in gen_lower for phrase in hallucination_phrases):
            if retrieved_count < 5:
                classifications.append(
                    ("hallucination", 0.6, "Claims memory with limited retrieval")
                )

        if not classifications:
            classifications.append(
                (
                    "semantic_mismatch",
                    0.5,
                    "Default: query may not match stored embeddings",
                )
            )

        classifications.sort(key=lambda x: x[1], reverse=True)
        best = classifications[0]

        return FailureClassification(
            category=best[0], confidence=best[1], reasoning=best[2]
        )

    def classify_batch(
        self, failures: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], FailureClassification]]:
        results = []
        for f in failures:
            classification = self.classify(
                question_text=f.get("question_text", ""),
                question_type=f.get("question_type", ""),
                gold_answer=f.get("gold_answer", ""),
                generated_answer=f.get("generated_answer", ""),
                retrieved_count=f.get("retrieved_node_count", 0),
                session_count=f.get("session_count", 0),
                nodes_created=f.get("nodes_created", 0),
            )
            results.append((f, classification))
        return results


if __name__ == "__main__":
    classifier = FailureClassifier()

    result = classifier.classify(
        question_text="How many projects have I led across all our conversations?",
        question_type="multi-session",
        gold_answer="3",
        generated_answer="2",
        retrieved_count=5,
        session_count=7,
        nodes_created=45,
    )

    print(f"Category: {result.category}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
