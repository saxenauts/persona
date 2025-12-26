"""RAGAS-style retrieval metrics for memory systems."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class RetrievalScore:
    precision: float
    recall: float
    f1: float
    retrieved_count: int
    relevant_count: int
    relevant_retrieved: int


class RetrievalMetrics:
    @staticmethod
    def compute(
        retrieved_ids: List[str],
        relevant_ids: List[str],
    ) -> RetrievalScore:
        if not relevant_ids:
            return RetrievalScore(
                precision=1.0 if not retrieved_ids else 0.0,
                recall=1.0,
                f1=1.0 if not retrieved_ids else 0.0,
                retrieved_count=len(retrieved_ids),
                relevant_count=0,
                relevant_retrieved=0,
            )

        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        relevant_retrieved = retrieved_set & relevant_set

        precision = (
            len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0.0
        )
        recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return RetrievalScore(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            retrieved_count=len(retrieved_ids),
            relevant_count=len(relevant_ids),
            relevant_retrieved=len(relevant_retrieved),
        )

    @staticmethod
    def compute_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        relevant_set = set(relevant_ids)
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def compute_ndcg(
        retrieved_ids: List[str], relevant_ids: List[str], k: int = 10
    ) -> float:
        import math

        relevant_set = set(relevant_ids)
        dcg = 0.0
        for i, rid in enumerate(retrieved_ids[:k]):
            if rid in relevant_set:
                dcg += 1.0 / math.log2(i + 2)

        ideal_dcg = sum(
            1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k))
        )
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    @staticmethod
    def aggregate(scores: List[RetrievalScore]) -> Dict[str, float]:
        if not scores:
            return {"avg_precision": 0, "avg_recall": 0, "avg_f1": 0}

        return {
            "avg_precision": round(sum(s.precision for s in scores) / len(scores), 4),
            "avg_recall": round(sum(s.recall for s in scores) / len(scores), 4),
            "avg_f1": round(sum(s.f1 for s in scores) / len(scores), 4),
            "total_retrieved": sum(s.retrieved_count for s in scores),
            "total_relevant": sum(s.relevant_count for s in scores),
            "total_relevant_retrieved": sum(s.relevant_retrieved for s in scores),
        }


if __name__ == "__main__":
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = ["a", "c", "f", "g"]

    score = RetrievalMetrics.compute(retrieved, relevant)
    print(f"Precision: {score.precision}")
    print(f"Recall: {score.recall}")
    print(f"F1: {score.f1}")
    print(f"MRR: {RetrievalMetrics.compute_mrr(retrieved, relevant)}")
    print(f"NDCG@5: {RetrievalMetrics.compute_ndcg(retrieved, relevant, k=5)}")
