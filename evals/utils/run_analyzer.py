"""
Analyze evaluation run results from deep_logs.jsonl files.
Provides extraction statistics, failure analysis, and accuracy breakdowns.
"""

import json
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ExtractionStats:
    total_questions: int = 0
    min_nodes: int = 0
    max_nodes: int = 0
    mean_nodes: float = 0.0
    median_nodes: float = 0.0
    stdev_nodes: float = 0.0
    low_extraction_count: int = 0
    low_extraction_threshold: int = 5

    def to_dict(self) -> dict:
        return {
            "total_questions": self.total_questions,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "mean_nodes": round(self.mean_nodes, 1),
            "median_nodes": self.median_nodes,
            "stdev_nodes": round(self.stdev_nodes, 1),
            "low_extraction_count": self.low_extraction_count,
            "low_extraction_pct": round(
                100 * self.low_extraction_count / self.total_questions, 1
            )
            if self.total_questions
            else 0,
        }


@dataclass
class FailureCluster:
    start_idx: int
    end_idx: int
    count: int
    question_ids: List[str] = field(default_factory=list)


class RunAnalyzer:
    def __init__(self, deep_logs_path: str):
        self.deep_logs_path = Path(deep_logs_path)
        self.data: List[dict] = []
        self._load()

    def _load(self):
        with open(self.deep_logs_path) as f:
            for line in f:
                self.data.append(json.loads(line))

    def get_extraction_stats(self, low_threshold: int = 5) -> ExtractionStats:
        node_counts = []
        for d in self.data:
            nodes = d.get("ingestion", {}).get("nodes_created", 0)
            node_counts.append(nodes)

        if not node_counts:
            return ExtractionStats()

        return ExtractionStats(
            total_questions=len(node_counts),
            min_nodes=min(node_counts),
            max_nodes=max(node_counts),
            mean_nodes=statistics.mean(node_counts),
            median_nodes=statistics.median(node_counts),
            stdev_nodes=statistics.stdev(node_counts) if len(node_counts) > 1 else 0,
            low_extraction_count=sum(1 for n in node_counts if n < low_threshold),
            low_extraction_threshold=low_threshold,
        )

    def get_node_distribution(
        self, buckets: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, int]:
        if buckets is None:
            buckets = [
                (0, 5),
                (5, 10),
                (10, 15),
                (15, 20),
                (20, 30),
                (30, 50),
                (50, 100),
            ]

        node_counts = [
            d.get("ingestion", {}).get("nodes_created", 0) for d in self.data
        ]

        dist = {}
        for lo, hi in buckets:
            count = sum(1 for n in node_counts if lo <= n < hi)
            dist[f"{lo}-{hi}"] = count

        count_100_plus = sum(1 for n in node_counts if n >= 100)
        dist["100+"] = count_100_plus

        return dist

    def get_failures(self, threshold: int = 5) -> List[dict]:
        failures = []
        for d in self.data:
            nodes = d.get("ingestion", {}).get("nodes_created", 0)
            if nodes < threshold:
                failures.append(
                    {
                        "question_id": d.get("question_id"),
                        "nodes": nodes,
                        "extract_ms": d.get("ingestion", {}).get("extract_ms", 0),
                        "correct": d.get("evaluation", {}).get("correct", False),
                        "timestamp": d.get("timestamp"),
                    }
                )
        return failures

    def find_failure_clusters(
        self, threshold: int = 5, gap: int = 5
    ) -> List[FailureCluster]:
        failures = self.get_failures(threshold)
        if not failures:
            return []

        def extract_idx(qid: str) -> int:
            try:
                return int(qid.split("_")[-1])
            except (ValueError, IndexError):
                return 0

        indices = sorted(
            [(extract_idx(f["question_id"]), f["question_id"]) for f in failures]
        )

        clusters = []
        current = [indices[0]]

        for i in range(1, len(indices)):
            if indices[i][0] - indices[i - 1][0] <= gap:
                current.append(indices[i])
            else:
                if len(current) > 1:
                    clusters.append(
                        FailureCluster(
                            start_idx=current[0][0],
                            end_idx=current[-1][0],
                            count=len(current),
                            question_ids=[c[1] for c in current],
                        )
                    )
                current = [indices[i]]

        if len(current) > 1:
            clusters.append(
                FailureCluster(
                    start_idx=current[0][0],
                    end_idx=current[-1][0],
                    count=len(current),
                    question_ids=[c[1] for c in current],
                )
            )

        return clusters

    def get_accuracy_by_type(self) -> Dict[str, dict]:
        by_type: Dict[str, List[bool]] = {}

        for d in self.data:
            qtype = d.get("question_type", "unknown")
            correct = d.get("evaluation", {}).get("correct", False)

            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(correct)

        return {
            qtype: {
                "total": len(results),
                "correct": sum(results),
                "accuracy": round(100 * sum(results) / len(results), 1),
            }
            for qtype, results in by_type.items()
        }

    def get_extraction_vs_accuracy_correlation(self, threshold: int = 5) -> dict:
        low_correct = 0
        low_total = 0
        normal_correct = 0
        normal_total = 0

        for d in self.data:
            nodes = d.get("ingestion", {}).get("nodes_created", 0)
            correct = d.get("evaluation", {}).get("correct", False)

            if nodes < threshold:
                low_total += 1
                if correct:
                    low_correct += 1
            else:
                normal_total += 1
                if correct:
                    normal_correct += 1

        return {
            "low_extraction": {
                "total": low_total,
                "correct": low_correct,
                "accuracy": round(100 * low_correct / low_total, 1) if low_total else 0,
            },
            "normal_extraction": {
                "total": normal_total,
                "correct": normal_correct,
                "accuracy": round(100 * normal_correct / normal_total, 1)
                if normal_total
                else 0,
            },
        }

    def summary(self) -> dict:
        total = len(self.data)
        correct = sum(
            1 for d in self.data if d.get("evaluation", {}).get("correct", False)
        )

        return {
            "total_questions": total,
            "correct": correct,
            "accuracy": round(100 * correct / total, 1) if total else 0,
            "extraction_stats": self.get_extraction_stats().to_dict(),
            "accuracy_by_type": self.get_accuracy_by_type(),
            "failure_clusters": [
                {"start": c.start_idx, "end": c.end_idx, "count": c.count}
                for c in self.find_failure_clusters()
            ],
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_analyzer.py <deep_logs.jsonl>")
        sys.exit(1)

    analyzer = RunAnalyzer(sys.argv[1])
    print(json.dumps(analyzer.summary(), indent=2))
