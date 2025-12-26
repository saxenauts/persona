"""Main analyzer that ties together database, metrics, and classification."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from .database import EvalDatabase
from .metrics import RetrievalMetrics, RetrievalScore
from .failure_classifier import FailureClassifier, FailureClassification


class EvalAnalyzer:
    def __init__(self, db_path: str = "eval_results.db"):
        self.db = EvalDatabase(db_path)
        self.classifier = FailureClassifier()

    def import_run(
        self, jsonl_path: str, run_id: str, system_name: str
    ) -> Dict[str, Any]:
        count = self.db.import_from_jsonl(jsonl_path, run_id, system_name)
        return {
            "imported": count,
            "run_id": run_id,
            "system_name": system_name,
            "db_path": str(self.db.db_path),
        }

    def auto_classify_failures(
        self, system_name: Optional[str] = None
    ) -> Dict[str, int]:
        failures = self.db.get_failures(system_name=system_name, limit=1000)
        category_counts: Dict[str, int] = {}

        for failure in failures:
            if failure.get("failure_category"):
                continue

            classification = self.classifier.classify(
                question_text=failure["question_text"],
                question_type=failure["question_type"],
                gold_answer=failure["gold_answer"],
                generated_answer=failure["generated_answer"],
                retrieved_count=failure["retrieved_node_count"] or 0,
                session_count=failure["session_count"] or 0,
                nodes_created=failure["nodes_created"] or 0,
            )

            self.db.annotate_failure(
                result_id=failure["id"],
                failure_category=classification.category,
                notes=f"Auto: {classification.reasoning} (conf: {classification.confidence})",
            )

            category_counts[classification.category] = (
                category_counts.get(classification.category, 0) + 1
            )

        return category_counts

    def get_summary(self, system_name: Optional[str] = None) -> Dict[str, Any]:
        return self.db.get_failure_summary(system_name)

    def get_failures_for_review(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.db.get_unannotated_failures(limit)

    def annotate(self, result_id: int, category: str, notes: Optional[str] = None):
        self.db.annotate_failure(result_id, category, notes)

    def get_taxonomy(self) -> List[Dict[str, Any]]:
        return self.db.get_taxonomy()

    def generate_report(self, system_name: Optional[str] = None) -> str:
        summary = self.get_summary(system_name)
        taxonomy = self.get_taxonomy()

        lines = [
            "# Evaluation Analysis Report",
            "",
            f"**System:** {system_name or 'All'}",
            "",
            "## Overall Performance",
            "",
            f"- **Total Questions:** {summary['overall']['total']}",
            f"- **Correct:** {summary['overall']['correct']}",
            f"- **Accuracy:** {summary['overall']['accuracy']}%",
            "",
            "## Performance by Question Type",
            "",
            "| Type | Total | Correct | Accuracy |",
            "|------|-------|---------|----------|",
        ]

        for t in summary["by_type"]:
            lines.append(
                f"| {t['question_type']} | {t['total']} | {t['correct']} | {t['accuracy']:.1f}% |"
            )

        lines.extend(
            [
                "",
                "## Failure Taxonomy",
                "",
                "| Category | Count | Description |",
                "|----------|-------|-------------|",
            ]
        )

        taxonomy_dict = {t["category"]: t for t in taxonomy}
        for cat in summary.get("by_failure_category", []):
            desc = taxonomy_dict.get(cat["failure_category"], {}).get("description", "")
            lines.append(f"| {cat['failure_category']} | {cat['count']} | {desc} |")

        return "\n".join(lines)


def analyze_graphiti_run():
    analyzer = EvalAnalyzer(
        db_path="/Users/saxenauts/Documents/InnerNets AI Inc/persona/persona/evals/results/eval_analysis.db"
    )

    jsonl_path = "/Users/saxenauts/Documents/InnerNets AI Inc/persona/persona/evals/results/run_20251225_124714/deep_logs.jsonl"

    print("Importing Graphiti PersonaMem run...")
    result = analyzer.import_run(jsonl_path, "graphiti_personamem_20251225", "graphiti")
    print(f"Imported {result['imported']} questions")

    print("\nAuto-classifying failures...")
    classifications = analyzer.auto_classify_failures("graphiti")
    print(f"Classified: {classifications}")

    print("\nGenerating summary...")
    summary = analyzer.get_summary("graphiti")
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(analyzer.generate_report("graphiti"))


if __name__ == "__main__":
    analyze_graphiti_run()
