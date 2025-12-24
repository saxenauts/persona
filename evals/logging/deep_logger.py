"""
Deep Logger Utility

Centralized logging utility for capturing detailed evaluation metrics
in structured JSONL format.
"""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from .log_schema import QuestionLog, RunMetadata


class DeepLogger:
    """Deep logger for evaluation runs"""

    def __init__(self, output_dir: str = "evals/results", run_id: Optional[str] = None):
        """
        Initialize deep logger

        Args:
            output_dir: Directory to save logs
            run_id: Unique run identifier (auto-generated if not provided)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate run ID if not provided
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id

        # Create run directory
        self.run_dir = self.output_dir / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Log file paths
        self.deep_logs_path = self.run_dir / "deep_logs.jsonl"
        self.metadata_path = self.run_dir / "run_metadata.json"

        print(f"Deep logger initialized: {self.run_dir}")

    def log_question(self, question_log: QuestionLog):
        """
        Log a single question evaluation

        Args:
            question_log: QuestionLog object
        """
        # Validate log
        log_dict = question_log.model_dump()

        # Append to JSONL
        with open(self.deep_logs_path, 'a') as f:
            f.write(json.dumps(log_dict) + '\n')

    def save_metadata(self, metadata: RunMetadata):
        """
        Save run metadata

        Args:
            metadata: RunMetadata object
        """
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata.model_dump(), f, indent=2)

    def update_metadata(self, **kwargs):
        """
        Update run metadata with new values

        Args:
            **kwargs: Key-value pairs to update
        """
        # Load existing metadata if it exists
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                metadata_dict = json.load(f)
        else:
            metadata_dict = {
                "run_id": self.run_id,
                "benchmark": kwargs.get("benchmark", "unknown"),
                "started_at": datetime.now().isoformat(),
                "total_questions": 0,
                "questions_completed": 0,
                "questions_failed": 0,
            }

        # Update with new values
        metadata_dict.update(kwargs)

        # Save
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

    def load_logs(self) -> list:
        """
        Load all question logs from this run

        Returns:
            List of question log dictionaries
        """
        logs = []

        if not self.deep_logs_path.exists():
            return logs

        with open(self.deep_logs_path, 'r') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))

        return logs

    def get_summary(self) -> dict:
        """
        Get summary statistics for this run

        Returns:
            Dictionary with summary stats
        """
        logs = self.load_logs()

        if not logs:
            return {
                "total_questions": 0,
                "accuracy": 0.0,
                "avg_retrieval_time_ms": 0.0,
                "avg_generation_time_ms": 0.0,
            }

        total = len(logs)
        results = [log['evaluation'].get('correct') for log in logs]
        judged = [value for value in results if value is not None]
        correct = sum(1 for value in judged if value)
        judged_total = len(judged)
        skipped = total - judged_total

        retrieval_times = [log['retrieval']['duration_ms'] for log in logs]
        generation_times = [log['generation']['duration_ms'] for log in logs]
        extract_times = [log['ingestion'].get('extract_ms') for log in logs if log['ingestion'].get('extract_ms') is not None]
        embed_times = [log['ingestion'].get('embed_ms') for log in logs if log['ingestion'].get('embed_ms') is not None]
        persist_times = [log['ingestion'].get('persist_ms') for log in logs if log['ingestion'].get('persist_ms') is not None]
        total_ingest_times = [log['ingestion'].get('total_ms') for log in logs if log['ingestion'].get('total_ms') is not None]

        summary = {
            "total_questions": total,
            "correct": correct,
            "incorrect": judged_total - correct,
            "skipped": skipped,
            "accuracy": correct / judged_total if judged_total > 0 else 0.0,
            "avg_retrieval_time_ms": sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0,
            "avg_generation_time_ms": sum(generation_times) / len(generation_times) if generation_times else 0.0,
            "avg_extract_time_ms": sum(extract_times) / len(extract_times) if extract_times else 0.0,
            "avg_embed_time_ms": sum(embed_times) / len(embed_times) if embed_times else 0.0,
            "avg_persist_time_ms": sum(persist_times) / len(persist_times) if persist_times else 0.0,
            "avg_total_ingest_time_ms": sum(total_ingest_times) / len(total_ingest_times) if total_ingest_times else 0.0,
        }

        # Breakdown by question type
        type_stats = {}
        for log in logs:
            qtype = log['question_type']
            if qtype not in type_stats:
                type_stats[qtype] = {"total": 0, "correct": 0, "skipped": 0}

            result = log['evaluation'].get('correct')
            if result is None:
                type_stats[qtype]["skipped"] += 1
                continue

            type_stats[qtype]["total"] += 1
            if result:
                type_stats[qtype]["correct"] += 1

        # Calculate accuracy per type
        for qtype, stats in type_stats.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

        summary["type_breakdown"] = type_stats

        return summary

    def print_summary(self):
        """Print summary statistics to console"""
        summary = self.get_summary()

        print("\n" + "="*60)
        print(f"Evaluation Run Summary: {self.run_id}")
        print("="*60)
        print(f"Total Questions: {summary['total_questions']}")
        print(f"Correct: {summary['correct']}")
        print(f"Incorrect: {summary['incorrect']}")
        print(f"Skipped: {summary.get('skipped', 0)}")
        print(f"Accuracy: {summary['accuracy']:.2%}")
        print(f"\nAvg Retrieval Time: {summary['avg_retrieval_time_ms']:.1f} ms")
        print(f"Avg Generation Time: {summary['avg_generation_time_ms']:.1f} ms")
        if summary.get("avg_total_ingest_time_ms", 0.0):
            print(
                "Avg Ingest Time: "
                f"{summary.get('avg_total_ingest_time_ms', 0.0):.1f} ms "
                f"(extract {summary.get('avg_extract_time_ms', 0.0):.1f}, "
                f"embed {summary.get('avg_embed_time_ms', 0.0):.1f}, "
                f"persist {summary.get('avg_persist_time_ms', 0.0):.1f})"
            )

        if "type_breakdown" in summary:
            print("\n" + "-"*60)
            print("Breakdown by Question Type:")
            print("-"*60)
            for qtype, stats in sorted(summary["type_breakdown"].items()):
                print(f"{qtype:40s}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")

        print("="*60 + "\n")

    def save_summary(self, output_path: Optional[str] = None):
        """
        Save summary to JSON file

        Args:
            output_path: Optional custom output path
        """
        summary = self.get_summary()

        if output_path is None:
            output_path = self.run_dir / "summary.json"

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    from .log_schema import (
        QuestionLog, IngestionLog, RetrievalLog, GenerationLog,
        EvaluationLog, VectorSearchLog, GraphTraversalLog,
        SeedNode, MemoryCreationStats
    )

    # Create logger
    logger = DeepLogger(run_id="test_run")

    # Create a sample log
    sample_log = QuestionLog(
        question_id="test_q1",
        user_id="user_123",
        benchmark="longmemeval",
        question_type="multi-session",
        question="How many times did I visit the gym?",
        ingestion=IngestionLog(
            duration_ms=15000,
            sessions_count=10,
            memories_created=MemoryCreationStats(episodes=15, psyche=5, goals=2),
            nodes_created=22,
            relationships_created=45,
            embeddings_generated=22,
        ),
        retrieval=RetrievalLog(
            query="How many times did I visit the gym?",
            duration_ms=1500,
            vector_search=VectorSearchLog(
                top_k=5,
                seeds=[
                    SeedNode(node_id="ep_1", score=0.95, node_type="episode"),
                    SeedNode(node_id="ep_2", score=0.89, node_type="episode"),
                ]
            ),
            graph_traversal=GraphTraversalLog(
                max_hops=2,
                nodes_visited=15,
                relationships_traversed=30,
                final_ranked_nodes=["ep_1", "ep_2", "ep_3"],
                duration_ms=200
            ),
            context_size_tokens=2500
        ),
        generation=GenerationLog(
            duration_ms=2000,
            model="gpt-4o-mini",
            temperature=0.7,
            prompt_tokens=2600,
            completion_tokens=50,
            answer="You visited the gym 3 times."
        ),
        evaluation=EvaluationLog(
            gold_answer="3",
            correct=True,
            judge_response="yes",
            judge_model="gpt-4o",
            score_type="binary"
        )
    )

    # Log it
    logger.log_question(sample_log)

    # Update metadata
    logger.update_metadata(
        benchmark="longmemeval",
        total_questions=1,
        questions_completed=1
    )

    # Print summary
    logger.print_summary()

    # Save summary
    logger.save_summary()

    print(f"\nTest logs saved to: {logger.run_dir}")
