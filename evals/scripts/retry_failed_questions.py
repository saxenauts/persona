#!/usr/bin/env python3
"""
Retry failed questions from a Graphiti eval run.

This script:
1. Loads failed question IDs from analysis files
2. Creates a filtered dataset with only those questions
3. Runs the eval on the filtered set
4. Saves results for merging with original run

Usage:
    python evals/scripts/retry_failed_questions.py

    # With environment overrides for timeout fix:
    GRAPHITI_HTTP_CLIENT_TIMEOUT_S=900 python evals/scripts/retry_failed_questions.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_failed_question_ids():
    """Load all failed question IDs from analysis files."""
    analysis_dir = Path("evals/analysis")

    failed_ids = set()

    # Load timeout questions
    timeout_file = analysis_dir / "graphiti_timeout_questions.json"
    if timeout_file.exists():
        with open(timeout_file) as f:
            data = json.load(f)
            for q in data:
                failed_ids.add(q["question_id"])
        print(f"[Retry] Loaded {len(data)} timeout questions from {timeout_file.name}")

    # Load incomplete questions (stuck run)
    incomplete_file = analysis_dir / "graphiti_incomplete_questions.json"
    if incomplete_file.exists():
        with open(incomplete_file) as f:
            data = json.load(f)
            questions = data.get("questions", data)  # Handle both formats
            if isinstance(questions, list):
                for q in questions:
                    failed_ids.add(q["question_id"])
                print(f"[Retry] Loaded {len(questions)} incomplete questions from {incomplete_file.name}")

    return failed_ids


def create_filtered_dataset(failed_ids: set, output_path: str):
    """Create a filtered dataset with only the failed questions."""
    # Load full LongMemEval dataset
    oracle_path = Path("evals/data/longmemeval_oracle.json")

    with open(oracle_path) as f:
        full_dataset = json.load(f)

    # Filter to only failed questions
    filtered = [q for q in full_dataset if q["question_id"] in failed_ids]

    # Save filtered dataset
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"[Retry] Created filtered dataset: {len(filtered)} questions -> {output}")

    # Print breakdown by type
    type_counts = {}
    for q in filtered:
        qtype = q.get("question_type", "unknown")
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    print(f"[Retry] Breakdown by type:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")

    return filtered


def create_retry_config(filtered_data_path: str, output_dir: str):
    """Create a YAML config for the retry run."""
    config_content = f"""# Retry config for failed questions
# Generated: {datetime.now().isoformat()}

longmemeval:
  source: {filtered_data_path}
  full_dataset: true

global:
  random_seed: 42
  adapters:
  - graphiti
  parallel_workers: 8
  checkpoint_dir: {output_dir}
  deep_logging: true
  output_dir: {output_dir}
"""

    config_path = Path("evals/configs/retry_failed.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"[Retry] Created config: {config_path}")
    return str(config_path)


def main():
    print("=" * 60)
    print("GRAPHITI FAILED QUESTIONS RETRY")
    print("=" * 60)
    print()

    # Check timeout configuration
    http_timeout = os.getenv("GRAPHITI_HTTP_CLIENT_TIMEOUT_S", "900")
    episode_timeout = os.getenv("GRAPHITI_INGEST_TIMEOUT_S", "600")
    print(f"[Config] HTTP client timeout: {http_timeout}s")
    print(f"[Config] Per-episode timeout: {episode_timeout}s")
    print()

    # Step 1: Load failed question IDs
    failed_ids = load_failed_question_ids()
    print(f"\n[Retry] Total unique failed questions: {len(failed_ids)}")

    if not failed_ids:
        print("[Retry] No failed questions found. Nothing to retry.")
        return

    # Step 2: Create filtered dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filtered_path = f"evals/data/retry_failed_{timestamp}.json"
    output_dir = f"evals/results/retry_{timestamp}"

    create_filtered_dataset(failed_ids, filtered_path)

    # Step 3: Create retry config
    config_path = create_retry_config(filtered_path, output_dir)

    # Step 4: Print run command
    print()
    print("=" * 60)
    print("READY TO RUN")
    print("=" * 60)
    print()
    print("Run the following command to start the retry:")
    print()
    print(f"  cd /Users/saxenauts/Documents/InnerNets\\ AI\\ Inc/persona/persona")
    print(f"  GRAPHITI_HTTP_CLIENT_TIMEOUT_S=900 python -m evals.cli run --config {config_path}")
    print()
    print(f"Results will be saved to: {output_dir}/")
    print()

    # Optional: Ask if user wants to run now
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        print("[Retry] Starting eval run...")
        os.system(f"cd /Users/saxenauts/Documents/InnerNets\\ AI\\ Inc/persona/persona && GRAPHITI_HTTP_CLIENT_TIMEOUT_S=900 python -m evals.cli run --config {config_path}")


if __name__ == "__main__":
    main()
