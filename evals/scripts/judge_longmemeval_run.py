"""
Judge LongMemEval questions for an existing eval run.

Fills in evaluation.correct for longmemeval entries that were skipped.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable

from evals.longmemeval.evaluate_qa import (
    get_anscheck_prompt,
    query_openai_with_retry,
    parse_judge_response,
)


def _load_logs(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _judge_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    if entry.get("benchmark") != "longmemeval":
        return entry

    evaluation = entry.get("evaluation") or {}
    if evaluation.get("correct") is not None:
        return entry

    question_type = entry.get("question_type", "")
    question = entry.get("question", "")
    gold_answer = evaluation.get("gold_answer", "")
    response = (entry.get("generation") or {}).get("answer", "")
    abstention = "_abs" in entry.get("question_id", "")

    prompt = get_anscheck_prompt(
        task=question_type,
        question=question,
        answer=gold_answer,
        response=response,
        abstention=abstention
    )

    judge_response = query_openai_with_retry(prompt)
    correct = parse_judge_response(judge_response)

    evaluation["correct"] = correct
    evaluation["judge_response"] = judge_response
    evaluation["judge_model"] = os.getenv("EVAL_JUDGE_MODEL", "gpt-5-mini")
    evaluation["score_type"] = "binary"
    entry["evaluation"] = evaluation

    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge LongMemEval entries for an existing run."
    )
    parser.add_argument(
        "run_id",
        help="Run ID (with or without run_ prefix)."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional path to input deep_logs.jsonl"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to output judged logs (default: overwrite input)."
    )
    args = parser.parse_args()

    run_id = args.run_id
    if run_id.startswith("run_"):
        run_id = run_id[4:]

    default_path = Path("evals/results") / f"run_{run_id}" / "deep_logs.jsonl"
    input_path = Path(args.input) if args.input else default_path

    if not input_path.exists():
        raise FileNotFoundError(f"Log file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path
    backup_path = None
    original_path = input_path
    temp_path = None
    backup_path = None
    if output_path == input_path:
        backup_path = input_path.with_suffix(".raw.jsonl")
        if not backup_path.exists():
            input_path.replace(backup_path)
            input_path = backup_path
        else:
            input_path = original_path
        temp_path = original_path.with_suffix(".tmp.jsonl")
        output_path = temp_path

    judged = 0
    total = 0

    with output_path.open("w") as f:
        for entry in _load_logs(input_path):
            total += 1
            before = (entry.get("evaluation") or {}).get("correct")
            entry = _judge_entry(entry)
            after = (entry.get("evaluation") or {}).get("correct")
            if before is None and after is not None and entry.get("benchmark") == "longmemeval":
                judged += 1
            f.write(json.dumps(entry) + "\n")

    if temp_path:
        temp_path.replace(original_path)

    print(f"Judged {judged} LongMemEval entries (total logs: {total}).")
    print(f"Output: {original_path}")
    if backup_path:
        print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()
