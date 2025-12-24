"""
Compare evaluation runs using a unified golden set manifest.

Outputs per-type accuracy plus latency/token averages for each run.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple


def _load_manifest(manifest_path: Path) -> Dict[Tuple[str, str], str]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Combined manifest not found at {manifest_path}. "
            "Run `python evals/scripts/generate_golden_sets.py` first."
        )

    with open(manifest_path, "r") as f:
        data = json.load(f)

    entries = data.get("questions", [])
    manifest = {}
    for entry in entries:
        benchmark = entry.get("benchmark")
        qid = entry.get("question_id")
        qtype = entry.get("question_type") or "unknown"
        if benchmark and qid:
            manifest[(benchmark, qid)] = qtype

    return manifest


def _load_logs(run_id: str) -> Iterable[Dict[str, Any]]:
    run_path = Path("evals/results") / run_id / "deep_logs.jsonl"
    if not run_path.exists():
        raise FileNotFoundError(f"Run logs not found at {run_path}")

    with open(run_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _format_metric(value: float, allow_zero: bool = False) -> str:
    if value <= 0 and not allow_zero:
        return "n/a"
    if value >= 1000:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _aggregate_run(run_id: str, manifest: Dict[Tuple[str, str], str]) -> Dict[str, Any]:
    manifest_keys = set(manifest.keys())
    seen_keys = set()

    by_type: Dict[str, Dict[str, Any]] = {}
    total_correct = 0
    total_count = 0

    for log in _load_logs(run_id):
        benchmark = log.get("benchmark")
        qid = log.get("question_id")
        if not benchmark or not qid:
            continue

        key = (benchmark, qid)
        if key not in manifest_keys:
            continue

        seen_keys.add(key)
        qtype = log.get("question_type") or manifest.get(key, "unknown")
        stats = by_type.setdefault(qtype, {
            "total": 0,
            "correct": 0,
            "retrieval_ms": [],
            "generation_ms": [],
            "prompt_tokens": [],
            "completion_tokens": []
        })

        stats["total"] += 1
        total_count += 1

        if log.get("evaluation", {}).get("correct"):
            stats["correct"] += 1
            total_correct += 1

        retrieval_ms = log.get("retrieval", {}).get("duration_ms") or 0
        generation_ms = log.get("generation", {}).get("duration_ms") or 0
        prompt_tokens = log.get("generation", {}).get("prompt_tokens") or 0
        completion_tokens = log.get("generation", {}).get("completion_tokens") or 0

        if retrieval_ms > 0:
            stats["retrieval_ms"].append(retrieval_ms)
        if generation_ms > 0:
            stats["generation_ms"].append(generation_ms)
        if prompt_tokens > 0:
            stats["prompt_tokens"].append(prompt_tokens)
        if completion_tokens > 0:
            stats["completion_tokens"].append(completion_tokens)

    missing = len(manifest_keys - seen_keys)
    summary = {
        "run_id": run_id,
        "total_questions": total_count,
        "correct": total_correct,
        "accuracy": total_correct / total_count if total_count else 0.0,
        "missing_questions": missing,
        "by_type": {}
    }

    for qtype, stats in by_type.items():
        total = stats["total"]
        correct = stats["correct"]
        summary["by_type"][qtype] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "avg_retrieval_ms": _mean(stats["retrieval_ms"]),
            "avg_generation_ms": _mean(stats["generation_ms"]),
            "avg_prompt_tokens": _mean(stats["prompt_tokens"]),
            "avg_completion_tokens": _mean(stats["completion_tokens"]),
        }

    return summary


def _print_run_summary(summary: Dict[str, Any]) -> None:
    print(f"\n=== {summary['run_id']} ===")
    print(f"Total: {summary['total_questions']} | Missing: {summary['missing_questions']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")


def _print_comparison(summaries: Dict[str, Dict[str, Any]], qtypes: list[str]) -> None:
    headers = ["Question Type"]
    for run_id in summaries.keys():
        headers.extend([
            f"{run_id} acc",
            f"{run_id} ret_ms",
            f"{run_id} prompt",
            f"{run_id} completion"
        ])

    col_widths = [max(12, len(h)) for h in headers]
    for idx, header in enumerate(headers):
        col_widths[idx] = max(col_widths[idx], len(header))

    def fmt_row(values: list[str]) -> str:
        return " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(values))

    print("\n" + fmt_row(headers))
    print("-" * (sum(col_widths) + (3 * (len(col_widths) - 1))))

    for qtype in qtypes:
        row = [qtype]
        for run_id, summary in summaries.items():
            stats = summary["by_type"].get(qtype, {})
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            acc = (correct / total) if total else 0.0
            row.extend([
                f"{acc:.2%} ({correct}/{total})" if total else "n/a",
                _format_metric(stats.get("avg_retrieval_ms", 0.0)),
                _format_metric(stats.get("avg_prompt_tokens", 0.0)),
                _format_metric(stats.get("avg_completion_tokens", 0.0)),
            ])
        print(fmt_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evaluation runs.")
    parser.add_argument(
        "--runs",
        required=True,
        help="Comma-separated run IDs (e.g., run_a,run_b)"
    )
    parser.add_argument(
        "--manifest",
        default="evals/data/golden_sets/combined_golden_set_manifest.json",
        help="Path to combined golden set manifest."
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write JSON summary."
    )
    args = parser.parse_args()

    manifest = _load_manifest(Path(args.manifest))
    summaries = {}

    for run_id in [r.strip() for r in args.runs.split(",") if r.strip()]:
        summary = _aggregate_run(run_id, manifest)
        summaries[run_id] = summary
        _print_run_summary(summary)

    qtypes = sorted({manifest[qid] for qid in manifest})
    _print_comparison(summaries, qtypes)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"manifest": args.manifest, "runs": summaries}, f, indent=2)
        print(f"\nSaved JSON summary to {args.output_json}")


if __name__ == "__main__":
    main()
