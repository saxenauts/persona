#!/usr/bin/env python3
"""
Live Eval Monitor - Real-time progress tracking with ETA and detailed metrics.

Watches deep_logs.jsonl and displays:
- Progress counter (X/Y questions)
- ETA countdown
- Per-question metrics (ingest time, retrieval time, accuracy)
- Running averages
- Success/failure breakdown

Usage:
    python -m evals.scripts.live_monitor [run_id]

    If run_id not specified, watches the latest run.
"""

import json
import time
import sys
import os
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import argparse


def fetch_rate_limiter_stats(base_url: str = "http://localhost:8000") -> Optional[dict]:
    try:
        resp = requests.get(f"{base_url}/api/v1/stats", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def find_latest_run(results_dir: str = "evals/results") -> Optional[Path]:
    """Find the most recent run directory."""
    results_path = Path(results_dir)
    run_dirs = sorted(
        [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def load_logs(log_file: Path) -> list:
    """Load all log entries from JSONL file."""
    logs = []
    if log_file.exists():
        with open(log_file, "r") as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
    return logs


def calculate_metrics(logs: list) -> dict:
    """Calculate aggregate metrics from logs."""
    if not logs:
        return {}

    correct = sum(1 for log in logs if log.get("evaluation", {}).get("correct", False))
    total = len(logs)

    ingest_times = [log.get("ingestion", {}).get("duration_ms", 0) for log in logs]
    retrieval_times = [log.get("retrieval", {}).get("duration_ms", 0) for log in logs]
    gen_times = [log.get("generation", {}).get("duration_ms", 0) for log in logs]

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total * 100 if total > 0 else 0,
        "avg_ingest_ms": sum(ingest_times) / len(ingest_times) if ingest_times else 0,
        "avg_retrieval_ms": sum(retrieval_times) / len(retrieval_times)
        if retrieval_times
        else 0,
        "avg_gen_ms": sum(gen_times) / len(gen_times) if gen_times else 0,
        "total_nodes": sum(
            log.get("ingestion", {}).get("nodes_created", 0) for log in logs
        ),
        "total_relationships": sum(
            log.get("ingestion", {}).get("relationships_created", 0) for log in logs
        ),
    }


def format_duration(ms: float) -> str:
    """Format milliseconds as human-readable duration."""
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_progress(
    logs: list,
    target_questions: int,
    start_time: datetime,
    rate_stats: Optional[dict] = None,
):
    os.system("clear" if os.name == "posix" else "cls")

    metrics = calculate_metrics(logs)
    elapsed = (datetime.now() - start_time).total_seconds()

    completed = metrics.get("total", 0)
    remaining = target_questions - completed

    if completed > 0:
        avg_time_per_q = elapsed / completed
        eta_seconds = remaining * avg_time_per_q
        eta = timedelta(seconds=int(eta_seconds))
    else:
        eta = "calculating..."

    print("=" * 60)
    print("🔬 PERSONA EVAL LIVE MONITOR")
    print("=" * 60)
    print()

    pct = completed / target_questions * 100 if target_questions > 0 else 0
    bar_len = 40
    filled = int(bar_len * completed / target_questions) if target_questions > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"Progress: [{bar}] {pct:.1f}%")
    print(f"          {completed}/{target_questions} questions")
    print()

    print(f"⏱️  Elapsed:  {timedelta(seconds=int(elapsed))}")
    print(f"📍 ETA:      {eta}")
    print()

    print(
        f"✅ Accuracy: {metrics.get('accuracy', 0):.1f}% ({metrics.get('correct', 0)}/{completed})"
    )
    print()

    if rate_stats and rate_stats.get("rate_limiters"):
        print("🚦 RATE LIMITER STATS")
        print("-" * 50)
        for limiter in rate_stats["rate_limiters"]:
            name = limiter.get("name", "unknown")[:25]
            total_req = limiter.get("total_requests", 0)
            total_tok = limiter.get("total_tokens", 0)
            wait_ms = limiter.get("wait_time_ms", 0)
            retries = limiter.get("retries_429", 0)
            bucket_req = limiter.get("bucket_requests", 0)

            if elapsed > 0:
                rpm = total_req / elapsed * 60
                tpm = total_tok / elapsed * 60
            else:
                rpm, tpm = 0, 0

            print(
                f"  {name:25} | RPM: {rpm:6.1f} | TPM: {tpm:>10,.0f} | Wait: {wait_ms:>6}ms | 429s: {retries}"
            )
        print()

    print("📊 PERFORMANCE METRICS")
    print("-" * 30)
    print(f"  Avg Ingest:    {format_duration(metrics.get('avg_ingest_ms', 0))}")
    print(f"  Avg Retrieval: {format_duration(metrics.get('avg_retrieval_ms', 0))}")
    print(f"  Avg Generate:  {format_duration(metrics.get('avg_gen_ms', 0))}")
    print()

    print("📈 GRAPH STATS")
    print("-" * 30)
    print(f"  Total Nodes:         {metrics.get('total_nodes', 0)}")
    print(f"  Total Relationships: {metrics.get('total_relationships', 0)}")
    print()

    if logs:
        print("📝 RECENT QUESTIONS")
        print("-" * 30)
        for log in logs[-5:]:
            status = "✅" if log.get("evaluation", {}).get("correct") else "❌"
            q_type = log.get("question_type", "?")[:15]
            ingest_t = log.get("ingestion", {}).get("duration_ms", 0)
            retr_t = log.get("retrieval", {}).get("duration_ms", 0)
            print(
                f"  {status} {q_type:15} | ingest: {format_duration(ingest_t):>7} | retr: {format_duration(retr_t):>7}"
            )

    print()
    print(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    print("Press Ctrl+C to exit")


def main():
    parser = argparse.ArgumentParser(description="Live Eval Monitor")
    parser.add_argument("run_id", nargs="?", help="Run ID to monitor (default: latest)")
    parser.add_argument(
        "--target", "-t", type=int, default=326, help="Target number of questions"
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=5.0, help="Refresh interval in seconds"
    )
    args = parser.parse_args()

    # Find run directory
    if args.run_id:
        run_dir = Path(f"evals/results/run_{args.run_id}")
        if not run_dir.exists():
            run_dir = Path(f"evals/results/{args.run_id}")
    else:
        run_dir = find_latest_run()

    if not run_dir or not run_dir.exists():
        print("❌ No run directory found. Start an evaluation first.")
        sys.exit(1)

    log_file = run_dir / "deep_logs.jsonl"
    print(f"🔍 Monitoring: {log_file}")
    time.sleep(1)

    start_time = datetime.now()
    prev_count = 0

    try:
        while True:
            logs = load_logs(log_file)
            rate_stats = fetch_rate_limiter_stats()

            if logs and prev_count == 0:
                first_ts = logs[0].get("timestamp")
                if first_ts:
                    start_time = datetime.fromisoformat(first_ts)

            print_progress(logs, args.target, start_time, rate_stats)

            if len(logs) >= args.target:
                print("\n🎉 EVALUATION COMPLETE!")
                break

            prev_count = len(logs)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")


if __name__ == "__main__":
    main()
