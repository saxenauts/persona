"""
CLI Interface for Evaluation Framework

Provides command-line interface for running evaluations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer

from .config import EvalConfig

app = typer.Typer(
    name="evals",
    help="Persona Memory System Evaluation Framework"
)


def _normalize_run_id(run_id: str) -> str:
    return run_id[4:] if run_id.startswith("run_") else run_id


def _load_logs(run_id: str) -> List[Dict[str, Any]]:
    run_id = _normalize_run_id(run_id)
    log_path = Path("evals/results") / f"run_{run_id}" / "deep_logs.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Run logs not found: {log_path}")

    logs = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


@app.command()
def run(
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration YAML file"
    ),
    benchmark: Optional[List[str]] = typer.Option(
        None,
        "--benchmark", "-b",
        help="Benchmark(s) to run: longmemeval, personamem"
    ),
    adapters: Optional[List[str]] = typer.Option(
        None,
        "--adapter", "-a",
        help="Adapter(s) to evaluate: persona, mem0, zep, graphiti"
    ),
    types: Optional[str] = typer.Option(
        None,
        "--types", "-t",
        help="Comma-separated question types to evaluate"
    ),
    samples: Optional[int] = typer.Option(
        None,
        "--samples", "-n",
        help="Number of samples per type"
    ),
    seed: int = typer.Option(
        42,
        "--seed", "-s",
        help="Random seed for reproducibility"
    ),
    workers: int = typer.Option(
        5,
        "--workers", "-w",
        help="Number of parallel workers"
    ),
    output_dir: str = typer.Option(
        "evals/results",
        "--output", "-o",
        help="Output directory for results"
    ),
    use_golden_set: bool = typer.Option(
        False,
        "--golden-set",
        help="Use pre-generated golden sets instead of sampling"
    ),
    skip_judge: bool = typer.Option(
        False,
        "--skip-judge",
        help="Skip LongMemEval LLM judge (log responses for later evaluation)"
    ),
):
    """
    Run evaluation benchmarks

    Examples:

        # Run with config file
        python -m evals.cli run --config evals/configs/full_eval.yaml

        # Quick test with LongMemEval
        python -m evals.cli run --benchmark longmemeval --samples 10

        # Run specific question types
        python -m evals.cli run --benchmark longmemeval \\
            --types multi-session,temporal-reasoning --samples 20

        # Compare multiple adapters
        python -m evals.cli run --benchmark personamem \\
            --adapter persona --adapter mem0
    """
    from .runner import EvaluationRunner

    # Load config from file or create from CLI args
    if config:
        eval_config = EvalConfig.from_yaml(config)
        print(f"✓ Loaded configuration from: {config}")
        if skip_judge:
            eval_config.skip_judge = True
        if adapters:
            eval_config.adapters = list(adapters)
    else:
        # Create config from CLI arguments
        from .config import BenchmarkConfig

        eval_config = EvalConfig()

        # Parse question types if provided
        question_types = None
        if types:
            question_types = [t.strip() for t in types.split(',')]

        # Set up benchmarks
        if benchmark:
            for bm in benchmark:
                bm = bm.lower()

                if bm == "longmemeval":
                    if question_types and samples:
                        sample_sizes = {qt: samples for qt in question_types}
                    else:
                        # Default sample sizes
                        sample_sizes = {
                            'single-session-user': samples or 10,
                            'multi-session': samples or 10,
                        }

                    eval_config.longmemeval = BenchmarkConfig(
                        source='evals/data/longmemeval_oracle.json',
                        sample_sizes=sample_sizes
                    )

                elif bm == "personamem":
                    if question_types and samples:
                        sample_sizes = {qt: samples for qt in question_types}
                    else:
                        # Default sample sizes
                        sample_sizes = {
                            'recall_user_shared_facts': samples or 10,
                        }

                    eval_config.personamem = BenchmarkConfig(
                        source='evals/data/personamem',
                        variant='32k',
                        sample_sizes=sample_sizes
                    )

        # Set adapters
        if adapters:
            eval_config.adapters = list(adapters)

        # Set other options
        eval_config.random_seed = seed
        eval_config.parallel_workers = workers
        eval_config.output_dir = output_dir
        eval_config.skip_judge = skip_judge

    # Create and run evaluation
    runner = EvaluationRunner(eval_config, use_golden_set=use_golden_set)
    results = runner.run()

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    for benchmark_name, result in results.items():
        print(f"\n{benchmark_name.upper()}:")
        print(f"  Overall Accuracy: {result['overall_accuracy']:.2%}")
        print(f"  Total Questions: {result['total_questions']}")
        if "judged_questions" in result:
            print(f"  Judged Questions: {result['judged_questions']}")
        if "skipped_questions" in result:
            print(f"  Skipped Questions: {result['skipped_questions']}")
        print("\n  By Question Type:")
        for qtype, stats in result.get('type_accuracies', {}).items():
            print(f"    {qtype:40s}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['count']})")

    # Print rate limiter stats
    try:
        from persona.llm.rate_limiter import get_rate_limiter_registry
        registry = get_rate_limiter_registry()
        stats = registry.get_all_stats()
        if stats:
            print("\n" + "-"*40)
            print("RATE LIMITER METRICS:")
            for s in stats:
                print(f"  {s['name']}:")
                print(f"    Requests: {s['total_requests']}, Tokens: {s['total_tokens']:,}")
                print(f"    Wait time: {s['wait_time_ms']:.0f}ms, 429s: {s['retries_429']}")
    except Exception as e:
        pass  # Rate limiter not used

    print(f"\nResults saved to: {eval_config.output_dir}")


@app.command()
def analyze(
    run_id: str = typer.Argument(..., help="Run ID to analyze"),
    summary: bool = typer.Option(False, "--summary", help="Show summary report"),
    qtype: Optional[str] = typer.Option(None, "--type", help="Filter by question type"),
    failures: bool = typer.Option(False, "--failures", help="Show only failures"),
    retrieval: bool = typer.Option(False, "--retrieval", help="Analyze retrieval quality"),
):
    """
    Analyze evaluation results

    Examples:

        # Summary report
        python -m evals.cli analyze run_20241221_143052 --summary

        # Analyze failures for a specific type
        python -m evals.cli analyze run_20241221_143052 \\
            --type multi-session --failures

        # Retrieval quality analysis
        python -m evals.cli analyze run_20241221_143052 --retrieval
    """
    from .logging.deep_logger import DeepLogger

    # Strip run_ prefix if user passed it (DeepLogger adds it)
    if run_id.startswith("run_"):
        run_id = run_id[4:]
    
    logger = DeepLogger(run_id=run_id)

    if summary:
        logger.print_summary()
        return

    # Load logs
    logs = logger.load_logs()

    if not logs:
        print(f"No logs found for run: {run_id}")
        return

    # Filter by question type if specified
    if qtype:
        logs = [log for log in logs if log['question_type'] == qtype]
        print(f"\nFiltered to {len(logs)} questions of type '{qtype}'")

    # Filter failures if specified
    if failures:
        logs = [log for log in logs if not log['evaluation']['correct']]
        print(f"\nShowing {len(logs)} failed questions")

    # Print log details
    for log in logs:
        print("\n" + "="*80)
        print(f"Question ID: {log['question_id']}")
        print(f"Type: {log['question_type']}")
        print(f"Question: {log['question']}")
        print(f"\nGenerated Answer: {log['generation']['answer']}")
        print(f"Gold Answer: {log['evaluation']['gold_answer']}")
        print(f"Correct: {log['evaluation']['correct']}")

        if retrieval:
            print("\nRetrieval Stats:")
            print(f"  Duration: {log['retrieval']['duration_ms']:.1f} ms")
            print(f"  Context tokens: {log['retrieval']['context_size_tokens']}")
            print(f"  Top seeds:")
            for seed in log['retrieval']['vector_search']['seeds'][:3]:
                print(f"    - {seed['node_id']} (score: {seed['score']:.3f})")


@app.command()
def compare(
    run_ids: List[str] = typer.Argument(..., help="Run IDs to compare"),
):
    """
    Compare multiple evaluation runs

    Examples:

        # Compare two runs
        python -m evals.cli compare run_a run_b

        # Compare three runs
        python -m evals.cli compare run_a run_b run_c
    """
    from .logging.deep_logger import DeepLogger

    print("\n" + "="*80)
    print("COMPARING EVALUATION RUNS")
    print("="*80)

    summaries = []
    for run_id in run_ids:
        clean_id = _normalize_run_id(run_id)
        logger = DeepLogger(run_id=clean_id)
        summary = logger.get_summary()
        summary['run_id'] = clean_id
        summaries.append(summary)

    # Print comparison table
    print(f"\n{'Run ID':<20s} {'Total':<8s} {'Accuracy':<10s} {'Avg Retrieval (ms)':<20s}")
    print("-" * 80)

    for summary in summaries:
        print(
            f"{summary['run_id']:<20s} "
            f"{summary['total_questions']:<8d} "
            f"{summary['accuracy']:>8.1%}  "
            f"{summary['avg_retrieval_time_ms']:>18.1f}"
        )

    # Compare by question type
    all_types = set()
    for summary in summaries:
        all_types.update(summary.get('type_breakdown', {}).keys())

    if all_types:
        print("\n" + "="*80)
        print("Accuracy by Question Type:")
        print("="*80)

        for qtype in sorted(all_types):
            print(f"\n{qtype}:")
            for summary in summaries:
                type_stats = summary.get('type_breakdown', {}).get(qtype, {})
                if type_stats:
                    acc = type_stats.get('accuracy', 0)
                    print(f"  {summary['run_id']:<20s}: {acc:>6.1%}")


@app.command()
def aggregate(
    runs: str = typer.Option(..., "--runs", help="Comma-separated run IDs"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Optional JSON output path"),
):
    """
    Aggregate timing metrics across multiple runs.
    """
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    if not run_ids:
        print("No run IDs provided.")
        raise typer.Exit(code=1)

    def parse_ts(ts: str) -> datetime:
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")

    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summaries = []
    for run_id in run_ids:
        logs = _load_logs(run_id)
        if not logs:
            continue

        start = min(parse_ts(log["timestamp"]) for log in logs)
        end = max(parse_ts(log["timestamp"]) for log in logs)
        elapsed = (end - start).total_seconds()
        qpm = len(logs) / (elapsed / 60) if elapsed > 0 else 0.0

        ingest_times = [log["ingestion"]["duration_ms"] for log in logs]
        retrieval_times = [log["retrieval"]["duration_ms"] for log in logs]
        generation_times = [log["generation"]["duration_ms"] for log in logs]
        prompt_tokens = [log["generation"]["prompt_tokens"] for log in logs if log["generation"]["prompt_tokens"]]
        completion_tokens = [log["generation"]["completion_tokens"] for log in logs if log["generation"]["completion_tokens"]]
        extract_times = [log["ingestion"].get("extract_ms") for log in logs if log["ingestion"].get("extract_ms") is not None]
        embed_times = [log["ingestion"].get("embed_ms") for log in logs if log["ingestion"].get("embed_ms") is not None]
        persist_times = [log["ingestion"].get("persist_ms") for log in logs if log["ingestion"].get("persist_ms") is not None]
        total_ingest_times = [log["ingestion"].get("total_ms") for log in logs if log["ingestion"].get("total_ms") is not None]

        summary = {
            "run_id": _normalize_run_id(run_id),
            "questions": len(logs),
            "elapsed_s": elapsed,
            "qpm": qpm,
            "avg_ingest_ms": mean(ingest_times),
            "avg_extract_ms": mean(extract_times),
            "avg_embed_ms": mean(embed_times),
            "avg_persist_ms": mean(persist_times),
            "avg_total_ingest_ms": mean(total_ingest_times),
            "avg_retrieval_ms": mean(retrieval_times),
            "avg_generation_ms": mean(generation_times),
            "avg_prompt_tokens": mean(prompt_tokens),
            "avg_completion_tokens": mean(completion_tokens),
        }
        summaries.append(summary)

        print(
            f"{summary['run_id']}: qpm {summary['qpm']:.2f} | "
            f"ingest {summary['avg_ingest_ms']:.0f}ms "
            f"(extract {summary['avg_extract_ms']:.0f}, embed {summary['avg_embed_ms']:.0f}, "
            f"persist {summary['avg_persist_ms']:.0f}) | "
            f"retrieval {summary['avg_retrieval_ms']:.0f}ms | "
            f"generation {summary['avg_generation_ms']:.0f}ms"
        )

    if summaries:
        def mean_key(key: str) -> float:
            return mean([s[key] for s in summaries])

        aggregate_summary = {
            "runs": [s["run_id"] for s in summaries],
            "avg_qpm": mean_key("qpm"),
            "avg_ingest_ms": mean_key("avg_ingest_ms"),
            "avg_extract_ms": mean_key("avg_extract_ms"),
            "avg_embed_ms": mean_key("avg_embed_ms"),
            "avg_persist_ms": mean_key("avg_persist_ms"),
            "avg_total_ingest_ms": mean_key("avg_total_ingest_ms"),
            "avg_retrieval_ms": mean_key("avg_retrieval_ms"),
            "avg_generation_ms": mean_key("avg_generation_ms"),
            "avg_prompt_tokens": mean_key("avg_prompt_tokens"),
            "avg_completion_tokens": mean_key("avg_completion_tokens"),
        }

        print(
            "\nAverage: "
            f"qpm {aggregate_summary['avg_qpm']:.2f} | "
            f"ingest {aggregate_summary['avg_ingest_ms']:.0f}ms | "
            f"retrieval {aggregate_summary['avg_retrieval_ms']:.0f}ms | "
            f"generation {aggregate_summary['avg_generation_ms']:.0f}ms"
        )

        if output_json:
            with open(output_json, "w") as f:
                json.dump({"runs": summaries, "aggregate": aggregate_summary}, f, indent=2)
            print(f"Saved aggregate stats to: {output_json}")


@app.command()
def judge(
    run_id: str = typer.Argument(..., help="Run ID to judge"),
    input_path: Optional[str] = typer.Option(None, "--input", help="Optional input log path"),
    output_path: Optional[str] = typer.Option(None, "--output", help="Optional output log path"),
):
    """
    Judge LongMemEval questions for an existing eval run.
    """
    from .longmemeval.evaluate_qa import (
        get_anscheck_prompt,
        query_openai_with_retry,
        parse_judge_response,
    )

    run_id = _normalize_run_id(run_id)
    default_path = Path("evals/results") / f"run_{run_id}" / "deep_logs.jsonl"
    input_file = Path(input_path) if input_path else default_path
    if not input_file.exists():
        print(f"Log file not found: {input_file}")
        raise typer.Exit(code=1)

    output_file = Path(output_path) if output_path else input_file
    backup_path = None
    temp_path = None
    if output_file == input_file:
        backup_path = input_file.with_suffix(".raw.jsonl")
        if not backup_path.exists():
            input_file.replace(backup_path)
        input_file = backup_path
        temp_path = output_file.with_suffix(".tmp.jsonl")
        output_file = temp_path

    judged = 0
    total = 0

    with output_file.open("w") as f:
        with input_file.open("r") as source:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                total += 1

                if entry.get("benchmark") == "longmemeval":
                    evaluation = entry.get("evaluation") or {}
                    if evaluation.get("correct") is None:
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
                        evaluation["judge_model"] = "longmemeval"
                        evaluation["score_type"] = "binary"
                        entry["evaluation"] = evaluation
                        judged += 1

                f.write(json.dumps(entry) + "\n")

    if temp_path:
        temp_path.replace(default_path)
        output_file = default_path

    print(f"Judged {judged} LongMemEval entries (total logs: {total}).")
    print(f"Output: {output_file}")
    if backup_path:
        print(f"Backup: {backup_path}")


@app.command()
def create_configs():
    """Create default configuration files"""
    from .config import create_default_configs
    create_default_configs()


if __name__ == "__main__":
    app()
