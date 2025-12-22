"""
CLI Interface for Evaluation Framework

Provides command-line interface for running evaluations.
"""

import typer
from pathlib import Path
from typing import Optional, List

from .config import EvalConfig

app = typer.Typer(
    name="evals",
    help="Persona Memory System Evaluation Framework"
)


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
        help="Adapter(s) to evaluate: persona, mem0, zep"
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
        print("\n  By Question Type:")
        for qtype, stats in result.get('type_accuracies', {}).items():
            print(f"    {qtype:40s}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['count']})")

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
        logger = DeepLogger(run_id=run_id)
        summary = logger.get_summary()
        summary['run_id'] = run_id
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
def create_configs():
    """Create default configuration files"""
    from .config import create_default_configs
    create_default_configs()


if __name__ == "__main__":
    app()
