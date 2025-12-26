"""
Main Engine class for orchestrating evaluations.

Design Principles:
- Engine is the high-level orchestrator
- Loads benchmarks, adapters, metrics from registries
- Delegates execution to AsyncExecutor
- Emits events to storage
- Produces aggregated reports
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping, Optional, Sequence

from ..core.models import (
    TestCase,
    QueryResult,
    MetricResult,
    EvalResult,
    RunSpec,
    Session,
)
from ..core.interfaces import (
    MemorySystemAdapter,
    Metric,
    Benchmark,
    Event,
    EventStore,
    AdapterCapabilities,
)
from .executor import AsyncExecutor, ExecutorConfig, TokenBucket


@dataclass
class EngineConfig:
    """Configuration for the evaluation engine."""

    # Execution settings
    concurrency: int = 10
    rate_limit_rpm: Optional[int] = None
    timeout_seconds: float = 300.0
    retry_max_attempts: int = 5

    # Sampling (for quick runs)
    sample_sizes: Optional[Mapping[str, int]] = None
    sample_seed: Optional[int] = None

    # Output settings
    verbose: bool = False
    progress_callback: Optional[Callable[[int, int, EvalResult], None]] = None


@dataclass
class RunResult:
    """
    Complete result of an evaluation run.

    Contains all individual results plus aggregate statistics.
    """

    run_id: str
    run_spec: RunSpec
    started_at: datetime
    finished_at: datetime

    # Individual results
    results: Sequence[EvalResult]

    # Aggregate statistics
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    error_cases: int = 0

    # Per-question-type breakdown
    by_question_type: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    # Per-metric breakdown
    by_metric: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Overall pass rate."""
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Run: {self.run_id}",
            f"Duration: {(self.finished_at - self.started_at).total_seconds():.1f}s",
            f"Total: {self.total_cases} | Passed: {self.passed_cases} | Failed: {self.failed_cases} | Errors: {self.error_cases}",
            f"Pass Rate: {self.pass_rate:.1%}",
        ]

        if self.by_question_type:
            lines.append("\nBy Question Type:")
            for qtype, stats in sorted(self.by_question_type.items()):
                rate = stats.get("pass_rate", 0.0)
                count = stats.get("total", 0)
                lines.append(f"  {qtype}: {rate:.1%} ({count} cases)")

        return "\n".join(lines)


class Engine:
    """
    Main evaluation engine.

    Orchestrates the evaluation lifecycle:
    1. Load benchmark → get test cases
    2. Initialize adapter
    3. Initialize metrics
    4. Run evaluation via executor
    5. Aggregate and report results

    Usage:
        engine = Engine(config=EngineConfig(concurrency=20))

        result = await engine.run(
            benchmark="personamem",
            adapter=my_adapter,
            metrics=["option_extractor", "context_relevance"],
            resources={"llm": llm_client},
        )

        print(result.summary())
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        event_store: Optional[EventStore] = None,
    ):
        self.config = config or EngineConfig()
        self.event_store = event_store

        # Create executor
        executor_config = ExecutorConfig(
            concurrency=self.config.concurrency,
            rate_limit_rpm=self.config.rate_limit_rpm,
            retry_max_attempts=self.config.retry_max_attempts,
            timeout_seconds=self.config.timeout_seconds,
        )
        self.executor = AsyncExecutor(config=executor_config)

        # Metric registry (name -> metric instance)
        self._metrics: dict[str, Metric] = {}

    def register_metric(self, metric: Metric) -> None:
        """Register a metric instance."""
        self._metrics[metric.name] = metric

    def register_metrics(self, metrics: Sequence[Metric]) -> None:
        """Register multiple metrics."""
        for m in metrics:
            self.register_metric(m)

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a registered metric by name."""
        return self._metrics.get(name)

    async def run(
        self,
        benchmark: Benchmark,
        adapter: MemorySystemAdapter,
        metric_names: Sequence[str],
        *,
        resources: Optional[Mapping[str, Any]] = None,
        variant: Optional[str] = None,
    ) -> RunResult:
        """
        Run a complete evaluation.

        Args:
            benchmark: Benchmark instance to load test cases from
            adapter: Memory system adapter to evaluate
            metric_names: Names of metrics to use (must be registered)
            resources: Shared resources (LLM clients, embedders, etc.)
            variant: Benchmark variant (e.g., "32k", "128k")

        Returns:
            RunResult with all results and aggregates
        """
        run_id = str(uuid.uuid4())[:8]
        started_at = datetime.utcnow()
        resources = resources or {}

        # Create run spec
        run_spec = RunSpec(
            run_id=run_id,
            created_at=started_at,
            adapters=(adapter.name,),
            benchmarks=(benchmark.name,),
            metric_names=tuple(metric_names),
            concurrency=self.config.concurrency,
            rate_limit_rpm=self.config.rate_limit_rpm,
            random_seed=self.config.sample_seed,
            parameters={
                "variant": variant,
                "sample_sizes": dict(self.config.sample_sizes or {}),
            },
        )

        # Emit run started event
        self._emit_event(
            "run_started",
            run_id,
            {
                "benchmark": benchmark.name,
                "adapter": adapter.name,
                "metrics": list(metric_names),
                "variant": variant,
            },
        )

        # Load test cases
        if self.config.sample_sizes:
            test_cases = benchmark.sample(
                self.config.sample_sizes,
                seed=self.config.sample_seed,
                variant=variant,
            )
        else:
            test_cases = benchmark.load(variant=variant)

        if self.config.verbose:
            print(f"Loaded {len(test_cases)} test cases from {benchmark.name}")

        # Get metrics
        metrics = []
        for name in metric_names:
            metric = self.get_metric(name)
            if metric is None:
                raise ValueError(f"Unknown metric: {name}. Register it first.")
            metrics.append(metric)

        # Run evaluation
        results = await self.executor.run_cases(
            adapter=adapter,
            test_cases=test_cases,
            metrics=metrics,
            resources=resources,
            run_id=run_id,
            on_progress=self.config.progress_callback,
        )

        finished_at = datetime.utcnow()

        # Compute aggregates
        run_result = self._aggregate_results(
            run_id=run_id,
            run_spec=run_spec,
            results=results,
            started_at=started_at,
            finished_at=finished_at,
        )

        # Emit run completed event
        self._emit_event(
            "run_completed",
            run_id,
            {
                "total": run_result.total_cases,
                "passed": run_result.passed_cases,
                "failed": run_result.failed_cases,
                "errors": run_result.error_cases,
                "pass_rate": run_result.pass_rate,
                "duration_seconds": (finished_at - started_at).total_seconds(),
            },
        )

        return run_result

    async def run_single(
        self,
        adapter: MemorySystemAdapter,
        test_case: TestCase,
        metric_names: Sequence[str],
        *,
        resources: Optional[Mapping[str, Any]] = None,
    ) -> EvalResult:
        """
        Run evaluation on a single test case.

        Useful for debugging and interactive exploration.
        """
        resources = resources or {}
        run_id = f"single_{uuid.uuid4().hex[:6]}"

        metrics = []
        for name in metric_names:
            metric = self.get_metric(name)
            if metric is None:
                raise ValueError(f"Unknown metric: {name}")
            metrics.append(metric)

        return await self.executor.run_test_case(
            adapter=adapter,
            test_case=test_case,
            metrics=metrics,
            resources=resources,
            run_id=run_id,
        )

    def _aggregate_results(
        self,
        run_id: str,
        run_spec: RunSpec,
        results: Sequence[EvalResult],
        started_at: datetime,
        finished_at: datetime,
    ) -> RunResult:
        """Compute aggregate statistics from individual results."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        errors = sum(1 for r in results if r.query_result.error)
        failed = total - passed - errors

        # Group by question type
        by_qtype: dict[str, dict[str, Any]] = {}
        for r in results:
            qtype = r.question_type or "unknown"
            if qtype not in by_qtype:
                by_qtype[qtype] = {"total": 0, "passed": 0}
            by_qtype[qtype]["total"] += 1
            if r.passed:
                by_qtype[qtype]["passed"] += 1

        # Calculate pass rates per type
        for qtype in by_qtype:
            total_t = by_qtype[qtype]["total"]
            passed_t = by_qtype[qtype]["passed"]
            by_qtype[qtype]["pass_rate"] = passed_t / total_t if total_t > 0 else 0.0

        # Group by metric
        by_metric: dict[str, dict[str, Any]] = {}
        for r in results:
            for mr in r.metric_results:
                if mr.metric not in by_metric:
                    by_metric[mr.metric] = {"total": 0, "passed": 0, "scores": []}
                by_metric[mr.metric]["total"] += 1
                if mr.passed:
                    by_metric[mr.metric]["passed"] += 1
                by_metric[mr.metric]["scores"].append(mr.score)

        # Calculate pass rates and averages per metric
        for mname in by_metric:
            total_m = by_metric[mname]["total"]
            passed_m = by_metric[mname]["passed"]
            scores = by_metric[mname]["scores"]
            by_metric[mname]["pass_rate"] = passed_m / total_m if total_m > 0 else 0.0
            by_metric[mname]["avg_score"] = sum(scores) / len(scores) if scores else 0.0
            del by_metric[mname]["scores"]  # Don't include raw scores in summary

        return RunResult(
            run_id=run_id,
            run_spec=run_spec,
            started_at=started_at,
            finished_at=finished_at,
            results=results,
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            error_cases=errors,
            by_question_type=by_qtype,
            by_metric=by_metric,
        )

    def _emit_event(
        self, event_type: str, run_id: str, payload: Mapping[str, Any]
    ) -> None:
        """Emit an event to the event store if configured."""
        if self.event_store is None:
            return

        event = Event(
            type=event_type,
            occurred_at=datetime.utcnow(),
            run_id=run_id,
            payload=payload,
        )
        self.event_store.append(event)


# === Convenience Functions ===


def create_default_engine(
    *,
    concurrency: int = 10,
    rate_limit_rpm: Optional[int] = None,
    verbose: bool = False,
) -> Engine:
    """
    Create an engine with default metrics registered.

    Includes all built-in metrics from the metrics module.
    """
    from ..metrics import (
        BinaryExactMatch,
        OptionExtractor,
        ContainsAnswer,
        ContextPrecision,
        ContextRecall,
        LLMBinaryJudge,
        AbstentionAccuracy,
        SemanticSimilarity,
    )

    engine = Engine(
        config=EngineConfig(
            concurrency=concurrency,
            rate_limit_rpm=rate_limit_rpm,
            verbose=verbose,
        )
    )

    # Register all built-in metrics
    engine.register_metrics(
        [
            BinaryExactMatch(),
            OptionExtractor(),
            ContainsAnswer(),
            ContextPrecision(),
            ContextRecall(),
            LLMBinaryJudge(),
            AbstentionAccuracy(),
            SemanticSimilarity(),
        ]
    )

    return engine


async def quick_eval(
    benchmark_name: str,
    adapter: MemorySystemAdapter,
    *,
    sample_per_type: int = 5,
    seed: int = 42,
    variant: Optional[str] = None,
    resources: Optional[Mapping[str, Any]] = None,
    verbose: bool = True,
) -> RunResult:
    """
    Quick evaluation for development and debugging.

    Samples a few cases per question type for fast iteration.

    Usage:
        result = await quick_eval(
            "personamem",
            my_adapter,
            sample_per_type=3,
            resources={"llm": client},
        )
        print(result.summary())
    """
    from ..benchmarks import get_registry

    registry = get_registry()
    benchmark = registry.get(benchmark_name)
    if benchmark is None:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    # Get question types from a sample load
    all_cases = benchmark.load(variant=variant)
    qtypes = set(tc.question_type for tc in all_cases if tc.question_type)

    # Build sample sizes
    sample_sizes = {qtype: sample_per_type for qtype in qtypes}

    engine = create_default_engine(concurrency=5, verbose=verbose)
    engine.config.sample_sizes = sample_sizes
    engine.config.sample_seed = seed

    # Use default metrics for the benchmark
    metric_names = benchmark.default_metrics()

    return await engine.run(
        benchmark=benchmark,
        adapter=adapter,
        metric_names=metric_names,
        resources=resources or {},
        variant=variant,
    )
