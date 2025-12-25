#!/usr/bin/env python3
"""
Graphiti vs Persona Analysis System

Comprehensive analysis script that:
1. Extracts ALL logged metrics from Graphiti eval run
2. Compares against Persona baseline by question type
3. Performs root cause analysis of failures AND successes
4. Outputs a human-readable markdown report

Usage:
    python evals/scripts/analyze_graphiti_run.py \
        --run-id run_20251224_174630 \
        --baseline-id run_20251223_100407 \
        --output evals/analysis/GRAPHITI_ANALYSIS_REPORT.md
"""

import argparse
import json
import statistics
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class IngestionMetrics:
    duration_ms: float = 0.0
    sessions_count: int = 0
    nodes_created: int = 0
    relationships_created: int = 0
    embeddings_generated: int = 0
    episodes: int = 0
    psyche: int = 0
    goals: int = 0
    events: int = 0
    errors: List[str] = field(default_factory=list)
    extract_ms: Optional[float] = None
    embed_ms: Optional[float] = None
    persist_ms: Optional[float] = None


@dataclass
class RetrievalMetrics:
    query: str = ""
    duration_ms: float = 0.0
    vector_top_k: int = 0
    vector_duration_ms: float = 0.0
    seed_count: int = 0
    seed_scores: List[float] = field(default_factory=list)
    max_hops: int = 0
    nodes_visited: int = 0
    relationships_traversed: int = 0
    final_nodes_count: int = 0
    graph_duration_ms: float = 0.0
    context_size_tokens: int = 0
    retrieved_context: str = ""


@dataclass
class GenerationMetrics:
    duration_ms: float = 0.0
    model: str = ""
    temperature: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    answer: str = ""


@dataclass
class EvaluationMetrics:
    gold_answer: str = ""
    correct: Optional[bool] = None
    judge_response: str = ""
    judge_model: str = ""
    score_type: str = ""


@dataclass
class QuestionMetrics:
    question_id: str = ""
    timestamp: str = ""
    benchmark: str = ""
    question_type: str = ""
    question: str = ""
    ingestion: IngestionMetrics = field(default_factory=IngestionMetrics)
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    evaluation: EvaluationMetrics = field(default_factory=EvaluationMetrics)


@dataclass
class TypeStats:
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_ingest_ms: float = 0.0
    avg_retrieval_ms: float = 0.0
    avg_generation_ms: float = 0.0
    avg_context_tokens: float = 0.0
    avg_nodes_visited: float = 0.0
    avg_seed_count: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    empty_context_rate: float = 0.0
    timeout_rate: float = 0.0


# ============================================================================
# METRIC EXTRACTION
# ============================================================================

def extract_metrics(log: Dict[str, Any]) -> QuestionMetrics:
    """Extract all metrics from a single question log entry."""

    # Ingestion
    ing = log.get("ingestion", {})
    memories = ing.get("memories_created", {})
    ingestion = IngestionMetrics(
        duration_ms=ing.get("duration_ms", 0) or 0,
        sessions_count=ing.get("sessions_count", 0) or 0,
        nodes_created=ing.get("nodes_created", 0) or 0,
        relationships_created=ing.get("relationships_created", 0) or 0,
        embeddings_generated=ing.get("embeddings_generated", 0) or 0,
        episodes=memories.get("episodes", 0) or 0,
        psyche=memories.get("psyche", 0) or 0,
        goals=memories.get("goals", 0) or 0,
        events=memories.get("events", 0) or 0,
        errors=ing.get("errors", []) or [],
        extract_ms=ing.get("extract_ms"),
        embed_ms=ing.get("embed_ms"),
        persist_ms=ing.get("persist_ms"),
    )

    # Retrieval
    ret = log.get("retrieval", {})
    vs = ret.get("vector_search", {})
    gt = ret.get("graph_traversal", {})
    seeds = vs.get("seeds", []) or []
    retrieval = RetrievalMetrics(
        query=ret.get("query", "") or "",
        duration_ms=ret.get("duration_ms", 0) or 0,
        vector_top_k=vs.get("top_k", 0) or 0,
        vector_duration_ms=vs.get("duration_ms", 0) or 0,
        seed_count=len(seeds),
        seed_scores=[s.get("score", 0) for s in seeds if s.get("score")],
        max_hops=gt.get("max_hops", 0) or 0,
        nodes_visited=gt.get("nodes_visited", 0) or 0,
        relationships_traversed=gt.get("relationships_traversed", 0) or 0,
        final_nodes_count=len(gt.get("final_ranked_nodes", []) or []),
        graph_duration_ms=gt.get("duration_ms", 0) or 0,
        context_size_tokens=ret.get("context_size_tokens", 0) or 0,
        retrieved_context=ret.get("retrieved_context", "") or "",
    )

    # Generation
    gen = log.get("generation", {})
    generation = GenerationMetrics(
        duration_ms=gen.get("duration_ms", 0) or 0,
        model=gen.get("model", "") or "",
        temperature=gen.get("temperature", 0) or 0,
        prompt_tokens=gen.get("prompt_tokens", 0) or 0,
        completion_tokens=gen.get("completion_tokens", 0) or 0,
        answer=gen.get("answer", "") or "",
    )

    # Evaluation
    ev = log.get("evaluation", {})
    evaluation = EvaluationMetrics(
        gold_answer=ev.get("gold_answer", "") or "",
        correct=ev.get("correct"),
        judge_response=ev.get("judge_response", "") or "",
        judge_model=ev.get("judge_model", "") or "",
        score_type=ev.get("score_type", "") or "",
    )

    return QuestionMetrics(
        question_id=log.get("question_id", "") or "",
        timestamp=log.get("timestamp", "") or "",
        benchmark=log.get("benchmark", "") or "",
        question_type=log.get("question_type", "") or "",
        question=log.get("question", "") or "",
        ingestion=ingestion,
        retrieval=retrieval,
        generation=generation,
        evaluation=evaluation,
    )


def load_run_logs(run_id: str, results_dir: Path) -> List[QuestionMetrics]:
    """Load all question logs from a run."""
    log_file = results_dir / run_id / "deep_logs.jsonl"
    if not log_file.exists():
        raise FileNotFoundError(f"Run logs not found at {log_file}")

    logs = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                logs.append(extract_metrics(data))

    return logs


# ============================================================================
# AGGREGATION
# ============================================================================

def safe_mean(values: List[float]) -> float:
    """Calculate mean, returning 0 for empty lists."""
    return statistics.mean(values) if values else 0.0


def aggregate_by_type(logs: List[QuestionMetrics]) -> Dict[str, TypeStats]:
    """Aggregate metrics by question type."""
    by_type: Dict[str, List[QuestionMetrics]] = defaultdict(list)

    for log in logs:
        by_type[log.question_type].append(log)

    stats = {}
    for qtype, type_logs in by_type.items():
        total = len(type_logs)
        correct = sum(1 for l in type_logs if l.evaluation.correct is True)

        ingest_times = [l.ingestion.duration_ms for l in type_logs if l.ingestion.duration_ms > 0]
        retrieval_times = [l.retrieval.duration_ms for l in type_logs if l.retrieval.duration_ms > 0]
        generation_times = [l.generation.duration_ms for l in type_logs if l.generation.duration_ms > 0]
        context_tokens = [l.retrieval.context_size_tokens for l in type_logs]
        nodes_visited = [l.retrieval.nodes_visited for l in type_logs]
        seed_counts = [l.retrieval.seed_count for l in type_logs]
        prompt_tokens = [l.generation.prompt_tokens for l in type_logs if l.generation.prompt_tokens > 0]
        completion_tokens = [l.generation.completion_tokens for l in type_logs if l.generation.completion_tokens > 0]

        empty_context = sum(1 for l in type_logs if l.retrieval.context_size_tokens < 100)
        timeouts = sum(1 for l in type_logs if l.ingestion.duration_ms > 300000)  # 5 min

        stats[qtype] = TypeStats(
            total=total,
            correct=correct,
            accuracy=correct / total if total > 0 else 0.0,
            avg_ingest_ms=safe_mean(ingest_times),
            avg_retrieval_ms=safe_mean(retrieval_times),
            avg_generation_ms=safe_mean(generation_times),
            avg_context_tokens=safe_mean(context_tokens),
            avg_nodes_visited=safe_mean(nodes_visited),
            avg_seed_count=safe_mean(seed_counts),
            avg_prompt_tokens=safe_mean(prompt_tokens),
            avg_completion_tokens=safe_mean(completion_tokens),
            empty_context_rate=empty_context / total if total > 0 else 0.0,
            timeout_rate=timeouts / total if total > 0 else 0.0,
        )

    return stats


# ============================================================================
# FAILURE PATTERN DETECTION
# ============================================================================

@dataclass
class FailurePattern:
    pattern_name: str
    description: str
    count: int
    examples: List[QuestionMetrics]


def detect_failure_patterns(logs: List[QuestionMetrics]) -> List[FailurePattern]:
    """Detect common failure patterns in failed questions."""
    failed = [l for l in logs if l.evaluation.correct is False]

    patterns = []

    # Pattern 1: Empty retrieval (no graph traversal)
    empty_traversal = [l for l in failed if l.retrieval.nodes_visited == 0]
    if empty_traversal:
        patterns.append(FailurePattern(
            pattern_name="Empty Graph Traversal",
            description="No nodes visited during graph traversal - retrieval returned nothing",
            count=len(empty_traversal),
            examples=empty_traversal[:3],
        ))

    # Pattern 2: Low context (< 100 tokens)
    low_context = [l for l in failed if l.retrieval.context_size_tokens < 100]
    if low_context:
        patterns.append(FailurePattern(
            pattern_name="Insufficient Context",
            description="Context size < 100 tokens - not enough information retrieved",
            count=len(low_context),
            examples=low_context[:3],
        ))

    # Pattern 3: No seeds found
    no_seeds = [l for l in failed if l.retrieval.seed_count == 0]
    if no_seeds:
        patterns.append(FailurePattern(
            pattern_name="No Vector Seeds",
            description="Vector search returned 0 seeds - semantic matching failed",
            count=len(no_seeds),
            examples=no_seeds[:3],
        ))

    # Pattern 4: Low similarity scores
    low_similarity = [l for l in failed
                      if l.retrieval.seed_scores and max(l.retrieval.seed_scores) < 0.5]
    if low_similarity:
        patterns.append(FailurePattern(
            pattern_name="Low Similarity Scores",
            description="Best seed score < 0.5 - poor semantic match quality",
            count=len(low_similarity),
            examples=low_similarity[:3],
        ))

    # Pattern 5: Model abstention
    abstention_phrases = ["i don't have", "no information", "cannot determine",
                          "not mentioned", "no context", "unable to"]
    abstained = [l for l in failed
                 if any(p in l.generation.answer.lower() for p in abstention_phrases)]
    if abstained:
        patterns.append(FailurePattern(
            pattern_name="Model Abstention",
            description="Model declined to answer - said 'I don't have information'",
            count=len(abstained),
            examples=abstained[:3],
        ))

    # Pattern 6: Ingestion timeout (> 5 min)
    timeouts = [l for l in failed if l.ingestion.duration_ms > 300000]
    if timeouts:
        patterns.append(FailurePattern(
            pattern_name="Ingestion Timeout",
            description="Ingestion took > 5 minutes - likely timeout or hang",
            count=len(timeouts),
            examples=timeouts[:3],
        ))

    # Pattern 7: Ingestion errors
    with_errors = [l for l in failed if len(l.ingestion.errors) > 0]
    if with_errors:
        patterns.append(FailurePattern(
            pattern_name="Ingestion Errors",
            description="Explicit errors during ingestion phase",
            count=len(with_errors),
            examples=with_errors[:3],
        ))

    # Pattern 8: Zero nodes created
    zero_nodes = [l for l in failed if l.ingestion.nodes_created == 0]
    if zero_nodes:
        patterns.append(FailurePattern(
            pattern_name="Zero Nodes Created",
            description="Ingestion created 0 nodes - nothing was indexed",
            count=len(zero_nodes),
            examples=zero_nodes[:3],
        ))

    # Sort by count (most common first)
    patterns.sort(key=lambda p: p.count, reverse=True)
    return patterns


def detect_success_patterns(logs: List[QuestionMetrics]) -> Dict[str, Any]:
    """Analyze what correlated with successful answers."""
    passed = [l for l in logs if l.evaluation.correct is True]
    failed = [l for l in logs if l.evaluation.correct is False]

    if not passed or not failed:
        return {}

    # Compare averages between passed and failed
    def compare_metric(name: str, extractor) -> Dict[str, float]:
        passed_vals = [extractor(l) for l in passed]
        failed_vals = [extractor(l) for l in failed]
        return {
            "passed_avg": safe_mean(passed_vals),
            "failed_avg": safe_mean(failed_vals),
            "delta": safe_mean(passed_vals) - safe_mean(failed_vals),
        }

    return {
        "context_tokens": compare_metric("context_tokens",
                                         lambda l: l.retrieval.context_size_tokens),
        "nodes_visited": compare_metric("nodes_visited",
                                        lambda l: l.retrieval.nodes_visited),
        "seed_count": compare_metric("seed_count",
                                     lambda l: l.retrieval.seed_count),
        "max_seed_score": compare_metric("max_seed_score",
                                         lambda l: max(l.retrieval.seed_scores) if l.retrieval.seed_scores else 0),
        "ingest_time": compare_metric("ingest_time",
                                      lambda l: l.ingestion.duration_ms),
        "nodes_created": compare_metric("nodes_created",
                                        lambda l: l.ingestion.nodes_created),
    }


# ============================================================================
# COMPARISON
# ============================================================================

def compare_runs(graphiti_stats: Dict[str, TypeStats],
                 persona_stats: Dict[str, TypeStats]) -> Dict[str, Dict[str, float]]:
    """Compare Graphiti run against Persona baseline."""
    comparison = {}

    all_types = set(graphiti_stats.keys()) | set(persona_stats.keys())

    for qtype in all_types:
        g = graphiti_stats.get(qtype)
        p = persona_stats.get(qtype)

        if g and p:
            comparison[qtype] = {
                "graphiti_accuracy": g.accuracy,
                "persona_accuracy": p.accuracy,
                "accuracy_delta": g.accuracy - p.accuracy,
                "graphiti_retrieval_ms": g.avg_retrieval_ms,
                "persona_retrieval_ms": p.avg_retrieval_ms,
                "retrieval_delta_ms": g.avg_retrieval_ms - p.avg_retrieval_ms,
            }
        elif g:
            comparison[qtype] = {
                "graphiti_accuracy": g.accuracy,
                "persona_accuracy": 0.0,
                "accuracy_delta": g.accuracy,
                "note": "Not in Persona baseline",
            }
        else:
            comparison[qtype] = {
                "graphiti_accuracy": 0.0,
                "persona_accuracy": p.accuracy if p else 0.0,
                "accuracy_delta": -(p.accuracy if p else 0.0),
                "note": "Not in Graphiti run",
            }

    return comparison


# ============================================================================
# REPORT GENERATION
# ============================================================================

def format_duration(ms: float) -> str:
    """Format milliseconds as human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"


def generate_report(
    run_id: str,
    logs: List[QuestionMetrics],
    graphiti_stats: Dict[str, TypeStats],
    persona_stats: Dict[str, TypeStats],
    comparison: Dict[str, Dict[str, float]],
    failure_patterns: List[FailurePattern],
    success_patterns: Dict[str, Any],
) -> str:
    """Generate comprehensive markdown report."""

    total = len(logs)
    correct = sum(1 for l in logs if l.evaluation.correct is True)
    failed = sum(1 for l in logs if l.evaluation.correct is False)
    skipped = sum(1 for l in logs if l.evaluation.correct is None)
    accuracy = correct / total if total > 0 else 0.0

    # Calculate overall timing
    total_ingest = sum(l.ingestion.duration_ms for l in logs)
    avg_ingest = total_ingest / len(logs) if logs else 0
    avg_retrieval = safe_mean([l.retrieval.duration_ms for l in logs])
    avg_generation = safe_mean([l.generation.duration_ms for l in logs])

    # Persona overall
    persona_total = sum(s.total for s in persona_stats.values())
    persona_correct = sum(s.correct for s in persona_stats.values())
    persona_accuracy = persona_correct / persona_total if persona_total > 0 else 0.0

    report = []

    # Header
    report.append(f"# Graphiti Eval Analysis Report")
    report.append(f"")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Run ID:** `{run_id}`")
    report.append(f"")
    report.append(f"---")
    report.append(f"")

    # Executive Summary
    report.append(f"## Executive Summary")
    report.append(f"")
    report.append(f"| Metric | Graphiti | Persona Baseline | Delta |")
    report.append(f"|--------|----------|------------------|-------|")
    report.append(f"| **Overall Accuracy** | {accuracy:.1%} ({correct}/{total}) | {persona_accuracy:.1%} ({persona_correct}/{persona_total}) | {accuracy - persona_accuracy:+.1%} |")
    report.append(f"| **Avg Ingest Time** | {format_duration(avg_ingest)} | - | - |")
    report.append(f"| **Avg Retrieval Time** | {format_duration(avg_retrieval)} | - | - |")
    report.append(f"| **Avg Generation Time** | {format_duration(avg_generation)} | - | - |")
    report.append(f"")

    # Key findings
    wins = [(t, c) for t, c in comparison.items() if c.get("accuracy_delta", 0) > 0.05]
    losses = [(t, c) for t, c in comparison.items() if c.get("accuracy_delta", 0) < -0.05]

    report.append(f"### Key Findings")
    report.append(f"")
    if wins:
        report.append(f"**Graphiti Wins ({len(wins)} types):**")
        for t, c in sorted(wins, key=lambda x: -x[1]["accuracy_delta"]):
            report.append(f"- {t}: +{c['accuracy_delta']:.1%}")
    if losses:
        report.append(f"")
        report.append(f"**Graphiti Gaps ({len(losses)} types):**")
        for t, c in sorted(losses, key=lambda x: x[1]["accuracy_delta"]):
            report.append(f"- {t}: {c['accuracy_delta']:.1%}")
    report.append(f"")
    report.append(f"---")
    report.append(f"")

    # Performance by Type
    report.append(f"## 1. Performance by Question Type")
    report.append(f"")
    report.append(f"| Question Type | Graphiti Acc | Persona Acc | Delta | Graphiti N | Avg Retr | Avg Ctx Tokens |")
    report.append(f"|--------------|-------------|-------------|-------|-----------|----------|----------------|")

    for qtype in sorted(graphiti_stats.keys()):
        g = graphiti_stats[qtype]
        p = persona_stats.get(qtype)
        p_acc = f"{p.accuracy:.1%}" if p else "N/A"
        delta = comparison.get(qtype, {}).get("accuracy_delta", 0)
        delta_str = f"{delta:+.1%}" if p else "N/A"
        report.append(f"| {qtype} | {g.accuracy:.1%} ({g.correct}/{g.total}) | {p_acc} | {delta_str} | {g.total} | {format_duration(g.avg_retrieval_ms)} | {g.avg_context_tokens:.0f} |")

    report.append(f"")
    report.append(f"---")
    report.append(f"")

    # Failure Patterns
    report.append(f"## 2. Failure Pattern Analysis")
    report.append(f"")
    report.append(f"Total failures: {failed}/{total} ({failed/total:.1%})")
    report.append(f"")

    if failure_patterns:
        report.append(f"### Detected Patterns")
        report.append(f"")
        report.append(f"| Pattern | Count | % of Failures | Description |")
        report.append(f"|---------|-------|---------------|-------------|")
        for p in failure_patterns:
            pct = p.count / failed if failed > 0 else 0
            report.append(f"| {p.pattern_name} | {p.count} | {pct:.1%} | {p.description} |")
        report.append(f"")

        # Deep dive into top patterns
        report.append(f"### Pattern Deep Dives")
        report.append(f"")
        for pattern in failure_patterns[:3]:  # Top 3 patterns
            report.append(f"#### {pattern.pattern_name} ({pattern.count} occurrences)")
            report.append(f"")
            report.append(f"{pattern.description}")
            report.append(f"")
            report.append(f"**Example Questions:**")
            for i, ex in enumerate(pattern.examples[:2], 1):
                report.append(f"")
                report.append(f"**Example {i}:** `{ex.question_id}`")
                report.append(f"- Type: `{ex.question_type}`")
                report.append(f"- Question: {ex.question[:100]}...")
                report.append(f"- Gold Answer: {ex.evaluation.gold_answer}")
                report.append(f"- Generated: {ex.generation.answer[:100]}...")
                report.append(f"- Context Tokens: {ex.retrieval.context_size_tokens}")
                report.append(f"- Nodes Visited: {ex.retrieval.nodes_visited}")
                report.append(f"- Seed Count: {ex.retrieval.seed_count}")
            report.append(f"")

    report.append(f"---")
    report.append(f"")

    # Success Patterns
    report.append(f"## 3. Success Pattern Analysis")
    report.append(f"")
    report.append(f"What correlated with correct answers?")
    report.append(f"")

    if success_patterns:
        report.append(f"| Metric | Passed Avg | Failed Avg | Delta | Insight |")
        report.append(f"|--------|------------|------------|-------|---------|")
        for metric, data in success_patterns.items():
            delta = data["delta"]
            insight = "Higher is better" if delta > 0 else "Lower is better" if delta < 0 else "No difference"
            if metric in ["ingest_time"]:
                report.append(f"| {metric} | {format_duration(data['passed_avg'])} | {format_duration(data['failed_avg'])} | {format_duration(abs(delta))} | {insight} |")
            else:
                report.append(f"| {metric} | {data['passed_avg']:.1f} | {data['failed_avg']:.1f} | {delta:+.1f} | {insight} |")

    report.append(f"")
    report.append(f"---")
    report.append(f"")

    # Retrieval Analysis
    report.append(f"## 4. Retrieval Analysis")
    report.append(f"")

    all_context = [l.retrieval.context_size_tokens for l in logs]
    all_seeds = [l.retrieval.seed_count for l in logs]
    all_nodes = [l.retrieval.nodes_visited for l in logs]
    empty_context = sum(1 for l in logs if l.retrieval.context_size_tokens < 100)

    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Avg Context Size | {safe_mean(all_context):.0f} tokens |")
    report.append(f"| Avg Seeds Found | {safe_mean(all_seeds):.1f} |")
    report.append(f"| Avg Nodes Visited | {safe_mean(all_nodes):.1f} |")
    report.append(f"| Empty Context Rate | {empty_context}/{total} ({empty_context/total:.1%}) |")
    report.append(f"")

    report.append(f"### Retrieval by Question Type")
    report.append(f"")
    report.append(f"| Type | Avg Context | Avg Seeds | Avg Nodes | Empty Rate |")
    report.append(f"|------|-------------|-----------|-----------|------------|")
    for qtype, stats in sorted(graphiti_stats.items()):
        report.append(f"| {qtype} | {stats.avg_context_tokens:.0f} | {stats.avg_seed_count:.1f} | {stats.avg_nodes_visited:.1f} | {stats.empty_context_rate:.1%} |")

    report.append(f"")
    report.append(f"---")
    report.append(f"")

    # Latency Analysis
    report.append(f"## 5. Latency Analysis")
    report.append(f"")
    report.append(f"| Phase | Avg Time | Min | Max |")
    report.append(f"|-------|----------|-----|-----|")

    ingest_times = [l.ingestion.duration_ms for l in logs if l.ingestion.duration_ms > 0]
    retrieval_times = [l.retrieval.duration_ms for l in logs if l.retrieval.duration_ms > 0]
    generation_times = [l.generation.duration_ms for l in logs if l.generation.duration_ms > 0]

    if ingest_times:
        report.append(f"| Ingestion | {format_duration(safe_mean(ingest_times))} | {format_duration(min(ingest_times))} | {format_duration(max(ingest_times))} |")
    if retrieval_times:
        report.append(f"| Retrieval | {format_duration(safe_mean(retrieval_times))} | {format_duration(min(retrieval_times))} | {format_duration(max(retrieval_times))} |")
    if generation_times:
        report.append(f"| Generation | {format_duration(safe_mean(generation_times))} | {format_duration(min(generation_times))} | {format_duration(max(generation_times))} |")

    report.append(f"")

    # Timeout analysis
    timeouts = [l for l in logs if l.ingestion.duration_ms > 300000]
    if timeouts:
        report.append(f"### Timeout Analysis")
        report.append(f"")
        report.append(f"**{len(timeouts)} questions had ingestion > 5 minutes:**")
        report.append(f"")
        for t in timeouts[:5]:
            report.append(f"- `{t.question_id}` ({t.question_type}): {format_duration(t.ingestion.duration_ms)}")

    report.append(f"")
    report.append(f"---")
    report.append(f"")

    # Sample Failures Deep Dive
    report.append(f"## 6. Sample Failure Deep Dives")
    report.append(f"")

    failed_logs = [l for l in logs if l.evaluation.correct is False]
    # Get diverse sample (different types)
    seen_types = set()
    samples = []
    for l in failed_logs:
        if l.question_type not in seen_types and len(samples) < 5:
            samples.append(l)
            seen_types.add(l.question_type)

    for i, sample in enumerate(samples, 1):
        report.append(f"### Failure {i}: {sample.question_type}")
        report.append(f"")
        report.append(f"**Question ID:** `{sample.question_id}`")
        report.append(f"")
        report.append(f"**Question:** {sample.question}")
        report.append(f"")
        report.append(f"**Gold Answer:** {sample.evaluation.gold_answer}")
        report.append(f"")
        report.append(f"**Generated Answer:** {sample.generation.answer[:200]}...")
        report.append(f"")
        report.append(f"**Metrics:**")
        report.append(f"```")
        report.append(f"Ingestion:")
        report.append(f"  Duration: {format_duration(sample.ingestion.duration_ms)}")
        report.append(f"  Nodes Created: {sample.ingestion.nodes_created}")
        report.append(f"  Relationships: {sample.ingestion.relationships_created}")
        report.append(f"")
        report.append(f"Retrieval:")
        report.append(f"  Duration: {format_duration(sample.retrieval.duration_ms)}")
        report.append(f"  Seeds: {sample.retrieval.seed_count}")
        report.append(f"  Nodes Visited: {sample.retrieval.nodes_visited}")
        report.append(f"  Context Tokens: {sample.retrieval.context_size_tokens}")
        if sample.retrieval.seed_scores:
            report.append(f"  Best Seed Score: {max(sample.retrieval.seed_scores):.3f}")
        report.append(f"")
        report.append(f"Generation:")
        report.append(f"  Model: {sample.generation.model}")
        report.append(f"  Prompt Tokens: {sample.generation.prompt_tokens}")
        report.append(f"  Completion Tokens: {sample.generation.completion_tokens}")
        report.append(f"```")
        report.append(f"")

        # Retrieved context preview
        if sample.retrieval.retrieved_context:
            ctx_preview = sample.retrieval.retrieved_context[:500].replace("\n", " ")
            report.append(f"**Context Preview:** {ctx_preview}...")
        report.append(f"")

    report.append(f"---")
    report.append(f"")

    # Recommendations
    report.append(f"## 7. Recommendations")
    report.append(f"")

    # Generate recommendations based on patterns
    if any(p.pattern_name == "Empty Graph Traversal" for p in failure_patterns):
        report.append(f"### Improve Graph Traversal")
        report.append(f"- Many failures had 0 nodes visited during graph traversal")
        report.append(f"- Check if vector seeds are being used as starting points for BFS")
        report.append(f"- Consider increasing max_hops or adjusting traversal strategy")
        report.append(f"")

    if any(p.pattern_name == "Insufficient Context" for p in failure_patterns):
        report.append(f"### Increase Context Retrieval")
        report.append(f"- Many failures had < 100 tokens of context")
        report.append(f"- Increase top_k for vector search")
        report.append(f"- Consider retrieving more edges/relationships")
        report.append(f"")

    if any(p.pattern_name == "Model Abstention" for p in failure_patterns):
        report.append(f"### Reduce Model Abstention")
        report.append(f"- Model frequently says 'I don't have information'")
        report.append(f"- Check if relevant context is being retrieved but not used")
        report.append(f"- Consider adjusting generation prompt to be less conservative")
        report.append(f"")

    if any(p.pattern_name == "Ingestion Timeout" for p in failure_patterns):
        report.append(f"### Address Ingestion Timeouts")
        report.append(f"- Some questions timed out during ingestion (> 5 min)")
        report.append(f"- Check for deadlocks or slow LLM calls during extraction")
        report.append(f"- Consider batching or parallel ingestion")
        report.append(f"")

    # Type-specific recommendations
    for qtype, comp in comparison.items():
        if comp.get("accuracy_delta", 0) < -0.1:  # 10%+ worse than Persona
            report.append(f"### Improve {qtype}")
            report.append(f"- Graphiti is {-comp['accuracy_delta']:.1%} worse than Persona on this type")
            gs = graphiti_stats.get(qtype)
            if gs:
                report.append(f"- Current stats: {gs.correct}/{gs.total} correct, avg {gs.avg_context_tokens:.0f} context tokens")
            report.append(f"")

    report.append(f"---")
    report.append(f"")
    report.append(f"*Report generated by analyze_graphiti_run.py*")

    return "\n".join(report)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze Graphiti eval run vs Persona baseline")
    parser.add_argument("--run-id", required=True, help="Graphiti run ID (e.g., run_20251224_174630)")
    parser.add_argument("--baseline-id", default="run_20251223_100407",
                        help="Persona baseline run ID")
    parser.add_argument("--output", default=None,
                        help="Output markdown file (default: analysis/GRAPHITI_ANALYSIS_REPORT.md)")
    parser.add_argument("--results-dir", default="evals/results",
                        help="Directory containing run results")
    parser.add_argument("--export-json", action="store_true",
                        help="Also export raw data as JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else Path("evals/analysis/GRAPHITI_ANALYSIS_REPORT.md")

    print(f"Loading Graphiti run: {args.run_id}")
    graphiti_logs = load_run_logs(args.run_id, results_dir)
    print(f"  Loaded {len(graphiti_logs)} questions")

    print(f"Loading Persona baseline: {args.baseline_id}")
    try:
        persona_logs = load_run_logs(args.baseline_id, results_dir)
        print(f"  Loaded {len(persona_logs)} questions")
    except FileNotFoundError:
        print(f"  Baseline not found, proceeding without comparison")
        persona_logs = []

    print("Aggregating metrics by type...")
    graphiti_stats = aggregate_by_type(graphiti_logs)
    persona_stats = aggregate_by_type(persona_logs)

    print("Comparing runs...")
    comparison = compare_runs(graphiti_stats, persona_stats)

    print("Detecting failure patterns...")
    failure_patterns = detect_failure_patterns(graphiti_logs)

    print("Analyzing success patterns...")
    success_patterns = detect_success_patterns(graphiti_logs)

    print("Generating report...")
    report = generate_report(
        run_id=args.run_id,
        logs=graphiti_logs,
        graphiti_stats=graphiti_stats,
        persona_stats=persona_stats,
        comparison=comparison,
        failure_patterns=failure_patterns,
        success_patterns=success_patterns,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {output_path}")

    if args.export_json:
        json_path = output_path.with_suffix(".json")
        export_data = {
            "run_id": args.run_id,
            "baseline_id": args.baseline_id,
            "graphiti_stats": {k: vars(v) for k, v in graphiti_stats.items()},
            "persona_stats": {k: vars(v) for k, v in persona_stats.items()},
            "comparison": comparison,
            "failure_patterns": [
                {"name": p.pattern_name, "count": p.count, "description": p.description}
                for p in failure_patterns
            ],
            "success_patterns": success_patterns,
        }
        with open(json_path, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"JSON data saved to: {json_path}")

    # Print quick summary
    total = len(graphiti_logs)
    correct = sum(1 for l in graphiti_logs if l.evaluation.correct is True)
    print(f"\n{'='*60}")
    print(f"QUICK SUMMARY")
    print(f"{'='*60}")
    print(f"Graphiti Accuracy: {correct}/{total} ({correct/total:.1%})")
    if persona_logs:
        p_total = len(persona_logs)
        p_correct = sum(1 for l in persona_logs if l.evaluation.correct is True)
        print(f"Persona Accuracy:  {p_correct}/{p_total} ({p_correct/p_total:.1%})")
        print(f"Delta: {(correct/total) - (p_correct/p_total):+.1%}")
    print(f"\nTop failure patterns:")
    for p in failure_patterns[:3]:
        print(f"  - {p.pattern_name}: {p.count} occurrences")


if __name__ == "__main__":
    main()
