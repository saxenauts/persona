#!/usr/bin/env python3
"""
Ablation Runner for PersonaMem Evaluation

Runs paired A/B experiments to measure causal impact of individual components.
Each question is evaluated twice: with component ON and with component OFF.
Uses McNemar's test for statistical significance on paired binary outcomes.

Usage:
    python scripts/ablation_runner.py --component memeplex --questions 20
    python scripts/ablation_runner.py --component session_close --questions 20 --dry-run
    python scripts/ablation_runner.py --component memeplex --questions 5 --seed 42

Toggles supported:
    - memeplex: EVAL_REFRESH_MEMEPLEX (world model index)
    - session_close: EVAL_CLOSE_SESSIONS (post-ingest consolidation)
    - entity_retrieval: EVAL_INCLUDE_ENTITIES (entity nodes in recall)
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MEMORY_EVALS_PATH = Path(__file__).parent.parent / "memory-evals"
sys.path.insert(0, str(MEMORY_EVALS_PATH))

SCIPY_AVAILABLE = False
try:
    import numpy as np
    from scipy import stats as scipy_stats

    SCIPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    scipy_stats = None  # type: ignore


COMPONENT_ENV_MAP = {
    "memeplex": "EVAL_REFRESH_MEMEPLEX",
    "session_close": "EVAL_CLOSE_SESSIONS",
    "entity_retrieval": "EVAL_INCLUDE_ENTITIES",
}


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""

    component: str
    questions_evaluated: int
    accuracy_with: float
    accuracy_without: float
    delta: float
    ci_95: Tuple[float, float]
    significant: bool
    p_value: float
    mcnemar_statistic: Optional[float]
    effect_details: Dict[str, Any]
    timestamp: str
    seed: int
    dry_run: bool


@dataclass
class QuestionResult:
    """Result for a single question in both conditions."""

    question_id: str
    question_type: str
    correct_with: Optional[bool]
    correct_without: Optional[bool]
    answer_with: str
    answer_without: str
    ingest_time_with_ms: float
    ingest_time_without_ms: float
    query_time_with_ms: float
    query_time_without_ms: float


def set_component(component: str, enabled: bool) -> None:
    """Set environment variable for component toggle."""
    env_var = COMPONENT_ENV_MAP.get(component)
    if not env_var:
        raise ValueError(
            f"Unknown component: {component}. Valid: {list(COMPONENT_ENV_MAP.keys())}"
        )
    os.environ[env_var] = "true" if enabled else "false"
    print(f"    {env_var}={'true' if enabled else 'false'}")


def mcnemar_test(b: int, c: int) -> Tuple[float, float]:
    """McNemar's test for paired binary outcomes."""
    if b + c == 0:
        return 0.0, 1.0

    if not SCIPY_AVAILABLE:
        statistic = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0.0
        p_value = 0.5 if b == c else (0.05 if abs(b - c) > 2 else 0.2)
        return float(statistic), p_value

    if b + c < 25:
        n = b + c
        k = min(b, c)
        p_value = 2 * scipy_stats.binom.cdf(k, n, 0.5)
        return float(n), min(1.0, p_value)
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - scipy_stats.chi2.cdf(statistic, df=1)
        return float(statistic), float(p_value)


def wilson_confidence_interval(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if total == 0:
        return (0.0, 0.0)

    if not SCIPY_AVAILABLE:
        import math

        z = 1.96
        p_hat = successes / total
        denominator = 1 + z**2 / total
        center = p_hat + z**2 / (2 * total)
        spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)
        lower = (center - spread) / denominator
        upper = (center + spread) / denominator
        return (max(0.0, float(lower)), min(1.0, float(upper)))

    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = p_hat + z**2 / (2 * total)
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)
    lower = (center - spread) / denominator
    upper = (center + spread) / denominator
    return (max(0.0, float(lower)), min(1.0, float(upper)))


def compute_delta_ci(
    results_on: List[bool], results_off: List[bool]
) -> Tuple[float, Tuple[float, float], float, Optional[float], bool]:
    """Compute accuracy delta with bootstrap confidence interval."""
    n = len(results_on)
    if n == 0:
        return 0.0, (0.0, 0.0), 1.0, None, False

    acc_on = sum(results_on) / n
    acc_off = sum(results_off) / n
    delta = acc_on - acc_off

    b = sum(1 for on, off in zip(results_on, results_off) if on and not off)
    c = sum(1 for on, off in zip(results_on, results_off) if not on and off)
    mcnemar_stat, p_value = mcnemar_test(b, c)

    n_bootstrap = 1000
    deltas = []
    for _ in range(n_bootstrap):
        indices = [random.randint(0, n - 1) for _ in range(n)]
        boot_on = [results_on[i] for i in indices]
        boot_off = [results_off[i] for i in indices]
        boot_delta = sum(boot_on) / n - sum(boot_off) / n
        deltas.append(boot_delta)

    deltas.sort()
    ci_lower = deltas[int(n_bootstrap * 0.025)]
    ci_upper = deltas[int(n_bootstrap * 0.975)]

    significant = p_value < 0.05 and not (ci_lower <= 0 <= ci_upper)
    return delta, (ci_lower, ci_upper), p_value, mcnemar_stat, significant


def evaluate_single_question(
    adapter,
    question,
    user_id: str,
    sessions: List[Dict],
    query_text: str,
    verbose: bool = True,
) -> Tuple[Optional[bool], str, float, float]:
    """
    Run evaluation for a single question.

    Returns: (correct, answer, ingest_time_ms, query_time_ms)
    """
    import re

    adapter.reset(user_id)

    start_ingest = time.time()
    adapter.add_sessions(user_id, sessions)
    ingest_time_ms = (time.time() - start_ingest) * 1000

    start_query = time.time()
    answer = adapter.query(user_id, query_text)
    query_time_ms = (time.time() - start_query) * 1000

    answer_lower = answer.lower().strip()
    cleaned = re.sub(r"^\s*(answer|option)\s*[:\-]*\s*", "", answer_lower).strip()

    extracted = None
    tokens = cleaned.split()
    if tokens:
        candidate = tokens[0].strip("().")
        if candidate in ["a", "b", "c", "d"]:
            extracted = candidate

    if not extracted:
        match = re.search(r"\b([abcd])\b", cleaned)
        if match:
            extracted = match.group(1)

    correct_answer = question.correct_answer.lower()
    correct = extracted == correct_answer if extracted else False

    if verbose:
        status = "?" if extracted is None else ("+" if correct else "x")
        print(f"      [{status}] {extracted or '?'} (expected: {correct_answer})")

    try:
        adapter.reset(user_id)
    except:
        pass

    return correct, answer, ingest_time_ms, query_time_ms


def prepare_personamem_sessions(question) -> List[Dict]:
    """Convert PersonaMem context to session format."""
    context = question.context
    if not context:
        return []

    lines = context.split("\n")
    sessions = []
    current_session = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("System:"):
            if current_session:
                sessions.append(
                    {"date": "unknown", "content": "\n".join(current_session)}
                )
                current_session = []
            current_session.append(line)
        else:
            current_session.append(line)

    if current_session:
        sessions.append({"date": "unknown", "content": "\n".join(current_session)})

    return sessions if sessions else [{"date": "unknown", "content": context}]


def format_personamem_query(question) -> str:
    """Format PersonaMem MCQ query."""
    letters = ["a", "b", "c", "d"]
    options_str = " ".join(
        f"({letter}) {question.options.get(letter, '')}"
        for letter in letters
        if question.options.get(letter)
    )
    return (
        f"Question: {question.question}\n"
        f"Options: {options_str}\n"
        "Answer with only the letter (a/b/c/d)."
    )


def run_ablation(
    component: str,
    questions: list,
    seed: int,
    dry_run: bool = False,
    verbose: bool = True,
) -> AblationResult:
    """
    Run paired ablation experiment for a component.

    Each question is evaluated twice: with component ON and OFF.
    """
    from mem_eval.adapters.persona_adapter import PersonaAdapter

    random.seed(seed)
    if SCIPY_AVAILABLE and np is not None:
        np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print(f"ABLATION: {component}")
    print(f"Questions: {len(questions)}")
    print(f"Seed: {seed}")
    print(f"Dry run: {dry_run}")
    print(f"{'=' * 60}")

    question_results: List[QuestionResult] = []
    results_on: List[bool] = []
    results_off: List[bool] = []

    for i, question in enumerate(questions):
        print(f"\n[{i + 1}/{len(questions)}] {question.question_type}")
        print(f"  Q: {question.question[:60]}...")

        if dry_run:
            correct_on = random.random() > 0.4
            correct_off = random.random() > 0.5
            results_on.append(correct_on)
            results_off.append(correct_off)
            question_results.append(
                QuestionResult(
                    question_id=question.question_id,
                    question_type=question.question_type,
                    correct_with=correct_on,
                    correct_without=correct_off,
                    answer_with="simulated",
                    answer_without="simulated",
                    ingest_time_with_ms=100.0,
                    ingest_time_without_ms=100.0,
                    query_time_with_ms=500.0,
                    query_time_without_ms=500.0,
                )
            )
            print(
                f"    [DRY-RUN] Simulated: ON={'+' if correct_on else 'x'}, OFF={'+' if correct_off else 'x'}"
            )
            continue

        sessions = prepare_personamem_sessions(question)
        query_text = format_personamem_query(question)
        user_id_base = f"ablation_{question.question_id}_{int(time.time())}"

        print(f"  [ON] {component}=true")
        set_component(component, enabled=True)
        adapter_on = PersonaAdapter()

        correct_on, answer_on, ingest_on, query_on = evaluate_single_question(
            adapter=adapter_on,
            question=question,
            user_id=f"{user_id_base}_on",
            sessions=sessions,
            query_text=query_text,
            verbose=verbose,
        )

        print(f"  [OFF] {component}=false")
        set_component(component, enabled=False)
        adapter_off = PersonaAdapter()

        correct_off, answer_off, ingest_off, query_off = evaluate_single_question(
            adapter=adapter_off,
            question=question,
            user_id=f"{user_id_base}_off",
            sessions=sessions,
            query_text=query_text,
            verbose=verbose,
        )

        results_on.append(correct_on if correct_on is not None else False)
        results_off.append(correct_off if correct_off is not None else False)

        question_results.append(
            QuestionResult(
                question_id=question.question_id,
                question_type=question.question_type,
                correct_with=correct_on,
                correct_without=correct_off,
                answer_with=answer_on,
                answer_without=answer_off,
                ingest_time_with_ms=ingest_on,
                ingest_time_without_ms=ingest_off,
                query_time_with_ms=query_on,
                query_time_without_ms=query_off,
            )
        )

    n = len(results_on)
    acc_on = sum(results_on) / n if n > 0 else 0.0
    acc_off = sum(results_off) / n if n > 0 else 0.0

    delta, ci_95, p_value, mcnemar_stat, significant = compute_delta_ci(
        results_on, results_off
    )

    both_correct = sum(1 for on, off in zip(results_on, results_off) if on and off)
    both_wrong = sum(
        1 for on, off in zip(results_on, results_off) if not on and not off
    )
    on_better = sum(1 for on, off in zip(results_on, results_off) if on and not off)
    off_better = sum(1 for on, off in zip(results_on, results_off) if not on and off)

    effect_details = {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "on_better": on_better,
        "off_better": off_better,
        "concordant": both_correct + both_wrong,
        "discordant": on_better + off_better,
        "question_results": [asdict(qr) for qr in question_results],
    }

    result = AblationResult(
        component=component,
        questions_evaluated=n,
        accuracy_with=acc_on,
        accuracy_without=acc_off,
        delta=delta,
        ci_95=ci_95,
        significant=significant,
        p_value=p_value,
        mcnemar_statistic=mcnemar_stat,
        effect_details=effect_details,
        timestamp=datetime.now().isoformat(),
        seed=seed,
        dry_run=dry_run,
    )

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {component}")
    print(f"{'=' * 60}")
    print(f"  Accuracy WITH {component}:    {acc_on:.1%} ({sum(results_on)}/{n})")
    print(f"  Accuracy WITHOUT {component}: {acc_off:.1%} ({sum(results_off)}/{n})")
    print(f"  Delta: {delta:+.1%}")
    print(f"  95% CI: [{ci_95[0]:+.1%}, {ci_95[1]:+.1%}]")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant: {'YES' if significant else 'NO'}")
    print(f"\n  Pair breakdown:")
    print(f"    Both correct:  {both_correct}")
    print(f"    Both wrong:    {both_wrong}")
    print(f"    ON better:     {on_better}")
    print(f"    OFF better:    {off_better}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study for PersonaMem evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --component memeplex --questions 20
  %(prog)s --component session_close --questions 10 --dry-run
  %(prog)s --component memeplex --questions 5 --seed 42

Components:
  memeplex        - World model index (EVAL_REFRESH_MEMEPLEX)
  session_close   - Post-ingest consolidation (EVAL_CLOSE_SESSIONS)
  entity_retrieval - Entity nodes in recall (EVAL_INCLUDE_ENTITIES)
        """,
    )

    parser.add_argument(
        "--component",
        type=str,
        required=True,
        choices=list(COMPONENT_ENV_MAP.keys()),
        help="Component to ablate",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=20,
        help="Number of questions to evaluate (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate evaluation without calling APIs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ablation_results.json",
        help="Output file path (default: ablation_results.json)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="32k",
        choices=["32k", "128k", "1M"],
        help="PersonaMem variant (default: 32k)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to PersonaMem data directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()
    verbose = args.verbose and not args.quiet

    print(f"Ablation Runner - PersonaMem Evaluation")
    print(f"Component: {args.component}")
    print(f"Questions: {args.questions}")
    print(f"Seed: {args.seed}")
    print(f"Dry run: {args.dry_run}")

    data_dir = args.data_dir
    candidates = [
        MEMORY_EVALS_PATH / "evals" / "data" / "personamem",
        MEMORY_EVALS_PATH / "data" / "personamem",
        Path("evals/data/personamem"),
        Path("data/personamem"),
    ]
    if not data_dir:
        for candidate in candidates:
            if candidate.exists():
                data_dir = str(candidate)
                break

    if not data_dir:
        print(f"\nERROR: PersonaMem data not found. Tried:")
        for c in candidates:
            print(f"  - {c}")
        print("\nDownload with: python memory-evals/scripts/download_personamem.py")
        sys.exit(1)

    print(f"Data dir: {data_dir}")

    from mem_eval.loaders.personamem_loader import PersonaMemLoader

    loader = PersonaMemLoader(data_dir=data_dir, variant=args.variant)
    all_questions = loader.load()

    random.seed(args.seed)
    if args.questions >= len(all_questions):
        questions = all_questions
    else:
        questions = random.sample(all_questions, args.questions)

    print(f"Sampled {len(questions)} questions from {len(all_questions)} total")

    result = run_ablation(
        component=args.component,
        questions=questions,
        seed=args.seed,
        dry_run=args.dry_run,
        verbose=verbose,
    )

    output_path = Path(args.output)
    output_data = asdict(result)
    output_data["ci_95"] = list(output_data["ci_95"])

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
