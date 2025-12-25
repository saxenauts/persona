#!/usr/bin/env python3
"""
Failure Inspector - Interactive tool for browsing eval failures.

Browse, filter, and analyze failed questions from an eval run.

Usage:
    python evals/scripts/failure_inspector.py --run-id run_20251224_174630

Commands:
    list [type]     - List failures (optionally filter by question type)
    show <id>       - Show full details for a question
    types           - Show failure count by question type
    search <text>   - Search failures by question/answer text
    context <id>    - Show retrieved context for a question
    compare <id>    - Show side-by-side gold vs generated answer
    export [type]   - Export failures to JSON (optionally filter by type)
    patterns        - Show detected failure patterns
    quit            - Exit
"""

import argparse
import json
import readline  # For command history
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


# ============================================================================
# DATA LOADING
# ============================================================================

def load_logs(run_id: str, results_dir: Path) -> List[Dict[str, Any]]:
    """Load all logs from a run."""
    log_file = results_dir / run_id / "deep_logs.jsonl"
    if not log_file.exists():
        raise FileNotFoundError(f"Run logs not found at {log_file}")

    logs = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


def get_failures(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get only failed questions."""
    return [l for l in logs if l.get("evaluation", {}).get("correct") is False]


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def truncate(text: str, length: int = 80) -> str:
    """Truncate text to length."""
    text = text.replace("\n", " ")
    if len(text) <= length:
        return text
    return text[:length-3] + "..."


def format_duration(ms: float) -> str:
    """Format milliseconds as human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"


def print_table(headers: List[str], rows: List[List[str]], max_widths: Optional[List[int]] = None):
    """Print a formatted table."""
    if not rows:
        print("No data")
        return

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Apply max widths if specified
    if max_widths:
        widths = [min(w, m) for w, m in zip(widths, max_widths)]

    # Print header
    header_row = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))

    # Print rows
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cell_str = str(cell)
            if i < len(widths) and len(cell_str) > widths[i]:
                cell_str = cell_str[:widths[i]-3] + "..."
            cells.append(cell_str.ljust(widths[i]) if i < len(widths) else cell_str)
        print(" | ".join(cells))


# ============================================================================
# COMMANDS
# ============================================================================

def cmd_list(failures: List[Dict], args: List[str]):
    """List failures, optionally filtered by type."""
    filter_type = args[0] if args else None

    filtered = failures
    if filter_type:
        filtered = [f for f in failures if f.get("question_type") == filter_type]

    if not filtered:
        print(f"No failures found" + (f" for type '{filter_type}'" if filter_type else ""))
        return

    rows = []
    for f in filtered[:50]:  # Limit to 50
        qid = f.get("question_id", "?")[:12]
        qtype = f.get("question_type", "?")[:20]
        question = truncate(f.get("question", ""), 50)
        ctx_tokens = f.get("retrieval", {}).get("context_size_tokens", 0)
        rows.append([qid, qtype, str(ctx_tokens), question])

    print(f"\nShowing {len(rows)}/{len(filtered)} failures" +
          (f" (type: {filter_type})" if filter_type else ""))
    print()
    print_table(
        ["ID", "Type", "Ctx", "Question"],
        rows,
        max_widths=[12, 20, 6, 60]
    )


def cmd_types(failures: List[Dict], args: List[str]):
    """Show failure count by question type."""
    by_type = defaultdict(int)
    for f in failures:
        by_type[f.get("question_type", "unknown")] += 1

    rows = [[t, str(c)] for t, c in sorted(by_type.items(), key=lambda x: -x[1])]
    print(f"\nFailures by Question Type ({len(failures)} total)")
    print()
    print_table(["Type", "Count"], rows)


def cmd_show(failures: List[Dict], args: List[str]):
    """Show full details for a question."""
    if not args:
        print("Usage: show <question_id>")
        return

    qid = args[0]
    matches = [f for f in failures if f.get("question_id", "").startswith(qid)]

    if not matches:
        print(f"No failure found with ID starting with '{qid}'")
        return
    if len(matches) > 1:
        print(f"Multiple matches found. Be more specific:")
        for m in matches[:5]:
            print(f"  - {m.get('question_id')}")
        return

    f = matches[0]

    print(f"\n{'='*70}")
    print(f"QUESTION: {f.get('question_id')}")
    print(f"{'='*70}")
    print()
    print(f"Type: {f.get('question_type')}")
    print(f"Benchmark: {f.get('benchmark')}")
    print(f"Timestamp: {f.get('timestamp')}")
    print()

    print("QUESTION:")
    print(f"  {f.get('question')}")
    print()

    print("GOLD ANSWER:")
    print(f"  {f.get('evaluation', {}).get('gold_answer')}")
    print()

    print("GENERATED ANSWER:")
    print(f"  {f.get('generation', {}).get('answer')}")
    print()

    print("INGESTION METRICS:")
    ing = f.get("ingestion", {})
    print(f"  Duration: {format_duration(ing.get('duration_ms', 0))}")
    print(f"  Sessions: {ing.get('sessions_count', 0)}")
    print(f"  Nodes Created: {ing.get('nodes_created', 0)}")
    print(f"  Relationships: {ing.get('relationships_created', 0)}")
    print(f"  Embeddings: {ing.get('embeddings_generated', 0)}")
    memories = ing.get("memories_created", {})
    print(f"  Memories: episodes={memories.get('episodes', 0)}, psyche={memories.get('psyche', 0)}, goals={memories.get('goals', 0)}")
    if ing.get("errors"):
        print(f"  ERRORS: {ing.get('errors')}")
    print()

    print("RETRIEVAL METRICS:")
    ret = f.get("retrieval", {})
    vs = ret.get("vector_search", {})
    gt = ret.get("graph_traversal", {})
    print(f"  Duration: {format_duration(ret.get('duration_ms', 0))}")
    print(f"  Context Tokens: {ret.get('context_size_tokens', 0)}")
    print(f"  Vector Search: top_k={vs.get('top_k', 0)}, seeds={len(vs.get('seeds', []))}")
    if vs.get("seeds"):
        scores = [s.get("score", 0) for s in vs["seeds"] if s.get("score")]
        if scores:
            print(f"    Seed Scores: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}")
    print(f"  Graph Traversal: hops={gt.get('max_hops', 0)}, nodes={gt.get('nodes_visited', 0)}, rels={gt.get('relationships_traversed', 0)}")
    print()

    print("GENERATION METRICS:")
    gen = f.get("generation", {})
    print(f"  Model: {gen.get('model')}")
    print(f"  Duration: {format_duration(gen.get('duration_ms', 0))}")
    print(f"  Prompt Tokens: {gen.get('prompt_tokens', 0)}")
    print(f"  Completion Tokens: {gen.get('completion_tokens', 0)}")
    print()

    print("JUDGE:")
    ev = f.get("evaluation", {})
    print(f"  Model: {ev.get('judge_model')}")
    print(f"  Response: {ev.get('judge_response')}")


def cmd_context(failures: List[Dict], args: List[str]):
    """Show retrieved context for a question."""
    if not args:
        print("Usage: context <question_id>")
        return

    qid = args[0]
    matches = [f for f in failures if f.get("question_id", "").startswith(qid)]

    if not matches:
        print(f"No failure found with ID starting with '{qid}'")
        return

    f = matches[0]
    ctx = f.get("retrieval", {}).get("retrieved_context", "")

    print(f"\n{'='*70}")
    print(f"RETRIEVED CONTEXT FOR: {f.get('question_id')}")
    print(f"{'='*70}")
    print()
    print(ctx if ctx else "(empty context)")


def cmd_compare(failures: List[Dict], args: List[str]):
    """Show side-by-side comparison of gold vs generated."""
    if not args:
        print("Usage: compare <question_id>")
        return

    qid = args[0]
    matches = [f for f in failures if f.get("question_id", "").startswith(qid)]

    if not matches:
        print(f"No failure found with ID starting with '{qid}'")
        return

    f = matches[0]

    print(f"\n{'='*70}")
    print(f"COMPARISON: {f.get('question_id')} ({f.get('question_type')})")
    print(f"{'='*70}")
    print()
    print("QUESTION:")
    print(f"  {f.get('question')}")
    print()
    print("-" * 70)
    print("GOLD ANSWER:")
    print(f.get("evaluation", {}).get("gold_answer", ""))
    print("-" * 70)
    print("GENERATED ANSWER:")
    print(f.get("generation", {}).get("answer", ""))
    print("-" * 70)


def cmd_search(failures: List[Dict], args: List[str]):
    """Search failures by question/answer text."""
    if not args:
        print("Usage: search <text>")
        return

    query = " ".join(args).lower()
    matches = []
    for f in failures:
        question = f.get("question", "").lower()
        answer = f.get("generation", {}).get("answer", "").lower()
        gold = f.get("evaluation", {}).get("gold_answer", "").lower()
        if query in question or query in answer or query in gold:
            matches.append(f)

    if not matches:
        print(f"No failures matching '{query}'")
        return

    rows = []
    for f in matches[:20]:
        qid = f.get("question_id", "?")[:12]
        qtype = f.get("question_type", "?")[:15]
        question = truncate(f.get("question", ""), 50)
        rows.append([qid, qtype, question])

    print(f"\nFound {len(matches)} matches for '{query}':")
    print()
    print_table(["ID", "Type", "Question"], rows, max_widths=[12, 15, 60])


def cmd_export(failures: List[Dict], args: List[str], run_id: str, output_dir: Path):
    """Export failures to JSON."""
    filter_type = args[0] if args else None

    filtered = failures
    if filter_type:
        filtered = [f for f in failures if f.get("question_type") == filter_type]

    filename = f"failures_{run_id}"
    if filter_type:
        filename += f"_{filter_type}"
    filename += ".json"

    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Exported {len(filtered)} failures to {output_path}")


def cmd_patterns(failures: List[Dict], args: List[str]):
    """Show detected failure patterns."""
    patterns = []

    # Empty traversal
    empty = [f for f in failures if f.get("retrieval", {}).get("graph_traversal", {}).get("nodes_visited", 0) == 0]
    if empty:
        patterns.append(("Empty Graph Traversal", len(empty), "0 nodes visited"))

    # Low context
    low_ctx = [f for f in failures if f.get("retrieval", {}).get("context_size_tokens", 0) < 100]
    if low_ctx:
        patterns.append(("Insufficient Context", len(low_ctx), "< 100 tokens"))

    # No seeds
    no_seeds = [f for f in failures if len(f.get("retrieval", {}).get("vector_search", {}).get("seeds", [])) == 0]
    if no_seeds:
        patterns.append(("No Vector Seeds", len(no_seeds), "0 seeds found"))

    # Model abstention
    abstain_phrases = ["i don't have", "no information", "cannot determine", "not mentioned"]
    abstained = [f for f in failures
                 if any(p in f.get("generation", {}).get("answer", "").lower() for p in abstain_phrases)]
    if abstained:
        patterns.append(("Model Abstention", len(abstained), "said 'no information'"))

    # Timeout
    timeouts = [f for f in failures if f.get("ingestion", {}).get("duration_ms", 0) > 300000]
    if timeouts:
        patterns.append(("Ingestion Timeout", len(timeouts), "> 5 min ingest"))

    # Zero nodes created
    zero_nodes = [f for f in failures if f.get("ingestion", {}).get("nodes_created", 0) == 0]
    if zero_nodes:
        patterns.append(("Zero Nodes Created", len(zero_nodes), "nothing indexed"))

    if not patterns:
        print("No patterns detected")
        return

    print(f"\n{'='*70}")
    print("FAILURE PATTERNS")
    print(f"{'='*70}")
    print()

    patterns.sort(key=lambda x: -x[1])
    rows = [[p[0], str(p[1]), f"{p[1]/len(failures)*100:.1f}%", p[2]] for p in patterns]
    print_table(["Pattern", "Count", "% of Failures", "Description"], rows)


# ============================================================================
# MAIN REPL
# ============================================================================

def run_repl(failures: List[Dict], run_id: str, results_dir: Path):
    """Run interactive REPL."""
    print(f"\n{'='*70}")
    print(f"FAILURE INSPECTOR - {run_id}")
    print(f"{'='*70}")
    print(f"Loaded {len(failures)} failures")
    print("Type 'help' for commands, 'quit' to exit")
    print()

    commands = {
        "list": lambda args: cmd_list(failures, args),
        "types": lambda args: cmd_types(failures, args),
        "show": lambda args: cmd_show(failures, args),
        "context": lambda args: cmd_context(failures, args),
        "compare": lambda args: cmd_compare(failures, args),
        "search": lambda args: cmd_search(failures, args),
        "patterns": lambda args: cmd_patterns(failures, args),
        "export": lambda args: cmd_export(failures, args, run_id, results_dir),
    }

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif cmd == "help":
            print(__doc__)
        elif cmd in commands:
            try:
                commands[cmd](args)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Unknown command: {cmd}. Type 'help' for available commands.")


def main():
    parser = argparse.ArgumentParser(description="Interactive failure inspector")
    parser.add_argument("--run-id", required=True, help="Run ID to inspect")
    parser.add_argument("--results-dir", default="evals/results",
                        help="Directory containing run results")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Just print summary and exit")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    logs = load_logs(args.run_id, results_dir)
    failures = get_failures(logs)

    print(f"Loaded {len(logs)} total questions, {len(failures)} failures")

    if args.non_interactive:
        cmd_types(failures, [])
        cmd_patterns(failures, [])
        return

    run_repl(failures, args.run_id, results_dir)


if __name__ == "__main__":
    main()
