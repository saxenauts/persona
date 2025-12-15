#!/usr/bin/env python3
"""
Analyze LongMemEval results produced under evals/results/.

Outputs:
- Summary metrics by task type (string-match heuristic)
- Failure phrase counts and examples
- Retrieval time distribution by task type
- A compact CSV of per-question diagnostics (optional)

Usage:
  python evals/scripts/analyze_results.py \
    --results evals/results/detailed_results_hybrid_hybrid.json \
    --references evals/data/longmemeval_oracle.json \
    --manifest evals/results/ingest_manifest_hybrid.json \
    --out evals/results/analysis_summary.txt

Note: This script uses only local files; it does not call external APIs.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


FAILURE_PATTERNS = [
    r"not (enough|sufficient) (info|information)",
    r"insufficient (info|information)",
    r"cannot (determine|tell|infer)",
    r"no (specific|clear) (date|information|mention)",
    r"not (provided|specified)",
    r"unknown",
    r"I (don'?t|do not) (know|have)",
    r"context does not include",
]


def load_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> List[dict]:
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def normalize(s) -> str:
    # Coerce to string for safety
    if s is None:
        return ""
    s = str(s)
    return re.sub(r"\s+", " ", s).strip().lower()


def contains_answer(gold: str, hyp: str) -> bool:
    """Simple string containment heuristic (lowercased)."""
    g = normalize(gold)
    h = normalize(hyp)
    if not g or not h:
        return False
    return g in h or h in g


def matches_failure_pattern(hyp: str) -> bool:
    h = normalize(hyp)
    return any(re.search(p, h) for p in FAILURE_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to detailed_results_*.json")
    ap.add_argument("--references", required=True, help="Path to longmemeval_oracle.json")
    ap.add_argument("--manifest", required=False, help="Path to ingest_manifest_*.json")
    ap.add_argument("--out", required=False, help="Where to write the text summary")
    args = ap.parse_args()

    results: List[Dict] = load_json(args.results)
    refs: List[Dict] = load_json(args.references)
    manifest: Dict[str, Dict] = load_json(args.manifest) if args.manifest else {}

    qid2ref = {r["question_id"]: r for r in refs}

    # Accumulators
    task_counts = Counter()
    task_containment_hits = Counter()
    task_failphrases = Counter()
    rt_by_task: Dict[str, List[float]] = defaultdict(list)
    examples_fail: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    examples_miss_containment: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    missing_in_refs = 0
    for r in results:
        qid = r.get("question_id")
        hyp = r.get("hypothesis", "")
        rt = float(r.get("retrieval_time", 0.0))
        ref = qid2ref.get(qid)
        if not ref:
            missing_in_refs += 1
            continue

        task = ref.get("question_type", "unknown")
        gold = ref.get("answer", "")
        question = ref.get("question", "")

        task_counts[task] += 1
        if contains_answer(gold, hyp):
            task_containment_hits[task] += 1
        if matches_failure_pattern(hyp):
            task_failphrases[task] += 1
            if len(examples_fail[task]) < 5:
                examples_fail[task].append((qid, question, hyp[:300]))
        else:
            # If not obviously a failure phrase but also not containing the gold string,
            # record a sample of potential misses.
            if not contains_answer(gold, hyp) and len(examples_miss_containment[task]) < 5:
                examples_miss_containment[task].append((qid, question, f"gold={gold} | hyp={hyp[:180]}"))

        rt_by_task[task].append(rt)

    # Prepare summary text
    lines: List[str] = []
    lines.append("Analysis Summary (heuristics; not official metrics)")
    lines.append("")
    total = sum(task_counts.values())
    lines.append(f"Total questions analyzed: {total} (missing_in_refs={missing_in_refs})")
    lines.append("")

    lines.append("By Task Type (string containment as proxy for correctness):")
    for t in sorted(task_counts.keys()):
        c = task_counts[t]
        hits = task_containment_hits[t]
        frac = hits / c if c else 0.0
        failp = task_failphrases[t]
        lines.append(f"- {t}: {hits}/{c} ≈ {frac:.3f} | failure-phrases={failp}")
    lines.append("")

    lines.append("Retrieval time (s) by task (mean):")
    for t in sorted(rt_by_task.keys()):
        arr = rt_by_task[t]
        if arr:
            mean_rt = sum(arr)/len(arr)
            lines.append(f"- {t}: {mean_rt:.2f}s over n={len(arr)}")
    lines.append("")

    if examples_fail:
        lines.append("Examples: failure phrases detected")
        for t in sorted(examples_fail.keys()):
            lines.append(f"- {t}:")
            for qid, q, hyp_snip in examples_fail[t]:
                lines.append(f"  • {qid}: {q}")
                lines.append(f"    hyp: {hyp_snip}")
        lines.append("")

    if examples_miss_containment:
        lines.append("Examples: no failure phrase, but no string containment")
        for t in sorted(examples_miss_containment.keys()):
            lines.append(f"- {t}:")
            for qid, q, pair in examples_miss_containment[t]:
                lines.append(f"  • {qid}: {q}")
                lines.append(f"    {pair}")
        lines.append("")

    out_text = "\n".join(lines)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(out_text)
    else:
        print(out_text)


if __name__ == "__main__":
    main()
