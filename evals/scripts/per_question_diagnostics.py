#!/usr/bin/env python3
"""
Build per-question diagnostics by merging:
- detailed_results_* (strategy/backend run results)
- reference dataset (oracle JSON)
- ingestion manifest (to map question_id -> user_id)
- optionally, graph_audit.json (merge per-user graph stats)

Outputs:
- evals/results/per_question_diagnostics.json (one record per question)

This helps correlate correctness proxies with graph structure.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


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


def load_json(p: str):
    with open(p, "r") as f:
        return json.load(f)


def normalize(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def contains_answer(gold: str, hyp: str) -> bool:
    g = normalize(gold)
    h = normalize(hyp)
    if not g or not h:
        return False
    return g in h or h in g


def matches_failure_pattern(h: str) -> bool:
    hh = normalize(h)
    return any(re.search(p, hh) for p in FAILURE_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to detailed_results_*.json")
    ap.add_argument("--references", required=True, help="Path to longmemeval_oracle.json")
    ap.add_argument("--manifest", required=True, help="Path to ingest_manifest_*.json")
    ap.add_argument("--graph_audit", required=False, help="Optional path to graph_audit.json")
    ap.add_argument("--out", default="evals/results/per_question_diagnostics.json", help="Output JSON path")
    args = ap.parse_args()

    results: List[Dict] = load_json(args.results)
    refs: List[Dict] = load_json(args.references)
    manifest: Dict[str, Dict] = load_json(args.manifest)
    audit_by_user: Dict[str, Dict] = {}
    if args.graph_audit and Path(args.graph_audit).exists():
        audit = load_json(args.graph_audit)
        for u in audit.get("users", []):
            if u.get("user_id"):
                audit_by_user[u["user_id"]] = u

    ref_by_qid = {r["question_id"]: r for r in refs}

    diag = []
    for r in results:
        qid = r.get("question_id")
        hyp = r.get("hypothesis", "")
        rr = ref_by_qid.get(qid, {})
        gold = rr.get("answer", "")
        qtype = rr.get("question_type", "unknown")
        qtext = rr.get("question", "")
        contains = contains_answer(gold, hyp)
        fail = matches_failure_pattern(hyp)

        # map to user_id
        user_id = manifest.get(qid, {}).get("user_id")
        stats = audit_by_user.get(user_id, {}) if user_id else {}

        row = {
            "question_id": qid,
            "user_id": user_id,
            "question_type": qtype,
            "question": qtext,
            "gold": gold,
            "hypothesis": hyp,
            "retrieval_time": r.get("retrieval_time"),
            "total_time": r.get("total_time"),
            "contains_gold": contains,
            "failure_phrase": fail,
            # ingestion fields
            "sessions_processed": manifest.get(qid, {}).get("sessions_processed"),
            "total_turns": manifest.get(qid, {}).get("total_turns"),
            # graph stats if present
            "nodes": stats.get("nodes"),
            "relationships": stats.get("relationships"),
            "avg_degree": stats.get("avg_degree"),
            "max_degree": stats.get("max_degree"),
            "orphan_ratio": stats.get("orphan_ratio"),
            "with_embeddings": stats.get("with_embeddings"),
            "type_counts": stats.get("types"),
        }
        diag.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"Wrote per-question diagnostics to {args.out}")


if __name__ == "__main__":
    main()

