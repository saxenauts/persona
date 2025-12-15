# LongMemEval Results Diagnosis

## Overview
- Overall accuracy: 44.38% (task-averaged 47.73%) from `evals/results/evaluation_hybrid_hybrid.json`.
- Weak tasks: temporal-reasoning (30.83%), multi-session (33.33%).
- Pipeline was successful (ingest: 12,918s for 499 instances; answers: 489s for 498 items).

## Evidence From This Run
- Heuristic analysis (`evals/scripts/analyze_results.py`) over `detailed_results_hybrid_hybrid.json` + `longmemeval_oracle.json`:
  - String-containment proxy accuracy (approximate):
    - temporal-reasoning: 35/133 ≈ 0.263; multi-session: 35/132 ≈ 0.265; single-session-assistant: 10/55 ≈ 0.182; single-session-user: 41/70 ≈ 0.586; knowledge-update: 43/78 ≈ 0.551; single-session-preference: 0/30 (rubric-style, proxy unsuitable).
  - Failure-phrase matches (e.g., “not enough information”, “cannot determine”): temporal-reasoning 42; multi-session 31; single-session-assistant 11.
  - Mean retrieval time (s): temporal-reasoning 42.58; multi-session 41.34; others 38–42.
- Sample failures (truncated):
  - multi-session • 6d550036: “How many projects have I led…?” → hyp: “no specific information… I am unable to determine…”.
  - temporal-reasoning • 0bb5a684: “How many days before… workshop?” → verbose timeline, lacks explicit dating edges.
  - single-session-assistant • c4f10528: asked for a specific restaurant → answer devolves into generic suggestions.

## Root Causes (Code-Backed)
- Vector-seeded retrieval only: `GraphContextRetriever.get_rich_context()` → `text_similarity_search()`; two-hop crawl over nearest nodes. No temporal or aggregation operators.
- No explicit temporal structure: dates embedded in node text; missing `OCCURRED_ON`, `BEFORE/AFTER`, or duration properties. See `persona/core/graph_ops.py` and `persona/core/rag_interface.py`.
- Session isolation: ingestion treats each session independently (see `evals/longmemeval/ingest.py`), with no cross-session entity resolution or dedup.
- Preference rubric mismatch: rubric-style “answers” are not string-contained in free-form model outputs, leading to mis-evaluation under naive checks and to generic responses.

## Recommendations
- Temporal grounding (minimal structure, schemaless-friendly):
  - Add Date nodes and edges: Event —OCCURRED_ON→ Date; Event —BEFORE/AFTER→ Event (when inferable).
  - Extract timestamps during node construction; persist `timestamp` and `session_id` in node properties.
- Cross-session linking and aggregation:
  - Lightweight coref linking: new mention —REFERS_TO→ canonical entity; merge/rollup properties across sessions.
  - Query operators: ORDER BY time; COUNT/SUM over linked mentions; earliest/latest selectors.
- Retrieval upgrades:
  - Hybrid seed = {vector top-k} ∪ {time-window filter around `question_date`}.
  - Task-aware retrieval: for temporal questions, bias edges on temporal relations; for “how many”, aggregate over linked entities.
- Instrumentation for future evals:
  - Log top-k retrieved nodes + edges per question_id; include time-window filters and crawl paths in results.
  - Save per-question evaluation labels to enable targeted error buckets.

## How to Reproduce Analysis
- Run: `python evals/scripts/analyze_results.py --results evals/results/detailed_results_hybrid_hybrid.json --references evals/data/longmemeval_oracle.json --manifest evals/results/ingest_manifest_hybrid.json --out evals/results/analysis_summary.txt`
- Inspect: `evals/results/analysis_summary.txt` for task breakdowns, failure samples, and retrieval-time stats.

