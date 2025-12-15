# Persona Hybrid Memory: Analysis Log

This is a living document tracking investigations, hypotheses, and experiments to improve the graph+vector hybrid vs pure vector recall on LongMemEval.

## 2025-08-30 – Initial Pass

What I ran/used:
- Re-ran `evals/scripts/analyze_results.py` on full run artifacts (detailed_results, manifest, oracle refs). Confirmed `evals/results/analysis_summary.txt`.
- Added scripts:
  - `evals/scripts/graph_audit.py` – DB-backed audit for node/rel/type distributions, embedding coverage, orphan ratios, degree stats.
  - `evals/scripts/per_question_diagnostics.py` – merges results+refs+manifest (+ optional graph audit) into per-question diagnostics for correlation.
  - `evals/scripts/neo4j_probe.py` – quick utility for vector index status, counts, type distributions.

Key observations from current run (string-containment proxy):
- Good: single-session-user (~0.586) and knowledge-update (~0.551).
- Underperformers: multi-session (~0.265), temporal-reasoning (~0.263), single-session-assistant (~0.182), single-session-preference (0.000).
- “Retrieval time” recorded is end-to-end RAG call latency (~39–43s mean) not pure retrieval.

Likely root causes:
- Indexing quality: Node extraction prompt only returns `name`/`type`; no structured `properties` (dates, counts, entities). This limits precise recall (esp. temporal, counts, preferences).
- Type coverage: Preference and assistant-derived facts likely sparse or not reliably extracted/linked; graph context may miss these entirely.
- Seed selection: Vector seeds are generic; graph traversal adds neighbors, but without strong semantics/ranking, context stays weak/noisy.
- One-size retrieval: No task-specific retrieval logic (temporal filtering; preference-only focus; assistant vs user source distinctions).

Immediate next steps:
- Run `graph_audit.py` and `per_question_diagnostics.py` to quantify type distributions, orphan ratios, and correlate with misses (esp. ‘single-session-preference’ and ‘single-session-assistant’).
- Add a DB query to count `(exists n.embedding)` coverage; confirm vector-ready coverage.
- If access allows, instrument retrieval to log initial vector seeds + traversed nodes for failure cases.

Design hypotheses to test:
1) Structured nodes: Extend extraction to include `properties` (e.g., `date`, `count`, `entity`, `location`, `source=User|Assistant`) and persist them; embed `name + canonicalized properties`.
2) Temporal anchoring: Attach session timestamps to nodes and filter/sort retrieval by proximity to `(date: …)` in queries.
3) Task-aware retrieval: Detect question_type → route:
   - preference: restrict to `type=Preference` + neighbors; heavier weight on preference relationships.
   - assistant: include `source=Assistant` nodes.
   - temporal: filter to events around question_date and use path scoring.
4) Path/ranking: Score by path types and centrality; demote orphan or generic nodes; promote chains with strong relations.
5) Relationship ontology: Normalize relation labels (e.g., HAS_VALUE, ATTENDED, BOUGHT, LIKES) to improve traversal semantics.

Planned experiments (scoped, incremental):
- E1: Offline audit with `graph_audit.py` + `per_question_diagnostics.py` to quantify coverage and mismatches per task.
- E2: Structured extraction (v2 prompt + properties) implemented. Next: re-run a small batch and compare per-task gains.
- E3: Add temporal filters using existing session dates; route temporal questions accordingly.

Artifacts to maintain:
- `evals/results/graph_audit.json`
- `evals/results/per_question_diagnostics.json`

Open questions:
- Do we have significant `(type='Preference')` nodes today? If not, preference tasks will flatline.
- How often are embeddings missing? Any index gaps?
- Are assistant-attributed facts indexed and retrievable?

Update (Implementation):
- Added properties-aware node extraction (prompt + parser) and richer embedding text (name+type+key properties).
- Retrieval routing heuristics (preference filter, temporal scoring; seed score usage) with ranked context and JSONL logging.
