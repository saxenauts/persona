Experiments Overview

This folder will track reproducible experiment configs and outcomes as we iterate on the hybrid memory system.

Suggested workflow
- Define scope: dataset slice, ingestion variant (e.g., v2 structured nodes), retrieval parameters.
- Capture config as JSON (what changed, seeds, limits, date, commit SHA).
- Run pipeline (ingest → answer → evaluate), plus audits:
  - graph audit: `python evals/scripts/graph_audit.py --manifest ...`
  - per-question diagnostics: `python evals/scripts/per_question_diagnostics.py --graph_audit ...`
- Store outputs under a timestamped directory.

Example structure
- 2025-08-30_exp01_structured_nodes/
  - config.json
  - results/
    - ingest_manifest_*.json
    - hypotheses_*.jsonl
    - detailed_results_*.json
    - evaluation_*.json
  - diagnostics/
    - graph_audit.json
    - per_question_diagnostics.json
  - notes.md

Tip
- Retrieval logs append to `evals/results/retrieval_logs.jsonl` for quick inspection.

