# Benchmark Truth Table (Audit-Grade)

**Date**: 2026-01-31  
**Scope**: This file is generated from on-disk JSON artifacts only. If a claim is not here, it is not release-grade.

---

## Truth Table (On-Disk)

| Benchmark | Accuracy | Samples | Seeds | Notes | Source |
|-----------|----------|---------|-------|-------|--------|
| PersonaMem (subset) | 65.3% | 150 | 42, 123, 456 | recall_user_shared_facts only | ../memory-evals/results/competitor_full/persona_personamem/persona_personamem_summary.json |
| PersonaMem (full) | 66.2% | 589 | 42 | full benchmark, single seed | ../memory-evals/results/persona_personamem_full/persona_personamem_seed42.json |
| BEAM (10 abilities) | 69.0% | 100 | 1 | baseline, event_ordering=0% | ../memory-evals/results/beam_100q_minimal_prompt/run_20260126_113003/final_results.json |

Deferred benchmark note: one previously published benchmark line is intentionally removed from v0.3 public release claim surfaces.

# Exploratory Competitor Snapshot

| System | Accuracy | Samples | Notes | Source |
|--------|----------|---------|-------|--------|
| Mem0 | 61.9% | 147 | Mixed schema across seeds | ../memory-evals/results/competitor_full/mem0_personamem/ |

---

## Interpretation Rules

1. **Audit-grade** = backed by JSON artifacts in `../memory-evals/results/` with clear seeds, sample counts, and run folders.
2. **Exploratory** = partial runs, mismatched schemas, or subset-only benchmarks. Not marketing-eligible.
3. **Adjusted accuracy** claims are **not allowed** unless a committed audit artifact exists (e.g., `gold_audit.jsonl`).
4. **BEAM** scores must specify whether the run covers **10 abilities** or a subset. Only full-ability runs are headline-worthy.

---

## Required Artifacts for Any Claim

- `summary.json` or per-seed JSON with total_questions + correct
- `deep_logs.jsonl` for traceability
- `manifest.json` (benchmark, model, dataset version, git commit, seeds, samples)
- A stable run directory name

---

## Repro Commands (Canonical)

```bash
# PersonaMem subset (150Q)
PYTHONUNBUFFERED=1 .venv/bin/mem-eval run   --benchmark personamem --adapter persona --samples 50   --seeds 42,123,456   --output results/persona_personamem_canonical_150q

# PersonaMem full benchmark (589Q, single seed example)
PYTHONUNBUFFERED=1 .venv/bin/mem-eval run   --benchmark personamem --adapter persona --seeds 42   --output results/persona_personamem_full_seed42

# BEAM full (10 abilities)
PYTHONUNBUFFERED=1 .venv/bin/mem-eval run   --benchmark beam --adapter persona --samples 100   --seeds 42,123,456,789,999,101,202,303,404,505   --output results/persona_beam_canonical_10seeds
```

---

## Notes

- This file is intentionally conservative. It is the source of truth for external claims.
- If a claim is missing, add the artifact first, then regenerate this file.
