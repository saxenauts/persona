# Persona Memory Evaluation Results - January 2026 (Audit-Grade)

**Date**: January 31, 2026  
**Evaluation Period**: January 26-31, 2026  
**Model**: GPT-5.2 via Azure Foundry  
**Benchmarks**: PersonaMem, BEAM (10 abilities)  
**Source of Truth**: `docs/BENCHMARK_TRUTH_TABLE.md`

---

## Executive Summary (HN-Grade Honest)

This document only reports results that can be traced to on-disk JSON artifacts. Prior claims of 80%+ PersonaMem and BEAM are **not** auditable and are **retired** until supporting artifacts are committed.

**Audit-grade results (on disk):**
- **PersonaMem (subset)**: **65.3%** on 150 questions, 3 seeds (42/123/456).  
  Source: `../memory-evals/results/competitor_full/persona_personamem/persona_personamem_summary.json`
- **PersonaMem (full benchmark, single seed)**: **66.2%** on 589 questions, seed 42.  
  Source: `../memory-evals/results/persona_personamem_full/persona_personamem_seed42.json`
- **BEAM (10 abilities, 100Q, 1 seed)**: **69.0%** baseline; **event_ordering = 0%**.  
  Source: `../memory-evals/results/beam_100q_minimal_prompt/run_20260126_113003/final_results.json`

---

## Audit-Grade Results (On-Disk)

| Benchmark | Accuracy | Samples | Seeds | Notes | Source |
|-----------|----------|---------|-------|-------|--------|
| PersonaMem (subset) | **65.3%** | 150 (50 per seed) | 42, 123, 456 | `recall_user_shared_facts` only | `../memory-evals/results/competitor_full/persona_personamem/persona_personamem_summary.json` |
| PersonaMem (full) | **66.2%** | 589 | 42 | Full benchmark, single seed | `../memory-evals/results/persona_personamem_full/persona_personamem_seed42.json` |
| BEAM (10 abilities) | **69.0%** | 100 | 1 seed | Event ordering still 0% | `../memory-evals/results/beam_100q_minimal_prompt/run_20260126_113003/final_results.json` |

---

## Exploratory Results (Not Release-Grade)

These runs are **real** but **not** comparable to the audit-grade set. Use only for debugging, not marketing.

| Benchmark | Accuracy | Samples | Seeds | Reason Not Release-Grade | Source |
|-----------|----------|---------|-------|--------------------------|--------|
| PersonaMem (subset, smaller) | 70.0% | 90 | 42, 123, 456 | Smaller sample than 150Q run | `../memory-evals/results/persona_personamem_summary.json` |
| BEAM (2 abilities only) | 79.6% | 240 (80 per seed) | 456, 789, 999 | Only info_extraction + temporal_reasoning | `../memory-evals/results/persona_beam_summary.json` |

---

## BEAM (10 Abilities) - Current Status

Baseline run (100Q, 10 abilities):
- **Overall**: 69.0%
- **event_ordering**: 0.0%
- **knowledge_update**: 60.0%
- **contradiction_resolution**: 50.0%
- **multi_session_reasoning**: 50.0%

This confirms the event ordering failure is **not** solved by prompt changes and likely needs a tooling/output-contract fix.

Source: `../memory-evals/results/beam_100q_minimal_prompt/run_20260126_113003/final_results.json`

---

## PersonaMem Retrieval Analysis (Authoritative 150Q Run)

**Key finding**: retrieval is strong but **not decisive**; answer selection is the bottleneck.

From `../memory-evals/results/competitor_full/persona_personamem/run_20260129_173442/deep_logs.jsonl`:

| Metric | Correct (n=98) | Incorrect (n=52) | Note |
|--------|----------------|------------------|------|
| Mean top recall score | 0.836 | 0.827 | Small gap |
| Median top recall score | 0.835 | 0.817 | Small gap |
| 0.8+ top recall | 86% | 71% | Many failures still have high recall |
| Mean recall count | 8.57 | 8.73 | Nearly identical |

Conclusion: retrieval quality is high for both correct and incorrect answers. Improvements should focus on **answer selection and preference inference**, not recall.

---

## Gold Label Audit Status

Claims of "40% gold label errors" and "82% adjusted accuracy" are **not auditable** because no stored audit artifact exists in this repo. Until an audit file is committed (e.g., `gold_audit.jsonl`), these claims must remain **hypotheses** only.

---

## Memeplex Presence

Memeplex + UserCard appear in **100%** of queries in the authoritative PersonaMem run (150/150). This rules out "missing world model" as the primary failure mode.

Source: `../memory-evals/results/competitor_full/persona_personamem/run_20260129_173442/deep_logs.jsonl`

---

## What We Can Claim Today

- **PersonaMem (subset)**: 65.3% on 150 questions (3 seeds).  
- **PersonaMem (full, 1 seed)**: 66.2% on 589 questions.  
- **BEAM (10 abilities, 1 seed)**: 69.0% baseline, event_ordering still 0%.

---

## What We Cannot Claim Yet

- 80%+ PersonaMem or BEAM headline results (no auditable artifacts).
- Adjusted accuracy based on gold label errors (no committed audit).
- Competitor outperformance without matched protocol + stored manifests.
- Memeplex causal improvement (no A/B test results committed).

---

## Recommended Next Steps (Data-Driven)

1. **Run a full PersonaMem multi-seed benchmark** (all 589 questions, 3+ seeds) and store per-run manifests.
2. **Fix event ordering at the tool/output level** (ordering a set of explicit candidates) and re-run BEAM-10.
3. **Commit a gold label audit artifact** before any adjusted accuracy claims.
4. **Add per-run manifest.json** with benchmark, seeds, samples, dataset version, model, temperature, git commit.

---

*This document is the audit-grade baseline as of 2026-01-31. All claims above are backed by on-disk JSON artifacts.*
