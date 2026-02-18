# Evaluation Comparison Report (Audit-Grade)

**Date**: January 31, 2026  
**Evaluation Framework**: memory-evals v0.1  
**Model**: GPT-5.2 (Azure Foundry)  

---

## Executive Summary

This report only includes results that can be traced to on-disk JSON artifacts. Prior claims of 70–82% PersonaMem and 80%+ BEAM are **not audit-grade** and are intentionally excluded.

Source of truth: `docs/BENCHMARK_TRUTH_TABLE.md`

---

## Verified Results (On-Disk)

| Benchmark | Accuracy | Samples | Seeds | Notes | Source |
|-----------|----------|---------|-------|-------|--------|
| PersonaMem (subset) | **65.3%** | 150 | 42, 123, 456 | recall_user_shared_facts only | `../memory-evals/results/competitor_full/persona_personamem/persona_personamem_summary.json` |
| PersonaMem (full) | **66.2%** | 589 | 42 | Full benchmark, single seed | `../memory-evals/results/persona_personamem_full/persona_personamem_seed42.json` |
| LongMemEval | **81.3%** | 300 | 42, 123, 456 | 3 seeds × 100 | `../memory-evals/results/persona_longmemeval_summary.json` |
| BEAM (10 abilities) | **69.0%** | 100 | 1 | Event ordering still 0% | `../memory-evals/results/beam_100q_minimal_prompt/run_20260126_113003/final_results.json` |

---

## Retrieval vs Answer Selection (Authoritative PersonaMem Run)

From `../memory-evals/results/competitor_full/persona_personamem/run_20260129_173442/deep_logs.jsonl`:

| Metric | Correct (n=98) | Incorrect (n=52) | Note |
|--------|----------------|------------------|------|
| Mean top recall score | 0.836 | 0.827 | Small gap |
| Median top recall score | 0.835 | 0.817 | Small gap |
| 0.8+ top recall | 86% | 71% | Many failures still have high recall |
| Mean recall count | 8.57 | 8.73 | Nearly identical |

**Conclusion**: Retrieval is strong but not decisive. The bottleneck is answer selection and preference inference.

---

## Exploratory Competitor Snapshot (Not Comparable)

| System | Accuracy | Samples | Notes | Source |
|--------|----------|---------|-------|--------|
| Mem0 | 61.9% | 147 | Mixed schema across seeds | `../memory-evals/results/competitor_full/mem0_personamem/` |

**Important**: This comparison is **not release-grade** due to schema mismatches and inconsistent sample sizes.

---

## What We Can Claim Today

- PersonaMem (subset): 65.3% on 150Q (3 seeds).
- PersonaMem (full, single seed): 66.2% on 589Q.
- LongMemEval: 81.3% on 300Q (3 seeds).
- BEAM (10 abilities, 1 seed): 69.0% baseline; event ordering remains 0%.

---

## What We Cannot Claim Yet

- 80%+ PersonaMem or BEAM headline performance.
- Adjusted accuracy without a committed gold-label audit artifact.
- Causal improvements from memeplex without A/B tests.
- Competitor outperformance without matched protocols.

---

## Recommended Next Steps

1. Full PersonaMem 589Q across ≥3 seeds with per-run manifests.
2. Event ordering tool fix; re-run BEAM-10 with ≥5 seeds.
3. Commit gold-label audit artifact before any adjusted-accuracy claim.
4. Standardize run manifests for every benchmark run.

---

*This report is intentionally conservative to survive HN-level scrutiny.*
