# PersonaMem Accuracy Audit Report

**Date**: 2026-01-30  
**Auditor**: Atlas (Orchestrator)  
**Scope**: Reconcile conflicting PersonaMem accuracy claims (70% vs 51.4%)

---

## Executive Summary

**The 51.4% claim is ERRONEOUS.** It references runs that do not exist in actual result files.

**Authoritative Accuracy**: **65.3%** (150 questions, 3 seeds × 50 questions, Jan 29 2026)

**Recommendation**: Retire 51.4% and 70% claims. Use 65.3% as primary benchmark number.

---

## Complete Catalogue of PersonaMem Runs

| Location | Date | Seeds | Q/Seed | Total | Accuracy | Status |
|----------|------|-------|--------|-------|----------|--------|
| `competitor_full/persona_personamem/` | Jan 29 | 42,123,456 | 50 | 150 | **65.3%** | **AUTHORITATIVE** |
| `persona_personamem_summary.json` | Jan 28 | 42,123,456 | 30 | 90 | 70.0% | Smaller sample |
| `persona_personamem_5seeds_summary.json` | Jan 27 | 42,123,456,789,999 | ~20 | ~100 | 62.0% | Mixed samples |
| `persona_personamem_full/` | Jan 27 | 42 | 589 | 589 | 66.2% | Full benchmark, 1 seed |
| `eval_run_persona_personamem/` | Jan 24 | 42,123,456 | 20 | 60 | 66.7% | Early run |
| `persona_personamem_final/` | Jan 27 | 789,999 | 10 | 20 | 55.0% | Incomplete |

---

## Authoritative Run: Per-Seed Breakdown

**Source**: `competitor_full/persona_personamem/run_20260129_173442/`

| Seed | Questions | Correct | Accuracy |
|------|-----------|---------|----------|
| 42 | 50 | 32 | 64% |
| 123 | 50 | 29 | 58% |
| 456 | 50 | 37 | 74% |
| **Total** | **150** | **98** | **65.3%** |

**Methodology**: Stratified sample of `recall_user_shared_facts` question type only (subset of full 589-question benchmark).

---

## Root Cause Analysis

### Why 70.0% Exists (Valid but Smaller Sample)

**Source**: `persona_personamem_summary.json` (root level)
- Sample size: 30 questions/seed instead of 50
- Same seeds: 42, 123, 456
- Per-seed: 70%, 73.3%, 66.7%
- **Interpretation**: Statistical variance from smaller sample (90 vs 150 questions)

### Why 51.4% Was Claimed (ERRONEOUS)

**Source**: `.opencode/RECOMMENDATIONS_FOR_RELEASE.md` states:
> "0.58 / 0.52 / 0.50 / 0.50 (50q each), plus partial runs (48q, 11q). Weighted overall: **133/259 = 51.4%**"

**This is provably incorrect.** Actual seed-level files show:
- Seed 42: **64%** (not 0.58)
- Seed 123: **58%** (matches one value)
- Seed 456: **74%** (not 0.50 or 0.52)

**Conclusion**: The claim references **non-existent runs**. No files with "0.52 / 0.50 / 0.50" accuracy exist in `memory-evals/results/`.

### Why 66.2% Exists (Full Benchmark, Single Seed)

**Source**: `persona_personamem_full/`
- **589 questions** - the FULL PersonaMem benchmark across all 7 question types
- Single seed (42) - more comprehensive but less statistically robust

**Question Type Breakdown**:

| Question Type | Count | Accuracy |
|---------------|-------|----------|
| recall_user_shared_facts | 129 | 65.9% |
| track_full_preference_evolution | 139 | 65.5% |
| recalling_the_reasons_behind_previous_updates | 99 | 86.9% |
| suggest_new_ideas | 93 | 36.6% |
| generalizing_to_new_scenarios | 57 | 71.9% |
| provide_preference_aligned_recommendations | 55 | 72.7% |
| recalling_facts_mentioned_by_the_user | 17 | 76.5% |

---

## Recommendation: Authoritative Baseline

**Primary Claim**: **65.3%** on PersonaMem
- **Source**: `competitor_full/persona_personamem/run_20260129_173442/summary.json`
- **Methodology**: 3 seeds (42, 123, 456) × 50 questions = 150 total
- **Question Type**: `recall_user_shared_facts` only (subset of full benchmark)
- **Date**: Jan 29, 2026

**Secondary Reference**: **66.2%** on Full PersonaMem (589 questions, single seed)
- More comprehensive but less statistically robust (1 seed vs 3)

**Retire These Claims**:
- ❌ 70.0% - Valid but smaller sample (90 questions); defer to larger run
- ❌ 51.4% - Erroneous calculation referencing non-existent runs

---

## Methodology for Future Runs

**Standard Protocol**:
1. **Fixed sample size**: 50 questions/seed minimum
2. **Fixed seeds**: Use 42, 123, 456 for reproducibility
3. **Full benchmark**: Run all 589 questions when making headline claims
4. **Manifest per run**: Include git commit, model, temperature, dataset version
5. **Question type breakdown**: Always report per-type accuracy for full runs

**Clean-Room Re-Run Protocol** (if needed):
```bash
# Reset environment
docker compose down -v
docker compose up -d

# Run with fixed config
EVAL_REFRESH_MEMEPLEX=true \
EVAL_CLOSE_SESSIONS=true \
mem-eval run \
  --benchmark personamem \
  --adapter persona \
  --seeds 42,123,456 \
  --samples 50 \
  --output results/clean_run_$(date +%Y%m%d)
```

---

## Competitor Comparison

| System | PersonaMem | Questions | Methodology |
|--------|-----------|-----------|-------------|
| **Persona** | **65.3%** | 150 | 3 seeds × 50q |
| Mem0 | 72.0% | 50 | Single run (not comparable) |
| Honcho | 30.0% | 40 | Severely underperforming |

**Note**: Mem0 run used different sample size. Not directly comparable until same-seed, same-sample runs exist.

---

## Files Referenced

```
memory-evals/results/
├── competitor_full/persona_personamem/           # AUTHORITATIVE (65.3%)
│   ├── persona_personamem_summary.json
│   ├── persona_personamem_seed{42,123,456}.json
│   └── run_20260129_173442/
│       ├── summary.json
│       └── deep_logs.jsonl
├── persona_personamem_summary.json               # Smaller sample (70.0%)
├── persona_personamem_5seeds_summary.json        # Mixed (62.0%)
├── persona_personamem_full/                      # Full benchmark (66.2%)
├── persona_personamem_final/                     # Incomplete (55.0%)
└── eval_run_persona_personamem/                  # Early run (66.7%)
```

---

## Conclusion

**Use 65.3% as the authoritative PersonaMem accuracy.**

The 51.4% claim is provably incorrect and must be removed from all documentation. The 70.0% claim is valid but based on a smaller sample; defer to the larger 150-question run for consistency.

**Next Steps**:
1. Update README.md benchmark table with 65.3%
2. Remove 51.4% references from `.opencode/RECOMMENDATIONS_FOR_RELEASE.md`
3. Add confidence interval: 65.3% ± [calculate from per-seed variance]
