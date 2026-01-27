# Persona v0.3 Evaluation Results

**Date**: January 26, 2026  
**Status**: PRELIMINARY - Official stratified eval pending  
**Benchmarks**: PersonaMem (run), LongMemEval (run), BEAM (not yet run)

## Summary

| Benchmark | Persona v0.3 | Status | Notes |
|-----------|--------------|--------|-------|
| PersonaMem | 50-75% | Preliminary | Single-type sampling, needs stratified run |
| LongMemEval | 75% | Preliminary | Limited samples |
| BEAM | — | Not run | Benchmark exists but no verified scores |

## Current Status

**What we have:**
- PersonaMem benchmark runs with varying results (50-75% depending on question type sampling)
- LongMemEval preliminary run showing 75%
- Working eval infrastructure in `memory-evals/`

**What we need:**
- Official stratified PersonaMem run (50 samples × 3 seeds, all question types)
- Verified reproducible results with methodology documented
- BEAM benchmark execution (currently planned but not run)

## PersonaMem Preliminary Results

PersonaMem tests personal memory recall using multiple-choice questions about user-shared facts.

| Run | Configuration | Accuracy | Notes |
|-----|--------------|----------|-------|
| Run 1 | 20 samples, seed 42 | 66.7% | May have type bias |
| Run 2 | 10 samples, seed 42 | 50.0% | Different question set |
| Run 3 | 10 samples, seed 42 | 75.0% | LongMemEval variant |

**Baseline**: Random chance = 25% (4-choice MCQ)  
**Known issue**: Results vary significantly based on which question types are sampled.

## Methodology (To Be Finalized)

**Planned official configuration:**
```bash
# Stratified sampling across all question types
mem-eval run --benchmark personamem --adapter persona --samples 50 --seeds 42,123,456

# Expected output structure:
# - 50 samples × 3 seeds = 150 total questions
# - Balanced across: single-session, multi-session, temporal, preference, evolution
```

**Fairness measures:**
- Same LLM for all adapters
- Same questions in same order (deterministic seeding)
- Same evaluation judge
- Multiple seeds to reduce variance

## BEAM Status

BEAM (Beyond a Million Tokens) benchmark code exists in `memory-evals/` but has **not been run** with verified results.

Previous claims of 90% BEAM accuracy were **incorrect** - those results cannot be reproduced and should not be cited.

## Reproducing Results

```bash
cd memory-evals

# Current working command (PersonaMem)
uv run mem-eval run --benchmark personamem --adapter persona --seeds 42 --samples 20

# Server must be running
cd ../persona && docker compose up -d
```

## Next Steps

1. Run official stratified PersonaMem eval (50 samples × 3 seeds)
2. Document exact configuration and environment
3. Run BEAM benchmark for first verified scores
4. Update this document with final results

---

*This document will be updated with official results after v0.3 eval run.*
