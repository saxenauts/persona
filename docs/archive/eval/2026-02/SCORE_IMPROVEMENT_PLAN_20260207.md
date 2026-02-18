# Score Improvement Plan (2026-02-07)

Purpose: keep a stable execution reference while implementing score improvements, so active coding stays focused and does not drift.

## Baseline problem statement

Primary degradants from `../memory-evals/results/LATEST_ANALYSIS_REFERENCE_20260207.md`:

1. Over-retrieval and context dilution
2. Temporal signal fragility across memory lifecycle
3. Prompt and loop contract mismatch
4. Temporal selection and interval math errors under noisy context

## Execution plan (implementation order)

### P0-A: Retrieval budget enforcement

- Enforce tighter practical limits in agent loop (recall/expansion/total calls)
- Add novelty-aware stopping behavior (no new evidence -> stop searching)
- Keep graceful fallback behavior by returning tool-budget feedback to the model

Success gate:
- Reduced average tool calls and context size in eval deep logs
- No regression in straightforward retrieval tasks

### P0-B: Prompt/schema alignment for low-call policy

- Remove contradictory guidance between system prompt and tool schema
- Standardize stop condition around 1-2 recalls and evidence sufficiency
- Preserve explicit instructions for chronology and evidence-first output

Success gate:
- Fewer deep tool loops in matched runs
- Stable or improved answer accuracy at lower call depth

### P1-A: Deterministic temporal interval assist

- Add a dedicated date interval tool for exact deltas from two resolved dates
- Update prompt/schema guidance to use tool for interval math instead of mental arithmetic

Success gate:
- Lower temporal MCQ failure rate where evidence dates are present

### P1-B: Context pressure control

- Tune default working-memory budget downward to reduce pre-loop context mass
- Keep enough recent episodes/psyche/notes to preserve personalization quality

Success gate:
- Lower prompt/context token footprint with no drop in non-temporal questions

## Evaluation protocol

- Use matched seed/slice/config comparisons only
- Track both exact-match and rubric-based score slices
- Record ingestion success, sample size, and scoring mode in every report

## Out of scope for this iteration

- Broad consolidation rewrite
- Benchmark-specific hacks
- Unsafe merge behavior changes without dedicated regression coverage
