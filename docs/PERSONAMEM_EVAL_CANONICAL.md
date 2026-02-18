# PersonaMem Evaluation Canonical Status

**Last Updated**: 2026-02-10  
**Purpose**: Single source of truth for current PersonaMem evaluation status, evidence-backed decisions, and next gate.
**Detailed Review**: `docs/PERSONAMEM_TWO_WEEK_REVIEW_20260210.md`

---

## Scope And Source Rules

This file is the only active summary document for PersonaMem evaluation status in `docs/`.

Every metric in this file includes:
- denominator (`N`),
- run path(s),
- and whether the claim is **audit-grade** or **experimental**.

If a claim is not linked to a concrete artifact path, treat it as hypothesis.

### Release-Grade Label Rules (v0.3)

Source policy: `release_artifacts/audit_2026-01-31/docs/RELEASE_GATE_PLAN.md` and `release_artifacts/audit_2026-01-31/docs/BENCHMARK_TRUTH_TABLE.md`.

**Audit-grade (release-eligible) only if all are true**:
- summary artifact exists (`summary.json` or equivalent per-seed JSON),
- `deep_logs.jsonl` exists for traceability,
- run scope and sample size are explicit,
- claim appears in benchmark truth table,
- no adjusted-accuracy claim without committed `gold_audit.jsonl`.

**Experimental (not release headline)** if any are true:
- fixed-ID subset experiments used for tuning,
- single-seed or partial-run evidence,
- claim not yet represented in audit truth table.

---

## Current Supported Results

### Audit-Grade Baseline (Jan 31 Freeze)

Source: `release_artifacts/audit_2026-01-31/docs/BENCHMARK_TRUTH_TABLE.md`

| Metric | Value | N | Seeds | Evidence |
|---|---:|---:|---|---|
| PersonaMem (subset) | 65.3% | 150 | 42,123,456 | `../memory-evals/results/competitor_full/persona_personamem/persona_personamem_summary.json` |
| PersonaMem (full) | 66.2% | 589 | 42 | `../memory-evals/results/persona_personamem_full/persona_personamem_seed42.json` |

### Experimental Paired A/B (Feb 8-9)

These are valid and reproducible, but are not yet promoted to external claim baseline.

| Experiment | Arm A | Arm B | Delta | Stats | Evidence |
|---|---|---|---:|---|---|
| Paired 50Q (fixed IDs) | 64.0% (32/50) | 68.0% (34/50) | +4pp | McNemar p=0.6875 | `../memory-evals/results/run_20260208_154926`, `../memory-evals/results/run_20260208_212320` |
| Paired 100Q (fixed IDs) | 63.0% (63/100) | 67.0% (67/100) | +4pp | McNemar p=0.34375 | `../memory-evals/results/ulw_paired100_20260209_armA/run_20260209_133301`, `../memory-evals/results/ulw_paired100_20260209_armB/run_20260209_035700` |

Denominator note for paired 100Q Arm B:
- Raw Arm B `summary.json` reports `82/119` because duplicate-ID entries are present in that run output.
- Canonical paired view is normalized to the shared 100 unique IDs against Arm A, yielding `67/100`.

Operational diagnostics on paired runs:
- no tool inflation (`1.04 -> 1.02` on 50Q, `1.02 -> 1.00` on 100Q),
- no cost/loop blow-up observed,
- effect is directionally positive but not significance-strong yet.

---

## Root-Cause Status (Most Reliable Findings)

Source: `../memory-evals/results/run_20260208_154926/deep_logs.jsonl`, `../memory-evals/results/ulw_paired100_20260209_armA/run_20260209_133301/deep_logs.jsonl`, `../memory-evals/results/ulw_paired100_20260209_armB/run_20260209_035700/deep_logs.jsonl`

1. **Temporal evolution misses are real (P0)**: at least 3 deterministic failures where earlier preference is stored and later reversal is not reflected (cases 342, 80, 131).
2. **Retrieval insufficiency is a major driver**: in 50Q forensics, initial classification marked `14/18` as ambiguous, but source-trace review reclassified most of those as retrieval insufficiency once full conversation arcs were examined.
3. **`suggest_new_ideas` is structurally weak**: ~30% in the 100Q paired slice (Arm B: 6/20), with option-style/verbosity effects interacting with weak evidence selection.
4. **Second-pass gate is provisional winner**: consistent `+4pp` in 50Q and 100Q paired tests, with stable cost profile.

---

## Decision Register

### Accepted (Provisional)
- Keep confidence-gated second pass enabled as current candidate default while repeatability is verified.

### Not Yet Accepted
- Any claim that PersonaMem has stably moved beyond Jan audit baseline.
- Any claim of statistically confirmed uplift from second-pass gating.

### Next Promotion Gate
- Run one more paired 100Q seed (`seed=123`) with identical ID set.
- Promotion condition: non-negative delta and no tool-call inflation.
- If gate fails, rollback second-pass default to Arm A behavior.

---

## v0.3 Release Alignment

Primary plan reference: `.sisyphus/plans/v03-release-execution.md`.

| Release requirement | Target in plan | Current evidence state | Status |
|---|---:|---:|---|
| PersonaMem release target | >=70% | 63-67% on recent paired 100Q experimental runs | Not met |
| BEAM release target | >=65% | Not tracked in this canonical PersonaMem doc | Out of scope here |
| Claim discipline | Artifact-backed only | Enforced in this file | Met |
| No inflated claims | Required | Enforced in this file | Met |

Interpretation:
- This document is compliant for **truth-first claim discipline**.
- This document does **not** assert v0.3 target attainment; it records current state and next gate.

---

## Update Protocol

This is the required update workflow for keeping this file accurate.

1. Complete run(s) and verify outputs exist (`summary.json`, `deep_logs.jsonl`).
2. Add/refresh audit artifacts and truth-table entries under `release_artifacts/audit_2026-01-31/docs/` (or newer dated audit snapshot when created).
3. Update `Current Supported Results` with:
   - run IDs,
   - denominator (`N`),
   - seed set,
   - paired delta/stats if applicable.
4. Update `Decision Register` with promote/hold/rollback outcome.
5. Append new evidence paths in `Canonical Evidence Index`.

Hard rule: if step 2 is missing, keep results labeled as experimental.

---

## Canonical Evidence Index

Use these first when answering "where are we now":

1. `release_artifacts/audit_2026-01-31/docs/BENCHMARK_TRUTH_TABLE.md` (audit baseline)
2. `../memory-evals/results/run_20260208_154926/summary.json` (50Q Arm A)
3. `../memory-evals/results/run_20260208_212320/summary.json` (50Q Arm B)
4. `../memory-evals/results/ulw_paired100_20260209_armA/run_20260209_133301/summary.json` (100Q Arm A)
5. `../memory-evals/results/ulw_paired100_20260209_armB/run_20260209_035700/summary.json` (100Q Arm B)
6. `../memory-evals/results/ulw_paired100_20260209_armA/run_20260209_133301/deep_logs.jsonl` (failure forensics)
7. `../memory-evals/results/ulw_paired100_20260209_armB/run_20260209_035700/deep_logs.jsonl` (failure forensics)

---

## Archived Working Notes

The following docs were useful during exploration but are now archived to reduce active-doc sprawl:
- `docs/archive/eval/2026-02/SCORE_IMPROVEMENT_PLAN_20260207.md`
- `docs/archive/eval/2026-02/SCORE_IMPROVEMENT_EXECUTION_20260207.md`
- `docs/archive/eval/2026-02/ULW_RESEARCH_BRIEF_20260208.md`
- `docs/archive/eval/2026-02/ULW_EXECUTION_TRACE_20260208.md`
- `docs/archive/eval/2026-02/ULW_85_EXECUTION_FRAMEWORK_20260208.md`
- `docs/archive/eval/2026-02/FAILURE_ANALYSIS_20260209.md`
- `docs/archive/eval/2026-02/MEMEPLEX_ARCHITECTURE_RESEARCH_20260209.md`
- `docs/archive/eval/2026-02/EVIDENCE_MAP_20260209.md`

Use archive docs for drill-down details, not top-line status.
