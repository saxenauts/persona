# Persona v0.3 Canonical Claims Table

Last updated: 2026-02-17 (America/Los_Angeles)

## Rules

- This table is the canonical claim source for release notes and paper drafting.
- No new eval runs are included; only existing artifacts are allowed.
- Every row must include metric scope, sample size (`N`), seed set, and artifact path.
- Claim labels:
  - `audit-grade`: release-eligible if artifacts are present and traceable.
  - `experimental`: informative only; not release headline claims.

## Audit-Grade Claims

| Claim ID | Claim | Metric | N | Seeds | Label | Artifact Path | Notes |
|---|---|---:|---:|---|---|---|---|
| A-001 | PersonaMem subset baseline | 65.3% | 150 | 42,123,456 | audit-grade | `release_artifacts/audit_2026-01-31/results/persona_personamem_summary.json` | Mean across 3 seeds |
| A-002 | PersonaMem full single-seed baseline | 66.2% | 589 | 42 | audit-grade | `release_artifacts/audit_2026-01-31/results/persona_personamem_seed42.json` | Single-seed full run |
| A-003 | LongMemEval baseline | 81.3% | 300 | 42,123,456 | audit-grade | `release_artifacts/audit_2026-01-31/results/persona_longmemeval_summary.json` | Mean across 3 seeds |
| A-004 | BEAM 10 abilities baseline | 69.0% | 100 | 1 | audit-grade | `release_artifacts/audit_2026-01-31/results/final_results.json` | `event_ordering=0%` |

## Experimental Claims (Non-Headline)

| Claim ID | Claim | Metric | N | Seeds | Label | Artifact Path | Notes |
|---|---|---:|---:|---|---|---|---|
| E-001 | Paired 50Q Arm A | 64.0% (32/50) | 50 | fixed-ID slice | experimental | `../memory-evals/results/run_20260208_154926/summary.json` | Used for paired A/B comparison |
| E-002 | Paired 50Q Arm B | 68.0% (34/50) | 50 | fixed-ID slice | experimental | `../memory-evals/results/run_20260208_212320/summary.json` | Directional +4pp vs Arm A |
| E-003 | Paired 100Q Arm A | 63.0% (63/100) | 100 | fixed-ID slice | experimental | `../memory-evals/results/ulw_paired100_20260209_armA/run_20260209_133301/summary.json` | Arm A baseline in paired 100Q |
| E-004 | Paired 100Q Arm B (normalized paired view) | 67.0% (67/100) | 100 | fixed-ID slice | experimental | `docs/PERSONAMEM_EVAL_CANONICAL.md` | Canonical paired view; raw Arm B summary file may include additional rows |

## Explicitly Not Allowed As Headline Claims

- PersonaMem `>=70%` release attainment.
- Statistically confirmed uplift from second-pass gating.
- Any superiority claim vs Graphiti/Mem0 without direct artifact-backed comparison.

## Supporting Canonical Sources

- `docs/PERSONAMEM_EVAL_CANONICAL.md`
- `docs/PERSONAMEM_TWO_WEEK_REVIEW_20260210.md`
- `docs/METHODOLOGY_CANONICAL_V03.md`
- `release_artifacts/audit_2026-01-31/docs/BENCHMARK_TRUTH_TABLE.md`
