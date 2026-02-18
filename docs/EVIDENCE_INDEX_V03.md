# Persona v0.3 Evidence Index (Phase 1)

Last updated: 2026-02-17 (America/Los_Angeles)

## Scope Lock

This index is built from existing artifacts only. No new eval runs are included.

## Active Canonical Evidence (Release + Paper)

- `docs/PERSONAMEM_EVAL_CANONICAL.md`
- `docs/PERSONAMEM_TWO_WEEK_REVIEW_20260210.md`
- `release_artifacts/methodology.md`
- `release_artifacts/audit_2026-01-31/docs/BENCHMARK_TRUTH_TABLE.md`
- `release_artifacts/audit_2026-01-31/docs/RELEASE_GATE_PLAN.md`
- `release_artifacts/audit_2026-01-31/results/persona_personamem_summary.json`
- `release_artifacts/audit_2026-01-31/results/persona_personamem_seed42.json`
- `release_artifacts/audit_2026-01-31/results/persona_longmemeval_summary.json`
- `release_artifacts/audit_2026-01-31/results/final_results.json`

### Active Evidence Classification

- `audit-grade`:
  - `release_artifacts/audit_2026-01-31/results/persona_personamem_summary.json`
  - `release_artifacts/audit_2026-01-31/results/persona_personamem_seed42.json`
  - `release_artifacts/audit_2026-01-31/results/persona_longmemeval_summary.json`
  - `release_artifacts/audit_2026-01-31/results/final_results.json`
- `experimental`:
  - `../memory-evals/results/run_20260208_154926/*`
  - `../memory-evals/results/run_20260208_212320/*`
  - `../memory-evals/results/ulw_paired100_20260209_armA/*`
  - `../memory-evals/results/ulw_paired100_20260209_armB/*`

## Archive and Hidden Evidence Inventory

### `docs/`

- `docs/archive/eval/2026-02/` contains 11 eval-analysis docs, including:
  - `docs/archive/eval/2026-02/EVIDENCE_MAP_20260209.md`
  - `docs/archive/eval/2026-02/FAILURE_ANALYSIS_20260209.md`
  - `docs/archive/eval/2026-02/generated/EVALUATION_REVIEW_INDEX.md`
- `docs/research/` contains:
  - `docs/research/AI_MEMORY_LANDSCAPE_2026.md`

### `.sisyphus/`

- `.sisyphus/reports/` contains 4 audit/review reports:
  - `.sisyphus/reports/release-audit-2026-02-01.md`
  - `.sisyphus/reports/beam-accuracy-improvement-codex-review.md`
  - `.sisyphus/reports/architecture-audit-2026-02-01.md`
  - `.sisyphus/reports/llm-first-architecture-audit-2026-02-01.md`
- `.sisyphus/plans/` contains 17 plan docs (active + archived plan history).
- `.sisyphus/notepads/eval-bulletproof-claims/` contains 34 execution/decision notes.
- `.sisyphus/notepads/eval-done-right/` contains 19 analysis/forensics notes.
- `.sisyphus/notepads/v03-release-prep/` contains 6 release prep notes.
- `.sisyphus/archive/` contains 36 historical cleanup artifacts.

### `.opencode/`

- `.opencode/eval/` contains 4 eval-tracking docs.
- `.opencode/release/` contains 1 release packet.
- `.opencode/research/` contains 1 research survey.
- `.opencode/archive/` contains 44 archived receipts/reports/plans.

## External Results Paths Referenced by Canonical Docs

Validated existing result folders under sibling repo path:

- `../memory-evals/results/run_20260208_154926/summary.json`
- `../memory-evals/results/run_20260208_154926/deep_logs.jsonl`
- `../memory-evals/results/run_20260208_212320/summary.json`
- `../memory-evals/results/ulw_paired100_20260209_armA/run_20260209_133301/summary.json`
- `../memory-evals/results/ulw_paired100_20260209_armA/run_20260209_133301/deep_logs.jsonl`
- `../memory-evals/results/ulw_paired100_20260209_armB/run_20260209_035700/summary.json`
- `../memory-evals/results/ulw_paired100_20260209_armB/run_20260209_035700/deep_logs.jsonl`

## Integrity Notes

- Audit snapshot artifacts inside `release_artifacts/audit_2026-01-31/results/` are present and readable.
- Some legacy source paths shown in `release_artifacts/audit_2026-01-31/docs/BENCHMARK_TRUTH_TABLE.md` point to old external run locations that are no longer present in the sibling `memory-evals` layout.
- For v0.3 release and paper claims, use in-repo audit snapshot result files and explicitly validated sibling run folders listed above.

## Failure and Rollback Ledger (Evidence-Mapped)

- 2026-01-31 BEAM strict prompt wave regressed from 69% to 39-41% and was fully reverted.
  - Evidence: `.sisyphus/reports/beam-accuracy-improvement-codex-review.md`
  - Revert commits recorded in report: `600b76a`, `c281c9a`, `d457dd7`, `1a9ac70`
- Release audit blocked release-grade state due unmet gates (event ordering, multi-seed completeness, integration validation).
  - Evidence: `.sisyphus/reports/release-audit-2026-02-01.md`
- Canonical policy keeps paired uplift experimental until promotion gate is passed.
  - Evidence: `docs/PERSONAMEM_EVAL_CANONICAL.md`

## Paper-Ready Fact Pack (Source Paths)

- Architecture and memory model:
  - `docs/ARCHITECTURE.md`
  - `docs/MEMORY_MODEL.md`
- Truth-first eval status and decision history:
  - `docs/PERSONAMEM_EVAL_CANONICAL.md`
  - `docs/PERSONAMEM_TWO_WEEK_REVIEW_20260210.md`
- Benchmark and methodology provenance:
  - `release_artifacts/methodology.md`
  - `release_artifacts/audit_2026-01-31/docs/BENCHMARK_TRUTH_TABLE.md`
  - `release_artifacts/audit_2026-01-31/docs/RELEASE_GATE_PLAN.md`

## Phase 1 Exit Receipt

- Evidence inventory created across active docs, archives, and hidden folders.
- Artifact-backed claim sources identified and validated for active use.
- Legacy path drift documented to prevent claim-path mismatch.
