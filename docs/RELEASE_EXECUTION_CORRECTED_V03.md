# Persona v0.3 Corrected Execution Plan (Latest-Work-First)

Last updated: 2026-02-18 (America/Los_Angeles)

## Why This Correction Exists

The prior release-control framing assumed `origin/main` was the release baseline. That assumption is superseded.

Canonical baseline for ongoing v0.3 execution is:

- Branch/worktree: `refactor/v0.3-cognitive-memory` in this repository.
- Evidence timeline: `.opencode/`, `.sisyphus/`, `docs/`, `docs/archive/`, and `release_artifacts/`; `../memory-evals/results/` is corroborative historical context, not required release evidence.

## Canonical Evidence Anchors

- `docs/PERSONAMEM_EVAL_CANONICAL.md`
- `docs/PERSONAMEM_TWO_WEEK_REVIEW_20260210.md`
- `docs/archive/eval/2026-02/generated/EVALUATION_TIMELINE_JAN27_FEB10.md`
- `.opencode/release/RELEASE_PACKET.md`
- `.sisyphus/plans/v03-release-execution.md`
- `release_artifacts/audit_2026-01-31/docs/BENCHMARK_TRUTH_TABLE.md`

## Guardrails

- No new eval runs are required for this correction pass.
- Claims remain artifact-backed only.
- Completed phases are not reopened except when a gate was satisfied under invalid assumptions; in that case, only the affected gate is reopened with correction receipts.

## Corrected Plan

### Phase C0 - Baseline Alignment (Now)

1. Update release checklist/strategy/receipts to latest-work-first assumptions.
2. Keep full traceability to prior assumptions and why they were replaced.
3. Re-open only the validation gates that were satisfied under invalid assumptions.

Exit criteria:

- Strategy and checklist consistently reference latest-work-first baseline.
- Receipts include git/worktree/session/artifact proof.

### Phase C1 - Validation Re-hardening

1. Restore Docker validation to full profile (`docker compose run --rm test`).
2. Re-run release smoke path and claims link validation.
3. Record pass/fail outcomes directly in checklist evidence.

Exit criteria:

- Full Docker profile gate is green, or failure is explicitly documented with blocker status.

### Phase C2 - Release Candidate Curation

1. Curate release-candidate diff from latest-work branch (retain evidence-backed changes; isolate noise).
2. Confirm claims consistency against canonical table and methodology source.
3. Prepare PR package with explicit provenance to latest-work evidence.

Exit criteria:

- Candidate branch is reproducible, claim-consistent (pre-audit), and provenance-complete.

## Sequencing Decision Update (2026-02-18)

- Release-first execution is now explicit: complete release operations before paper drafting/submission.
- Paper assembly is intentionally deferred until after release tag and final release claims audit.
- This sequencing change does not modify scope lock: no new eval runs are admitted.

## Detours and Divergence (This Correction Pass)

- Baseline detour: replaced stale-mainline assumption with latest-work-first baseline.
- Validation detour: reverted temporary unit-only Docker gate to full-profile gate with explicit skip rationale.
- Branch detour: transitioned release-control path from `release/v0.3-integrity-mainline` to `release/v0.3-integrity-latest-work`.
- Claims detour: removed partial benchmark wording drift from README and aligned to canonical claim IDs.
- Governance detour: reconciled evidence-scope and phase-lock/revalidation wording drift.

## Notes on Historical Mismatch

- `origin/main` remains the default remote branch, but not the canonical v0.3 execution baseline for current work.
- Any older docs using `origin/main baseline` are historical unless updated with explicit correction receipts.
