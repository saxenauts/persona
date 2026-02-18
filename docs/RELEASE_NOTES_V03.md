# Persona v0.3 Release Notes

Last updated: 2026-02-17 (America/Los_Angeles)

## Scope and Integrity

- These notes use canonical claims from `docs/CLAIMS_TABLE_V03.md` only.
- No new eval runs are included in this release program.
- Methodology source of truth: `release_artifacts/methodology.md`.

## Headline Metrics (Audit-Grade)

- PersonaMem subset baseline: 65.3% (N=150, seeds 42/123/456).
- PersonaMem full single-seed baseline: 66.2% (N=589, seed 42).
- LongMemEval baseline: 81.3% (N=300, seeds 42/123/456).
- BEAM (10 abilities) baseline: 69.0% (N=100, seed 1; `event_ordering=0%`).

All metric artifacts are listed in `docs/CLAIMS_TABLE_V03.md` and traced to `release_artifacts/audit_2026-01-31/results/`.

## Key Improvements in v0.3

- Temporal bug-fix set H1-H5 is integrated and validated in this release line.
- Agent/tool observability improvements improve retrieval and debugging transparency.
- Canonical methodology and claims governance are now explicit and frozen for release and paper consistency.

## Limitations and Safe-Claim Boundaries

- Experimental paired-slice results are informative and non-headline.
- No claim of `>=70%` PersonaMem release attainment is made.
- No superiority claim versus Graphiti/Mem0 is made without direct artifact-backed comparison.

## Canonical References

- `docs/CLAIMS_TABLE_V03.md`
- `docs/METHODOLOGY_CANONICAL_V03.md`
- `release_artifacts/methodology.md`
- `docs/PERSONAMEM_EVAL_CANONICAL.md`
