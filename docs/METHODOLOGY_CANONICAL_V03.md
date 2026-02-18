# Canonical Methodology Decision (v0.3)

Last updated: 2026-02-17 (America/Los_Angeles)

## Decision

For v0.3 release and paper work, the single canonical methodology source is:

- `release_artifacts/methodology.md`

## Why This Source Won

- Includes explicit reproducibility package references (`docker-compose.yml`, `poetry.lock`, `model_config.json`).
- Includes implemented bug-fix context (H1-H5) and partial-validation limitations.
- Aligns with truth-first release posture and artifact-backed framing.

## Non-Canonical Methodology Docs

The following are historical/reference-only for v0.3:

- `docs/EVAL_METHODOLOGY_V03.md`
- `docs/EVAL_METHODOLOGY.md`

They may contain useful context but are not source-of-truth for external claims.

See also: `docs/LEGACY_DOCS_ARCHIVE_POINTERS_V03.md` for the broader legacy-doc pointer map used during Phase 3 hardening.

## Enforcement

- README/release/paper claims must point to canonical methodology source above.
- If a claim cannot be justified by canonical methodology + canonical claims table, it is out of scope.
- No new eval runs are admitted under this program scope.
