# Persona v0.3 Release + Paper Phase Checklist

Last updated: 2026-02-18 (America/Los_Angeles)

## Hard Rules (Non-Negotiable)

- [x] Scope lock: no new benchmark/eval runs are admitted in this release/paper program from this checklist onward.
- [x] Evidence lock: primary evidence must come from `docs/`, `docs/archive/`, `release_artifacts/`, `.sisyphus/`, and `.opencode/`; `../memory-evals/results/` may be cited only as corroborative historical context.
- [x] Phase lock: completed phases stay closed unless a gate is proven to have been satisfied under invalid assumptions; in that case, reopen only the affected gate with explicit correction receipts.
- [x] Task admission gate: new tasks can be added only if they are phase-level critical (release integrity, claim integrity, reproducibility, or paper credibility).
- [x] Validation gate: no phase is complete without explicit tests/validation evidence listed in that phase.
- [x] WIP gate: one active phase at a time.

## Current Baseline Snapshot (Captured)

- [x] GitHub CLI installed and available (`gh version 2.60.0`).
- [x] GitHub CLI authenticated (`gh auth status` active for `saxenauts`).
- [x] Repo remote verified (`origin=https://github.com/saxenauts/persona.git`).
- [x] Branch staleness measured: current branch `refactor/v0.3-cognitive-memory` is `157` commits ahead and `1` commit behind `origin/main`.
- [x] Working tree confirmed active and extensive; release prep requires curation + validation (latest-work-first), not reset-to-main.

---

## Phase 0 - Control Plane and Branch Hygiene

Goal: establish execution control before any release/paper edits.

- [x] Confirm default branch and protections via GitHub CLI.
- [x] Create a release-control branch strategy doc (how we move from stale branch to clean release branch).
- [x] Decide source branch for release work (`refactor/v0.3-cognitive-memory` latest-work baseline).
- [x] Record branch divergence and risk notes in checklist receipts.

Validation and tests:

- [x] `gh repo view saxenauts/persona --json nameWithOwner,defaultBranchRef` succeeds.
- [x] `git fetch origin --prune` succeeds with no auth/network failure.
- [x] `git rev-list --left-right --count HEAD...origin/main` output is captured in receipts.
- [x] Branch strategy is written and approved in-repo before Phase 1 starts.

Exit criteria:

- [x] We can point to one approved branch strategy and one clean control path.

---

## Phase 1 - Evidence Harvest (Existing Artifacts Only)

Goal: build a complete evidence inventory from existing docs/artifacts, including hidden folders.

- [x] Build evidence index from:
  - `docs/PERSONAMEM_EVAL_CANONICAL.md`
  - `docs/PERSONAMEM_TWO_WEEK_REVIEW_20260210.md`
  - `docs/archive/eval/2026-02/`
  - `release_artifacts/audit_2026-01-31/docs/`
  - `.sisyphus/reports/`
  - `.sisyphus/notepads/` and `.sisyphus/archive/`
  - `.opencode/archive/`, `.opencode/research/`, `.opencode/release/`, `.opencode/eval/`
- [x] Tag each evidence item as `audit-grade` or `experimental`.
- [x] Extract known failures/rollbacks and their root causes into a single ledger.
- [x] Extract paper-ready facts (architecture, methodology, limitations, result tables).

Validation and tests:

- [x] Every external-facing claim has at least one concrete artifact path.
- [x] Every artifact path in the evidence index resolves to a file in the workspace.
- [x] Rollback events are mapped to commit IDs or report evidence.
- [x] No new metrics are introduced without pre-existing artifact backing.

Exit criteria:

- [x] One complete evidence index exists and is referenced by downstream phases.

---

## Phase 2 - Canonical Docs and Claims Freeze

Goal: eliminate doc drift and freeze a single truth ladder for release + paper.

- [x] Choose one canonical v0.3 methodology source (single winner).
- [x] Mark non-canonical methodology docs as historical/archived references.
- [x] Create canonical claims table doc (release and paper both consume this).
- [x] Update README/release docs to reference canonical sources only.
- [x] Add explicit statement: "no new eval runs; claims are artifact-backed from existing evidence only."

Validation and tests:

- [x] There is only one active methodology source for v0.3.
- [x] Claims table rows include: metric, N, seeds, artifact paths, status (`audit-grade` or `experimental`), and limitations.
- [x] Conflicting metric statements across active docs are resolved.
- [x] Canonical docs link-check passes (all local links valid).

Exit criteria:

- [x] Canonical truth set is frozen and reused by both release and paper tracks.

---

## Phase 3 - Release Hardening (No New Evals)

Goal: cut a clean, reproducible release package from existing validated evidence.

- [x] Stabilize release branch from latest-work baseline (per Phase 0 strategy) via `release/v0.3-integrity-latest-work`.
- [x] Resolve stale/legacy docs clutter into archive with pointers, not deletions of evidence.
- [x] Finalize simple onboarding flow (install, run, minimal ingest/query, troubleshooting).
- [x] Prepare release notes only from canonical claims table.

Validation and tests:

- [x] Unit tests pass (`poetry run pytest tests/unit -v`).
- [x] Docker test path passes (`docker compose run --rm test`).
- [x] API smoke path passes (`/health`, minimal ingest, minimal query).
- [x] Repro instructions execute on a clean shell without ad-hoc steps.

Validation evidence (2026-02-17):

- `poetry run pytest tests/unit -v` -> `153 passed`.
- `docker compose run --rm --build test` -> `153 passed` (this was a temporary deterministic unit-only gate; full Docker profile validation later restored).
- `curl http://localhost:8000/health` -> `{"status":"ok"}`.
- Sequential smoke flow succeeded:
  - `POST /api/v1/users/release_smoke_user2` -> user created.
  - `POST /api/v1/users/release_smoke_user2/ingest` -> ingest success, memories created.
  - `POST /api/v1/users/release_smoke_user2/chat` -> completed response with recalled preference.
- Release-doc local link check across `README.md`, `docs/`, and `release_artifacts/` -> `42 files scanned, 0 missing local links`.

Full Docker profile recheck (2026-02-18):

- `docker compose run --rm test` with full `tests/` profile -> `211 passed, 8 skipped` in `179.07s`.
- Final re-verification run: `docker compose run --rm test` -> `211 passed, 8 skipped` in `190.61s`.
- Skips are expected and gated for seed-data/vector-data dependent coverage (`fitness_test_v2` not seeded in this runtime):
  - `tests/integration/test_retrieval.py` -> 6 skipped
  - `tests/integration/test_v2_intelligence.py` -> 2 skipped
- Phase 3 Docker gate is satisfied for the current reproducible profile with explicit skip rationale documented.

Release-branch stabilization receipt (2026-02-18):

- Created control branch from latest-work baseline: `git branch release/v0.3-integrity-latest-work refactor/v0.3-cognitive-memory`.
- Baseline parity verified: `git rev-list --left-right --count release/v0.3-integrity-latest-work...refactor/v0.3-cognitive-memory` -> `0 0`.
- Commit anchor parity verified: both branches resolve to `d71b290`.

Latest-work-first correction receipt (2026-02-18):

- Canonical execution model corrected to latest-work-first (`refactor/v0.3-cognitive-memory` + artifact/session timeline evidence).
- Correction plan captured in `docs/RELEASE_EXECUTION_CORRECTED_V03.md`.
- Source evidence includes:
  - `docs/PERSONAMEM_EVAL_CANONICAL.md`
  - `docs/PERSONAMEM_TWO_WEEK_REVIEW_20260210.md`
  - `docs/archive/eval/2026-02/generated/EVALUATION_TIMELINE_JAN27_FEB10.md`
  - `.opencode/release/RELEASE_PACKET.md`
  - `.sisyphus/plans/v03-release-execution.md`

Exit criteria:

- [x] Release candidate is technically clean and reproducible; claim wording is draft-safe pending the final claims audit in Phase 4.

---

## Phase 4 - Release Operations (Ship First)

Goal: ship the audited release first; defer paper drafting until after release tag.

- [ ] Create release PR from clean release branch.
- [ ] Run final claims audit on PR description and release notes.
- [ ] Merge and tag release.

Validation and tests:

- [ ] PR includes links to canonical claims table and methodology source.
- [ ] Release tag points to audited commit.
- [ ] Release notes retain safe-claim boundaries (including BEAM `event_ordering=0%` caveat).

Exit criteria:

- [ ] Release shipped from an audited commit with consistent claim set.

---

## Phase 5 - Paper Assembly and Submission (Post-Release)

Goal: produce and submit paper package after release is shipped, using frozen evidence only.

- [ ] Build paper outline and section skeleton from canonical docs.
- [ ] Populate result tables from claims table only.
- [ ] Include explicit limitations and rollback lessons learned.
- [ ] Add competitor framing (Graphiti/Mem0) with safe-claim boundaries.
- [ ] Add reproducibility appendix with artifact paths and environment details.
- [ ] Freeze paper version and submit to selected venue.

Validation and tests:

- [ ] Every claim in abstract/conclusion maps to an `audit-grade` row or is clearly labeled as `experimental`.
- [ ] No unsupported superiority claims appear.
- [ ] Methods section references one canonical methodology source.
- [ ] Evidence appendix paths are all resolvable.
- [ ] Submission package includes reproducibility appendix and limitations section.

Exit criteria:

- [ ] Paper draft is submission-ready and submitted with release-consistent claim set.

---

## Detours and Divergence Log (Recorded)

- [x] Baseline correction detour: mainline-first assumption replaced by latest-work-first execution (`refactor/v0.3-cognitive-memory`).
- [x] Validation detour: temporary unit-only Docker gate replaced by full-profile Docker validation with documented skip rationale.
- [x] Branch control detour: legacy `release/v0.3-integrity-mainline` superseded by `release/v0.3-integrity-latest-work` parity branch.
- [x] Claims wording detour: README benchmark narrative aligned to canonical claims table (A-001 through A-004), with BEAM caveat carried into release notes.
- [x] Governance wording detour: evidence-scope and phase-lock/revalidation semantics reconciled across checklist, corrected plan, and receipts.

---

## Enforcement Rules (Phase Movement and New Task Control)

- [ ] Move-on rule: when a phase exit criteria is met, immediately mark phase complete and start the next phase.
- [ ] No-reopen rule: completed phases stay closed unless a gate was satisfied under invalid assumptions, in which case only that gate is reopened with correction receipts.
- [ ] New-task rule: new tasks are allowed only if all are true:
  - [ ] task is phase-level critical,
  - [ ] task has explicit validation evidence,
  - [ ] task does not violate scope lock (no new eval runs admitted after checklist scope lock),
  - [ ] task does not conflict with canonical truth sources.
- [ ] If a task fails this gate, defer it to post-release backlog.

---

## Immediate Start (First Few Phases)

- [x] Execute Phase 0 completion items (control plane and branch strategy).
- [x] Execute Phase 1 evidence harvest from archives and hidden docs.
- [x] Execute Phase 2 canonical freeze before any release-note or paper drafting.
