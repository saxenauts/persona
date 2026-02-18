# Release Control Receipts (Phase 0)

Last updated: 2026-02-18 (America/Los_Angeles)

## GitHub CLI Setup

- Command: `gh --version`
- Result: `gh version 2.60.0`

- Command: `gh auth status`
- Result: logged in as `saxenauts`, active account `true`, scopes include `repo` and `workflow`.

## Repo and Branch Control

- Command: `git fetch origin --prune`
- Result: fetch succeeded; `origin/main` advanced (`8e886c9 -> 1c4d26a`).

- Command: `git rev-list --left-right --count HEAD...origin/main`
- Result: `157 1`
- Interpretation: current branch is 157 commits ahead and 1 commit behind `origin/main`.

- Command: `gh repo view saxenauts/persona --json nameWithOwner,defaultBranchRef`
- Result: `nameWithOwner=saxenauts/persona`, `defaultBranchRef.name=main`.

- Command: `gh api repos/saxenauts/persona/branches/main/protection`
- Result: HTTP 404 `Branch not protected`.

## Working Tree Risk Snapshot

- Command: `git status --short --branch`
- Result: branch is heavily dirty with wide modification/untracked footprint across code + docs + hidden folders.
- Interpretation (latest-work-first): this is the active v0.3 workstream, so release prep must curate and validate this state rather than discarding it in favor of `origin/main`.

## Latest-Work-First Receipts

- Command: `git worktree list --porcelain`
- Result: primary worktree on `refactor/v0.3-cognitive-memory`; parallel release worktree exists on `release/v0.3-integrity-mainline`.

- Command: `git log --all --decorate --date=iso --pretty=format:'%h|%ad|%d|%s' -60`
- Result: latest commit activity for active workstream is on `refactor/v0.3-cognitive-memory` (through `d71b290`).

- Command: `git reflog --date=iso -60`
- Result: confirms dense local evolution on `refactor/v0.3-cognitive-memory` through Feb 4 and subsequent artifact-backed work captured in docs/eval logs.

- Artifact evidence checked:
  - `docs/PERSONAMEM_EVAL_CANONICAL.md`
  - `docs/PERSONAMEM_TWO_WEEK_REVIEW_20260210.md`
  - `docs/archive/eval/2026-02/generated/EVALUATION_TIMELINE_JAN27_FEB10.md`
  - `.opencode/release/RELEASE_PACKET.md`
  - `.sisyphus/plans/v03-release-execution.md`

## Scope Lock Receipt

- Program rule recorded: no new benchmark/eval runs.
- Evidence source rule recorded: use existing artifacts/docs, including hidden `.sisyphus` and `.opencode` folders plus `docs/archive` and `release_artifacts`.
- `../memory-evals/results/` may be used as corroborative historical context, but in-repo artifacts remain the minimum reproducible release evidence set.

## Phase 3 Release-Branch Stabilization Receipt

- Command: `git branch release/v0.3-integrity-latest-work refactor/v0.3-cognitive-memory`
- Result: local release-control branch created from canonical latest-work baseline.

- Command: `git rev-parse --short refactor/v0.3-cognitive-memory && git rev-parse --short release/v0.3-integrity-latest-work`
- Result: `d71b290` and `d71b290`.

- Command: `git rev-list --left-right --count release/v0.3-integrity-latest-work...refactor/v0.3-cognitive-memory`
- Result: `0 0` (exact baseline parity).

- Note: legacy `release/v0.3-integrity-mainline` remains as historical branch context; latest-work control execution now tracks `release/v0.3-integrity-latest-work`.

## Detour and Divergence Receipts

- Sequencing detour captured: release-first execution with paper deferred to post-release phase.
- Validation detour captured: temporary unit-only Docker gate superseded by full-profile gate evidence (`211 passed, 8 skipped`).
- Claims detour captured: README benchmark wording aligned to canonical claims table IDs (A-001 through A-004).
- Governance detour captured: evidence-scope and phase-lock/revalidation wording normalized across active control docs.
