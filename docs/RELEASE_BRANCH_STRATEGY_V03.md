# Persona v0.3 Release Branch Strategy

Last updated: 2026-02-18 (America/Los_Angeles)

## Objective

Move from a stale, dirty working branch to a clean release control flow without losing historical work.

## Current State

- Current local branch: `refactor/v0.3-cognitive-memory`
- Divergence vs `origin/main`: `157` ahead, `1` behind
- Working tree: dirty (many modified/deleted/untracked files)
- GitHub default branch: `main`
- Branch protection on `main`: not configured (API 404 branch not protected)

## Control Decision

- Canonical source branch for release work: `refactor/v0.3-cognitive-memory` (latest-work-first model).
- Existing branch state is treated as the active v0.3 source of truth, with curation and validation gates before release packaging.
- Stabilized release-control branch: `release/v0.3-integrity-latest-work` anchored to `refactor/v0.3-cognitive-memory` commit `d71b290`.
- No force pushes to `main`.

## Execution Plan

### Step A - Preserve Current State (No Data Loss)

- Keep current branch intact as historical workspace.
- Produce receipts for:
  - branch divergence,
  - critical modified paths,
  - recent commit anchors.

### Step B - Start Clean Release Track

- Curate a release candidate branch from `refactor/v0.3-cognitive-memory` plus validated artifact/session evidence.
- Active control branch for this step is `release/v0.3-integrity-latest-work` (parity check `0 0` vs canonical baseline).
- Preserve latest-work code + docs that are evidence-backed; remove only proven noise.
- Validate claim integrity against canonical audit and paired-run artifacts before PR cut.

### Step C - Merge Discipline

- Use PR-only merge to `main`.
- Enforce claim-audit checks before PR approval.
- Tag release from audited commit only.

## Phase-Gated Change Rules

- Phase 0-2: docs/evidence/control-plane changes only.
- Phase 3+: release hardening and packaging, still no new eval runs.
- Release sequencing rule: ship audited release first; paper drafting/submission is post-release.
- Completed phases are not reopened.

## Validation Checks

- `git fetch origin --prune`
- `git rev-list --left-right --count HEAD...origin/main`
- `gh repo view saxenauts/persona --json nameWithOwner,defaultBranchRef`
- `gh api repos/saxenauts/persona/branches/main/protection` (record protected/unprotected status)

## Exit Criteria

- A documented clean-branch strategy exists and is referenced by the phase checklist.
- Source-of-truth branch for release work is explicitly defined as `refactor/v0.3-cognitive-memory` with artifact/session receipts.
