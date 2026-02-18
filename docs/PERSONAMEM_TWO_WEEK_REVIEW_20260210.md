# PersonaMem Two-Week Review (Jan 27 to Feb 10)

**Last Updated**: 2026-02-10  
**Linked Canonical Doc**: `docs/PERSONAMEM_EVAL_CANONICAL.md`  
**Review Scope**: code changes, eval runs, decision traces, reversals, and documentation sprawl over the last two weeks.

---

## 1) Review Objective

This review answers four questions with evidence:

1. What changed (including small tweaks and reversions)?
2. Why each change was made (decision rationale and source trace)?
3. How far we are from the original v0.3 vision and release gates?
4. What excess exists now, and what cleanup plan should be executed next?

---

## 2) Source-Of-Truth Rules Used In This Review

Evidence confidence levels:

- **Tier 1 (Verified)**: backed by run artifacts and/or git commits (`summary.json`, `deep_logs.jsonl`, commit hash, file path).
- **Tier 2 (Inferred)**: analysis statements that reference evidence but are not directly materialized in run summary artifacts.
- **Tier 3 (Proposed)**: architecture/design proposals not implemented.

Primary audit anchors:

- `release_artifacts/audit_2026-01-31/docs/BENCHMARK_TRUTH_TABLE.md`
- `release_artifacts/audit_2026-01-31/docs/RELEASE_GATE_PLAN.md`
- `docs/PERSONAMEM_EVAL_CANONICAL.md`
- `docs/archive/eval/2026-02/EVIDENCE_MAP_20260209.md`

---

## 3) Original Vision And Release Intent (Baseline)

### Vision-level intent

From `docs/vision/ai-memory-vision-2026-2041.md`: memory as identity layer, graph-structured temporal reasoning, continuous evolution, and durable behavior across model upgrades.

### v0.3 execution intent

From `.sisyphus/plans/v03-release-execution.md` and `release_artifacts/audit_2026-01-31/docs/RELEASE_GATE_PLAN.md`:

- Truth-first claims only.
- PersonaMem target >=70, BEAM target >=65.
- No inflated or artifact-free benchmark claims.
- Release gates must be explicit and blocking when unmet.

---

## 4) Two-Week Change Timeline (Including Tiny Tweaks)

This timeline is commit-backed (Tier 1).

| Date | Change | Why (stated/implied) | Evidence |
|---|---|---|---|
| 2026-01-28 | `fix(eval): add as_of parameter to memeplex for eval compatibility` | Fix eval-time mismatch for memeplex context | `50d13535...` |
| 2026-01-28 | `fix(consolidation): timezone comparison + datetime serialization` | Repair temporal/consolidation correctness | `bfdabe21...` |
| 2026-01-28 | `fix(memeplex): model_dump(mode='json')` | Stabilize datetime serialization to storage | `b6b6113c...` |
| 2026-01-30 | `feat: complete truth-first baseline with verified 65.3% PersonaMem` | Move from claim drift to verified baseline framing | `a605c5cd...` |
| 2026-01-30 | `fix(prompts): improve answer selection` then `fix(prompts): revert benchmark hacks` | Recover from overfitting and restore generality | `967d650f...`, `e463cb60...` |
| 2026-01-30 | Psyche extraction and consolidation updates + H1-H5 fixes | Improve inference fidelity and temporal/date correctness | `e3df2f0a...`, `106bfc37...`, `92cf37c4...`, `09bfe205...`, `4b05fae8...` |
| 2026-01-30 | Repro package and benchmark doc updates | Make claims reproducible and auditable | `e944f9cb...`, `b5e563d7...` |
| 2026-01-31 | Prompt/schema hardening for BEAM ordering, then full rollback | Experiment regressed hard; reverted to baseline | `c25022cb...`, `5753a2dc...`, `56b8aca2...`, `2667fa1b...`, then reverts `600b76a8...`, `c281c9a8...`, `d457dd75...`, `1a9ac707...` |
| 2026-02-04 | Retrieval/timezone/service/integration/prompt/document cleanup pass | Remove stale debt and align with LLM-first + truth-first docs | `d9f601d0...`, `caac7893...`, `3c282689...`, `655026f5...`, `e3cdcbf0...`, `efe31ef2...` |
| 2026-02-08 to 2026-02-09 | ULW paired E1 implementation and 50Q/100Q paired A/B | Test confidence-gated second pass under fixed-ID protocol | See Section 6 artifacts |

---

## 5) Decision Ledger (Why, Evidence, Outcome)

| Decision | Why | Evidence | Outcome |
|---|---|---|---|
| Enforce truth-first claim policy | Prior 80%+ narratives lacked durable artifacts | `release_artifacts/audit_2026-01-31/docs/BENCHMARK_TRUTH_TABLE.md`, `.../EVALUATION_RESULTS_2026-01.md` | Adopted as canonical policy |
| Revert BEAM prompt hardening wave | Global output constraints caused severe regression | `.sisyphus/reports/beam-accuracy-improvement-codex-review.md` + revert commits | Reverted fully (correct call) |
| Keep 1-call default and gate second pass | Multi-call drift seen in aggregates; targeted ambiguity handling expected safer | `docs/archive/eval/2026-02/ULW_85_EXECUTION_FRAMEWORK_20260208.md` | Provisional winner (+4pp paired) |
| Promote paired fixed-ID methodology | Random slices produced misleading variance | `docs/archive/eval/2026-02/ULW_RESEARCH_BRIEF_20260208.md` | Applied in 50Q + 100Q paired tests |
| Classify failures with source forensics | "Ambiguous" bucket was hiding retrieval insufficiency | `docs/archive/eval/2026-02/FAILURE_ANALYSIS_20260209.md` | Root-cause map became actionable |
| Archive overlapping eval docs under one canonical root | Active docs had mixed-era contradictions | `docs/PERSONAMEM_EVAL_CANONICAL.md` + archived set in `docs/archive/eval/2026-02/` | Reduced top-level doc sprawl |

---

## 6) What We Achieved In The Last Two Weeks (Evidence-Backed)

### A) Governance and evidence discipline

- Single canonical status file created: `docs/PERSONAMEM_EVAL_CANONICAL.md`.
- Audit-grade baseline preserved: PersonaMem 65.3% subset, 66.2% full (single seed), BEAM 69.0 (event_ordering 0).
- Experimental runs explicitly labeled non-headline until promotion criteria are met.

### B) Runtime and eval interventions executed

- Confidence-gated second pass added and measured under paired protocol.
- 50Q paired: 64.0 -> 68.0 (+4pp), McNemar p=0.6875.
- 100Q paired: 63.0 -> 67.0 (+4pp), McNemar p=0.34375.
- No tool inflation observed in paired comparisons.

### C) Forensics quality upgrade

- Failure analysis moved from coarse labels to case-level source evidence checks.
- Temporal evolution misses isolated as a high-impact deterministic failure mode.
- `suggest_new_ideas` shown as critical weak type in current slice.

Primary artifact refs:

- `../memory-evals/results/run_20260208_154926/summary.json`
- `../memory-evals/results/run_20260208_212320/summary.json`
- `../memory-evals/results/ulw_paired100_20260209_armA/run_20260209_133301/summary.json`
- `../memory-evals/results/ulw_paired100_20260209_armB/run_20260209_035700/summary.json`
- `docs/archive/eval/2026-02/FAILURE_ANALYSIS_20260209.md`

---

## 7) Gap-To-Vision Matrix (Where We Stand Now)

| Original vision/goal | Current state | Gap level | Next required move |
|---|---|---|---|
| Truth-first, auditable claims | Implemented and enforced in canonical docs | Low | Keep strict update protocol |
| PersonaMem >=70 for release narrative | Current paired evidence sits in 63-67 band | Medium/High | Run additional paired 100Q seed and then broader lift experiments |
| BEAM >=65 with robust ordering | Historical 69 exists but event ordering fragility and regressions documented | High | Focused BEAM ordering/tool-output contract path before claim elevation |
| Dynamic identity over time (temporal evolution handling) | Deterministic misses still present in analyzed cases | High | Consolidation + retrieval evolution chain handling |
| Graph/link model used in runtime reasoning | Tools exist but low usage in eval traces | High | Controlled tool-policy changes and measured link traversal experiments |

---

## 8) What We Had To Do To Reach Current State

1. Retire non-auditable benchmark narratives and lock claims to artifact-backed truth tables.
2. Reverse harmful benchmark-overfit prompt changes (full rollback discipline).
3. Shift from random-slice intuition to fixed-ID paired experimentation.
4. Build deeper failure forensics using source conversation evidence, not shallow label assumptions.
5. Consolidate documentation into one canonical status + archive layers to reduce operator confusion.

---

## 9) Excess Inventory (Current)

### Excess pattern summary

- Multiple overlapping status summaries across `.sisyphus/`, `.opencode/`, and archived docs.
- Mixed-era claims (especially adjusted/high-score narratives) coexisting with stricter audit-grade baselines.
- Repeated "session complete" style documents with near-duplicate content.

### Current keep set (active)

- `docs/PERSONAMEM_EVAL_CANONICAL.md` (top-line source of truth)
- `docs/PERSONAMEM_TWO_WEEK_REVIEW_20260210.md` (this detailed review)
- `release_artifacts/audit_2026-01-31/docs/BENCHMARK_TRUTH_TABLE.md`
- `release_artifacts/audit_2026-01-31/docs/RELEASE_GATE_PLAN.md`
- `docs/EVAL_METHODOLOGY_V03.md`

### Current archive set (drill-down only)

- `docs/archive/eval/2026-02/*`
- `.sisyphus/notepads/*`
- `.opencode/archive/*`

---

## 10) Excess-Removal Plan (Planned, Not Executed Yet)

### Phase C1: De-duplicate active narrative sources

- Keep only 2 active docs in `docs/` for eval status/review:
  - canonical status,
  - two-week review.
- Move any newly created overlapping status docs to `docs/archive/eval/` immediately after extraction.

### Phase C2: Collapse session-noise in `.sisyphus/notepads/`

- Create one index file per notepad stream with links to the top 3 evidence-rich docs.
- Move repetitive "status" and "completion" notes into dated archive subfolders.

### Phase C3: Freeze mixed-era claims

- Mark legacy high-claim docs as `historical_noncanonical` inside title/header blocks.
- Add explicit pointers to canonical file for current truth.

### Phase C4: Enforce creation rule

- New eval-status docs can only be created if they add new run artifacts not already represented in canonical/review docs.

Success criteria for cleanup:

- Active eval status docs in `docs/` stay <=2.
- Every external claim maps to truth-table entry + artifact path.
- No mixed-era claim appears in active docs without confidence labeling.

---

## 11) Next Work Plan (Evidence-Driven)

1. Complete additional paired 100Q seed (`seed=123`) under same fixed-ID protocol.
2. Recompute paired metrics and promotion decision for second-pass gate.
3. Execute next ranked lever from framework (world-model budget or composition steering), still under paired protocol.
4. Run targeted BEAM recovery path focused on ordering/tool contract, avoiding global prompt constraints.
5. Execute cleanup phases C1-C4 with receipts added to this review file.

---

## 12) Confidence Notes

- **Tier 1 confidence (high)**: commit timeline, run summaries, paired deltas, release/audit docs.
- **Tier 2 confidence (medium)**: deeper causal interpretations from manual forensics analyses.
- **Tier 3 confidence (proposal only)**: architecture extensions such as memeplex attractor redesign and settle-style retrieval tooling.

This review intentionally separates these tiers so strategic decisions are made on verified evidence first.
