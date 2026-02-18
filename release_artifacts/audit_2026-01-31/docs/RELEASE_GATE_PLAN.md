# Release Gate Plan (Truth-First)

**Date**: 2026-01-31  
**Objective**: Release with audit-grade claims, no benchmark inflation, and clear evidence trails.

---

## Guiding Principles (Non-Negotiable)

1. **Claims must be backed by artifacts** in `../memory-evals/results/` with summary JSON + deep logs.
2. **No adjusted accuracy** claims without a committed audit artifact (`gold_audit.jsonl`).
3. **No partial/short-run results** in headline numbers.
4. **All benchmark claims must specify scope** (e.g., BEAM 10 abilities vs 2 abilities).

---

## Current State (Audit-Grade)

- PersonaMem (subset, 150Q): 65.3%
- PersonaMem (full, 589Q, 1 seed): 66.2%
- BEAM (10 abilities, 100Q, 1 seed): 69.0%, event_ordering = 0%

Source of truth: `docs/BENCHMARK_TRUTH_TABLE.md`

---

## Release Gating Criteria

### Gate A: Claims Integrity
- [ ] All public claims appear in `docs/BENCHMARK_TRUTH_TABLE.md`.
- [ ] Every claim has a `summary.json` / per-seed JSON + `deep_logs.jsonl`.
- [ ] Every run includes `manifest.json` with benchmark, model, dataset version, git commit, seeds, samples.

### Gate B: Benchmark Coverage
- [ ] PersonaMem full benchmark run (589Q) completed across ≥3 seeds.
- [ ] BEAM full (10 abilities) completed across ≥5 seeds.

### Gate C: Known Failure Mode Fixes
- [ ] Event ordering > 40% on BEAM-10 (tooling/output fix).
- [ ] Integration agent creates links (links_created > 0 in eval).
- [ ] Answer selection improves without global prompt constraints.

---

## Execution Plan

### Phase 1: Artifact Hygiene (Immediate)
1. Create `manifest.json` schema and write it for every future run.
2. Regenerate `docs/BENCHMARK_TRUTH_TABLE.md` after each run.

### Phase 2: Event Ordering Fix (High Priority)
1. Add ordering tool that accepts explicit candidate events and returns ordered list with timestamps.
2. Validate on BEAM event_ordering subset before full BEAM run.
3. Re-run BEAM-10 with ≥5 seeds.

### Phase 3: Answer Selection Improvements (High Impact)
1. Test minimal prompt additions that prefer evidence-backed personalized responses.
2. Validate on PersonaMem subset (150Q) before full run.
3. Compare against baseline with statistical tests (paired where possible).

### Phase 4: Full Re-runs (Release-Grade)
1. PersonaMem full benchmark ×3 seeds.
2. BEAM-10 ×5–10 seeds.
3. Commit all artifacts + manifests.

---

## Strict Release Policy

- If any gate fails, **release is blocked**.
- If any claim lacks artifacts, **claim is removed**.
- If any run is partial or mismatched, **mark as exploratory** only.

---

## Deliverables Before Release

- `docs/BENCHMARK_TRUTH_TABLE.md` (updated)
- `docs/EVALUATION_RESULTS_2026-01.md` (audited)
- `docs/EVALUATION_RESULTS.md` (snapshot)
- `manifest.json` per run
- `gold_audit.jsonl` if adjusted accuracy is claimed

---

## Long-Term (Impact)

To win the “cross‑web AI memory” category:
- Memory must be **auditable**, **user‑governed**, and **temporally reliable**.
- Preference inference needs to be consistent and reversible.
- Event ordering must be robust under ambiguity and tie cases.

---

*This plan is intentionally strict. It is designed to survive HN-level scrutiny.*
