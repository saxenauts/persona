# ULW Research Brief: Variance, Tool Loops, and Path to >85%

Date: 2026-02-08
Scope: non-execution synthesis only (evidence review, no architecture changes in this brief)

## 1) Evidence-grounded root-cause model

### A. The 50 vs 80 split is not a pure stability signal

- `run_20260207_193016` (50%) and `run_20260207_202510` (80%) used disjoint question IDs (0 overlap), so sample difficulty changed between runs.
- Historical priors for those exact IDs from prior deep logs:
  - 50% run set expected accuracy: ~0.497
  - 80% run set expected accuracy: ~0.671
- Interpretation: reported spread is partly sampling variance and partly model behavior.

### B. Why one tool call can look confident

- PersonaMem prompt format in eval is strict MCQ (`Question + Options + answer letter only`) via `mem_eval/runner.py`.
- With one `recall`, the model gets semantically strong snippets and is forced to map to one option immediately.
- This creates high local confidence even when options are close paraphrases or framing differs.

### C. Why multiple tool calls often degrade

- Across recent logged rows with `persona_context.tool_calls_made`, accuracy trends down after first call in many runs (especially 3+).
- Mechanistic explanation in current architecture:
  1. additional calls often happen when first evidence is ambiguous/noisy,
  2. subsequent recalls increase entity-heavy competing evidence,
  3. MCQ selection drifts under over-complete or conflicting context,
  4. budget-block feedback in loop can perturb final selection behavior.

### D. Retrieval quality issue is composition, not just score

- Failed samples often have high top recall scores and non-empty context.
- Differentiator is evidence composition: higher entity share and weaker decisive psyche/episode signal correlate with wrong option picks.
- This aligns with prior bloat findings (`.opencode/MEMORY_BLOAT_REPORT.md`): entity-heavy retrieval is a persistent pattern.

### E. Benchmark label/framing sensitivity is real

- Several sampled failure items show retrieved evidence that supports one plausible option while benchmark key prefers another near-paraphrase framing.
- Effect: confidence without correctness when choice hinges on stylistic nuance.

## 2) Self-critique of initial hypothesis

Initial hypothesis: "variance is mostly seed/sample effect."  
Revised hypothesis: seed/sample effect is necessary but not sufficient. The dominant controllable factors are:

1. MCQ forcing after one evidence pass,
2. retrieval composition skew (entity-heavy, not decisively preference-grounded),
3. multi-call context drift under ambiguity.

## 3) Keep / Change / Defer by subsystem

### Ingestion
- Keep: 4-pillar model and LLM-first extraction contract.
- Change: strengthen preference/sentiment signal capture for recurrent behavior patterns (without hard keyword routers).
- Defer: heavy schema expansion beyond current memory types.

### Integration
- Keep: contradiction and temporal guardrails already added.
- Change: improve confidence-aware linking between episodes and psyche for preference evolution chains.
- Defer: broad graph ontology expansion.

### Consolidation
- Keep: user card + memeplex refresh pipeline.
- Change: add stronger inferred psyche quality loop (clear provenance and recency weighting in synthesis text).
- Defer: full asynchronous dreaming layer until online metrics are stable.

### Retrieval
- Keep: single-call bias as default for cost and stability.
- Change: composition-aware rerank objective favoring decisive psyche/episode evidence over entity duplicates.
- Defer: introducing many new retrieval tools.

### Tool loop
- Keep: bounded execution and budget controls.
- Change: robust second-call policy only when first-call evidence confidence is low, contradictory, or option-separability is weak.
- Defer: unconstrained multi-call exploration.

## 4) Strategy options to push >85% (benchmark-agnostic core)

### Option A: Confidence-gated two-pass answering (highest leverage)

Core idea: keep one-call default, but perform a deterministic second pass only when option separability is low.

- Pass 1: retrieve + draft candidate option with evidence traces.
- Confidence gate: if top two options are too close or evidence contradictions detected, run targeted second recall.
- Pass 2: resolve ambiguity with constrained remap to option letter.

Why this can cross 85%:
- Preserves low-noise one-call success regime.
- Activates extra retrieval only for ambiguous cases where current system fails.

### Option B: Retrieval composition control (psyche/episode-first tie-break)

Core idea: for preference-style questions, require at least one decisive psyche or high-evidence episode in top hits before final answer.

- Soft quota in rerank, not hard filtering.
- Penalize near-duplicate entities with low reuse utility.

Why this can cross 85%:
- Reduces wrong confident picks from entity-only context.

### Option C: Eval-side calibration layer (strictly eval-only)

Core idea: preserve production behavior; add evaluator-side option calibration protocol.

- Compare generated rationale against each option for semantic entailment.
- Select option with strongest entailment under deterministic rules.

Why this can cross 85%:
- Handles near-paraphrase option collisions without changing core product logic.

## 5) Phased non-execution roadmap

### Phase R1: Measurement hardening (no behavior changes)

Goal: eliminate ambiguity in interpretation of runs.

- Use fixed matched question set for all comparisons.
- Report per-question historical difficulty prior.
- Add confidence/separability diagnostics per question.

Success criteria:
- CI shrinkage from sampling control,
- reproducible before/after analysis.

### Phase R2: Ambiguity diagnostics

Goal: identify exactly where one-call confidence fails.

- Label each miss as: retrieval miss, composition skew, option-framing mismatch, or contradiction unresolved.
- Quantify contribution per bucket.

Success criteria:
- >=90% failures categorized with evidence.

### Phase R3: Option A simulation (offline)

Goal: estimate gain from confidence-gated second pass before implementation.

- Replay logs and simulate gate triggers.
- Estimate potential uplift and extra token cost.

Success criteria:
- projected >=+8 to +12 points on hard subset with bounded cost.

### Phase R4: Strategy stack selection

Goal: choose minimum-change route to >85.

- Compare projected lift for A, B, C and combinations.
- Select smallest stack meeting target with stable variance.

Success criteria:
- selected stack with clear implementation contract and risk bounds.

## 6) Experiment matrix (non-execution design)

| Exp ID | Hypothesis | Design | Primary Metric | Guardrail |
|---|---|---|---|---|
| X1 | One-call is optimal only for high separability cases | Stratify by option separability and compare 1-call vs 2-call simulated policy | Accuracy by separability bin | Avg tool calls <= 1.4 |
| X2 | Entity-heavy top hits cause wrong confident picks | Replay with composition-aware rerank (psyche/episode tie-break) | Accuracy on recall_user_shared_facts | No drop on easy subset |
| X3 | Option framing mismatch causes avoidable misses | Eval-side entailment remap over existing rationale | Delta accuracy without new retrieval | Deterministic remap reproducibility |
| X4 | Variance mostly sample-driven below 20Q | Fixed-ID repeated seeds vs random-ID seeds | CI width | Same model/config |
| X5 | Temporal failures need interval certainty, not more calls | Apply date-consistency checks on temporal subset | Temporal accuracy | No hallucination increase |

## 7) Risks

- Overfitting to PersonaMem letter mapping instead of general memory quality.
- Extra second-pass calls can reintroduce drift if not tightly gated.
- Composition controls that are too strong may suppress useful entity cues.

## 8) Evidence references

- `../memory-evals/results/score_plan_20260207/run_20260207_193016/deep_logs.jsonl`
- `../memory-evals/results/score_plan_20260207/run_20260207_202510/deep_logs.jsonl`
- `../memory-evals/results/score_plan_20260207/run_20260207_193016/summary.json`
- `../memory-evals/results/score_plan_20260207/run_20260207_202510/summary.json`
- `.opencode/MEMORY_BLOAT_REPORT.md`
- `.opencode/BEAM100Q_POSTMORTEM_AND_PLAN.md`
- `.opencode/notes/ulw_audit_results_2026-02-05.md`
- `.opencode/notes/ulw_failure_patterns_2026-02-05.md`
