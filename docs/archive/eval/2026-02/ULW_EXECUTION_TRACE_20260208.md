# ULW Execution Trace - 2026-02-08

## Objective
Execute an end-to-end path toward >85% PersonaMem with evidence-backed interventions, continuous logging, and reproducible verification.

## Baseline Evidence Snapshot
- Internal aggregate over PersonaMem deep logs shows 1-call outperforms 2+ calls (`63.96%` vs `58.16%`).
- Score-plan runs confirm instability on tiny slices and show residual error after 1-call constraints.
- Memory composition remains entity-heavy in many failures; option-mapping fragility persists.

## Execution Plan (Active)
1. Implement deterministic eval-side MCQ calibration and diagnostics.
2. Implement runtime composition-aware recall shaping and confidence-gated second pass.
3. Run diagnostics, unit tests, type/build checks, and targeted eval runs.
4. Update this trace with observations, metrics, and decisions.

## Notes (Live)
- 2026-02-08T16:00:00-08:00: Started implementation phase with doc-first trace.
- 2026-02-08T16:18:00-08:00: Implemented eval-side deterministic MCQ calibration with diagnostics in `memory-evals-repo/mem_eval/runner.py` and schema extensions in `memory-evals-repo/mem_eval/logging/log_schema.py`.
- 2026-02-08T16:24:00-08:00: Implemented composition-aware recall shaping in `persona/tools/memory.py` and recall telemetry/confidence-stop signals in `persona/tools/runner.py`.
- 2026-02-08T16:28:00-08:00: Implemented MCQ low-confidence second-pass gating in `persona/services/persona_service.py`; prompt guidance updated in `persona/llm/prompts.py`.
- 2026-02-08T16:35:00-08:00: Verification pass complete for runtime code: `pytest` targeted set passed (30/30), `poetry build` succeeded, `compileall` succeeded for all modified Python files.
- 2026-02-08T16:45:00-08:00: Launched benchmark runs from canonical `../memory-evals`; resumed interrupted run and captured partial metrics. Clean run with 4 samples completed at 50% (2/4). This confirms pipeline execution, but sample is too small for directional claims.

## Verification Artifacts
- Unit tests: `poetry run pytest tests/unit/test_tools_runner.py tests/unit/test_memory_tools.py tests/unit/test_services.py tests/unit/test_prompts.py -q` -> `30 passed`.
- Eval-framework tests: `poetry run pytest memory-evals-repo/mem_eval/tests/test_metrics.py memory-evals-repo/mem_eval/tests/test_benchmarks.py -q` -> `31 passed, 7 skipped`.
- Build: `poetry build` -> success.
- Syntax: `python -m compileall` on all modified Python files -> success.

## Bench Execution Notes
- Canonical run (10 sample, seed 42) in `../memory-evals`:
  - initial run timed out mid-way due ingestion cost per sample.
  - resumed run id `20260208_032644` completed remaining work; checkpoint indicates 9 completed with 1 previously crashed item skipped.
  - summary artifact: `../memory-evals/results/ulw_exec_20260208/run_20260208_032644/summary.json` (`5/9`, 55.56%).
- Clean completion run (4 sample, seed 42):
  - artifact: `../memory-evals/results/ulw_exec_20260208_clean4/run_20260208_041738/summary.json` (`2/4`, 50%).

## Observations
- Full end-to-end benchmark execution is constrained primarily by ingest runtime (roughly 4.5 to 6 minutes per sample on this environment).
- Small-N runs remain variance-heavy and are unsuitable for go/no-go on the 85% target; fixed-ID larger runs remain required.

## Morning Continuation (2026-02-08)
- Ported eval-calibration changes into canonical repo used by `uv run mem-eval`: `../memory-evals/mem_eval/runner.py`, `../memory-evals/mem_eval/logging/log_schema.py`.
- Verified syntax for canonical changes with `python -m compileall mem_eval/runner.py mem_eval/logging/log_schema.py`.
- Executed paired seeds in canonical repo:
  - `../memory-evals/results/ulw_exec_v2_20260208/run_20260208_105312/summary.json` -> 50% (2/4)
  - `../memory-evals/results/ulw_exec_v2_20260208/run_20260208_112045/summary.json` -> 50% (2/4)
- Confirmed new diagnostics are present in deep logs:
  - retrieval keys now include `confidence_signals` and `calibration`
  - example calibration payload includes `method`, `predicted_option`, `separation_margin`, `ambiguous`

## Current Assessment
- Instrumentation and control surfaces are now active in both persona runtime and canonical eval runner.
- Accuracy has not moved on tiny samples yet; no claim of uplift is made.
- Next required step remains larger fixed-ID paired runs (>=50 questions) to tune thresholds and estimate true lift.

## Analysis Update (Signal vs Noise)
- Aggregated `personamem` judged rows with populated persona context and recall composition (`n=5940`) show:
  - `tool_calls=1` -> `64.80%` vs `tool_calls>=2` -> `58.16%` (delta `-6.64pp` for multi-call).
  - `world_model_chars > 3000` -> `43.60%` vs `<=3000` -> `64.94%` (delta `-21.34pp`).
  - `psy_share > 0` in retrieved recall items -> `56.57%` vs `67.28%` when `psy_share=0` (delta `-10.71pp`).
  - `entity_share <= 0.20` -> `50.35%` vs `67.19%` otherwise (delta `-16.85pp`), indicating too-little entity signal is also harmful.
- Dominant question type (`recall_user_shared_facts`, `n=3531`) preserves the same pattern:
  - `tool_calls>=2`: `58.82%` vs `63.11%`
  - `world_model_chars>3000`: `41.67%` vs `63.88%`
  - `psy_share>0`: `57.18%` vs `66.59%`
- Wilson 95% intervals for key bins:
  - `tool_calls=1`: `0.6265-0.6516`; `tool_calls>=2`: `0.5340-0.6276`
  - `world_model_chars>3000`: `0.3620-0.5060` (consistently poor zone)

## Oracle Guidance (Decision Layer)
- 85% is unlikely from generic complexity increase; prioritize causal levers with fixed-ID paired tests.
- Ranked highest-probability path:
  1. freeze question IDs and run paired measurement first,
  2. suppress psyche-heavy retrieval for fact-recall style prompts,
  3. keep 1-call default and only gated second pass,
  4. tighten composition control,
  5. treat eval-side option calibration as benchmark-lift lever and measure separately from product-lift.
- Kill criteria emphasized:
  - if avg tool calls rises >1.2-1.4 without net gain, revert;
  - if paired fixed-ID uplift <+1pp for a lever, drop it;
  - do not trust tiny-N swings.

## Stratified Check (Confounding Guard)
- Type-stratified analysis (rows with full persona context + recall composition) confirms global patterns are not only cross-type artifacts:
  - `recall_user_shared_facts` (`n=3531`):
    - `world_model_chars>3000`: `41.67%` vs `63.88%`
    - `psy_share>0`: `57.18%` vs `66.59%`
    - `tool_calls>=2`: `58.82%` vs `63.11%`
  - `track_full_preference_evolution` (`n=538`):
    - `psy_share>0`: `58.14%` vs `67.07%`
    - `tool_calls>=2`: `58.60%` vs `70.45%`
  - `provide_preference_aligned_recommendations` (`n=363`):
    - `psy_share>0`: `54.39%` vs `77.45%`
    - `tool_calls>=2`: `16.67%` vs `74.79%` (small n in multi-call bucket but direction is strongly negative)
- Interpretation: psyche-heavy retrieved context behaves as broad noise in current MCQ-style scoring setup; 1-call default remains robust.

## Deliverable Added
- Added consolidated execution/measurement blueprint: `docs/ULW_85_EXECUTION_FRAMEWORK_20260208.md`.
- Blueprint includes fixed-ID protocol, intervention ranking, experiment matrix (E1-E5), and go/no-go gates for 50Q -> 100Q promotion.

## Paired 50Q Experiment (E1) - Completed
- Added runtime switch for second-pass gating in `persona/services/persona_service.py`:
  - `PERSONA_ENABLE_MCQ_SECOND_PASS` (default true)
  - `PERSONA_MCQ_LOW_CONFIDENCE_THRESHOLD` (default 0.78)
- Validation:
  - `poetry run pytest tests/unit/test_services.py -q` -> `5 passed`.

- Arm A (second pass disabled):
  - command: `PERSONA_ENABLE_MCQ_SECOND_PASS=false uv run mem-eval run --config mem_eval/configs/persona_personamem_quick.yaml --workers 2`
  - run: `../memory-evals/results/run_20260208_154926`
  - full-set accuracy: `64.00%` (`32/50`)

- Arm B (second pass enabled):
  - command: `PERSONA_ENABLE_MCQ_SECOND_PASS=true uv run mem-eval run --config mem_eval/configs/persona_personamem_quick.yaml --workers 2`
  - run: `../memory-evals/results/run_20260208_212320`
  - recovered one checkpoint-marked started question (`personamem_32k_257`) to complete 50/50 comparison.
  - full-set accuracy: `68.00%` (`34/50`)

- Paired comparison on identical 50 IDs:
  - Delta: `+4.00pp` (B - A)
  - Discordant pairs: gain `4`, loss `2`
  - McNemar exact p-value: `0.6875` (not statistically significant at this sample size)
  - Wilson 95% CI:
    - Arm A: `0.5014 - 0.7586`
    - Arm B: `0.5419 - 0.7924`

- Diagnostic movement (paired IDs):
  - mean tool calls: `1.04 -> 1.02`
  - no explosion in retrieval depth; effect is small and uncertain.

- Discordant question IDs (for targeted error analysis):
  - gains: `personamem_32k_103`, `personamem_32k_257`, `personamem_32k_267`, `personamem_32k_527`
  - losses: `personamem_32k_130`, `personamem_32k_302`

## Decision After E1
- Keep second-pass gate available, but treat current +4pp as provisional due high uncertainty.
- Next highest-value step: run a larger fixed-ID confirmation (100Q) and/or repeat 50Q with additional seeds before promoting E1 as a stable lift.

## Progression Started: 100Q Confirmation
- Launched background 100Q confirmation run (Arm B) with the same type mix, `samples=20` per type:
  - command (background): `PERSONA_ENABLE_MCQ_SECOND_PASS=true uv run mem-eval run --benchmark personamem --types recall_user_shared_facts,provide_preference_aligned_recommendations,suggest_new_ideas,recalling_the_reasons_behind_previous_updates,generalizing_to_new_scenarios --samples 20 --seed 42 --workers 2 --output results/ulw_paired100_20260209_armB`
  - run directory: `../memory-evals/results/ulw_paired100_20260209_armB/run_20260209_035331`

## 100Q Paired Confirmation (Completed)
- Arm B (second-pass enabled):
  - run: `../memory-evals/results/ulw_paired100_20260209_armB/run_20260209_035700`
  - accuracy: `67.00%` (`67/100`)
- Arm A (second-pass disabled):
  - run: `../memory-evals/results/ulw_paired100_20260209_armA/run_20260209_133301`
  - accuracy: `63.00%` (`63/100`)

- Paired comparison on identical 100 IDs:
  - Delta: `+4.00pp` (B - A)
  - Discordant pairs: gain `7`, loss `3`
  - McNemar exact p-value: `0.34375` (directionally positive, not statistically significant)
  - Wilson 95% CI:
    - Arm A: `0.5322 - 0.7182`
    - Arm B: `0.5731 - 0.7544`

- Diagnostic movement (paired 100 IDs):
  - mean tool calls: `1.02 -> 1.00`
  - context tokens: `14198.48 -> 14014.48`
  - retrieval ms: `5766.51 -> 5622.73`
  - no cost/loop blow-up detected.

- Per-type paired deltas:
  - `recall_user_shared_facts`: `+20.0pp` (`55% -> 75%`)
  - `suggest_new_ideas`: `+5.0pp` (`25% -> 30%`)
  - `provide_preference_aligned_recommendations`: `0.0pp`
  - `recalling_the_reasons_behind_previous_updates`: `0.0pp`
  - `generalizing_to_new_scenarios`: `-5.0pp` (`80% -> 75%`)

## Decision After 100Q
- **Promotion decision**: keep the second-pass gate as current default candidate (consistent `+4pp` at 50Q and 100Q, no cost increase).
- **Statistical caveat**: evidence is still not significance-strong; treat as provisional until one more paired 100Q seed confirms direction.
- **Next action**: run seed `123` paired 100Q A/B and require non-negative net delta with no tool-call inflation before declaring stable lift.
