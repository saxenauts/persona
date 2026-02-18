# ULW 85 Execution Framework - 2026-02-08

## Goal
Reach a defensible `>=85%` PersonaMem result with matched, fixed-ID evaluation and explicit separation of product lift vs eval-mapping lift.

## What Helps vs Adds Noise (Current Evidence)
- Data base: `../memory-evals/results/**/deep_logs.jsonl`.
- Filtered cohort with full persona context + recall composition (`n=5940`):
  - `tool_calls=1`: `64.80%` vs `tool_calls>=2`: `58.16%`.
  - `world_model_chars>3000`: `43.60%` vs `<=3000`: `64.94%`.
  - `psy_share>0`: `56.57%` vs `psy_share=0`: `67.28%`.
  - `entity_share<=0.20`: `50.35%` vs otherwise `67.19%`.
- Dominant type (`recall_user_shared_facts`, `n=3531`) keeps same direction:
  - `tool_calls>=2`: `58.82%` vs `63.11%`.
  - `world_model_chars>3000`: `41.67%` vs `63.88%`.
  - `psy_share>0`: `57.18%` vs `66.59%`.
- Option extraction is not the main failure source:
  - incorrect rows with `could not extract option`: `~1.21%`.
  - incorrect rows with extracted letter but wrong semantic choice: `~95.76%`.

## Interpretation
- Primary bottleneck is semantic evidence selection/composition quality under retrieval context, not MCQ parser extraction.
- Global historical aggregate mixes cohorts with different process quality; use fixed-ID paired runs for causal conclusions.
- Current non-archive cohorts show process-level variance (`~62%` to `~74%`) under same model family, indicating pipeline/configuration dominates model variance right now.

## Measurement Protocol (Mandatory)
- **Dataset lock**: Use fixed IDs from `../memory-evals/data/golden_sets/personamem_golden_set_manifest.json`.
- **Run design**: A/B paired comparison on identical IDs, same seed list, same infra, same adapter.
- **Primary metric**: exact-match accuracy.
- **Statistical reporting**:
  - Wilson 95% CI for each arm.
  - Paired significance (McNemar on per-question wins/losses).
  - Report absolute delta in percentage points.
- **Secondary diagnostics** (must move in intended direction):
  - `tool_calls_made` mean,
  - percent rows with `world_model_chars>3000`,
  - recall composition (`entity/episode/psyche share`),
  - calibration ambiguity rate (from `retrieval.calibration`).

## Intervention Stack (Ranked)
1. **Lock 1-call default with strict gate for pass-2**
   - Mechanism: keep first pass as default; permit second pass only on low separability/explicit ambiguity.
   - Expected lift: `+1` to `+4pp` on mixed cohort; protects against over-retrieval drift.
   - Kill criteria: mean tool calls rises above `1.2-1.4` with no paired uplift.

2. **World-model budget control**
   - Mechanism: cap effective world model/context budget to avoid `>3000` harmful regime.
   - Expected lift: `+1` to `+3pp` on cohorts where oversized world model appears.
   - Kill criteria: if `>3000` incidence already near-zero and no uplift.

3. **Composition steering for fact-recall prompts**
   - Mechanism: steer toward balanced entity + episode evidence and avoid psyche over-dominance when question asks factual recall.
   - Expected lift: `+2` to `+5pp` in fact-recall-heavy slices.
   - Kill criteria: composition metrics shift but accuracy does not.

4. **Eval-side deterministic calibration (separate track)**
   - Mechanism: deterministic remap for option framing edge cases.
   - Expected lift: benchmark-only uplift likely modest on current stack (parser errors low).
   - Kill criteria: `<+1pp` on fixed-ID paired run.

## Five Fast Experiments (Fixed-ID)
1. **E1: Hard 1-call vs gated 2nd-pass**
   - Arms: A hard 1-call; B gated pass-2.
   - Success: B beats A by `>=+2pp` with mean tool calls `<=1.2`.

2. **E2: World-model cap stress test**
   - Arms: A current budget; B reduced world-model/context cap.
   - Success: lower `>3000` incidence and `>=+1.5pp` uplift.

3. **E3: Fact-recall composition steering**
   - Arms: A baseline retrieval composition; B fact-recall steering prompt/tool guidance.
   - Success: psyche-share down and `>=+2pp` uplift on `recall_user_shared_facts`.

4. **E4: Cohort-stratified confidence gate**
   - Arms: A single global gate; B gate tuned by separability bins from calibration diagnostics.
   - Success: gain on ambiguous subset without regression on easy subset.

5. **E5: Calibration-only arm (eval track)**
   - Arms: A no remap; B deterministic remap.
   - Success: quantify mapping-only lift; keep separate from runtime claims.

## Execution Timeline
- `T+0-6h`: paired 50Q run (baseline vs E1).
- `T+6-12h`: paired 50Q run (best of E1 vs E2).
- `T+12-24h`: paired 100Q confirmation on best stack and CI reporting.

## E1 Status (Executed)
- Completed paired 50Q run on identical IDs:
  - Arm A (`PERSONA_ENABLE_MCQ_SECOND_PASS=false`): `32/50` (`64.0%`)
  - Arm B (`PERSONA_ENABLE_MCQ_SECOND_PASS=true`): `34/50` (`68.0%`)
- Paired delta: `+4.0pp` with discordant `(gain=4, loss=2)`.
- Statistical status: McNemar exact `p=0.6875` -> inconclusive at `n=50`.
- Operational status: tool-call average stayed flat (`1.04` vs `1.02`), so no cost blow-up observed.

## E1 100Q Confirmation
- Arm A (`PERSONA_ENABLE_MCQ_SECOND_PASS=false`): `63/100` (`63.0%`) in `../memory-evals/results/ulw_paired100_20260209_armA/run_20260209_133301`.
- Arm B (`PERSONA_ENABLE_MCQ_SECOND_PASS=true`): `67/100` (`67.0%`) in `../memory-evals/results/ulw_paired100_20260209_armB/run_20260209_035700`.
- Paired delta on identical IDs: `+4.0pp` with discordant `(gain=7, loss=3)`.
- Statistical status: McNemar exact `p=0.34375` (positive direction, not yet significance-strong).
- Diagnostics: no cost regression (`tool_calls` and retrieval/context budget did not increase).

## Immediate Next Move
- Promote E1 to **provisional winner**, then run either:
  - fixed-ID 100Q confirmation at same settings, or
  - two additional fixed-ID 50Q repeats (different seeds) for variance reduction.
- Do not claim stable lift until paired significance or consistent repeatability is observed.

## Updated Next Move
- Fixed-ID 100Q confirmation is complete and directionally consistent.
- Next requirement for stable promotion: one additional paired 100Q seed (recommended `seed=123`) with non-negative delta and no tool inflation.

## Go / No-Go Gates
- Proceed only if each accepted lever shows `>=+1pp` paired uplift and no diagnostic regression.
- Promote a stack to 100Q confirmation only if 50Q CI and paired test agree on positive lift.
- Claim path-to-85 only after fixed-ID 100Q evidence with maintained lift and no cost explosion.
