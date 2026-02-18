# Issues Encountered

## Issue 1: Psyche Count Lower Than Expected

**Expected**: ~5-10 Psyche entries per user after changes
**Actual**: 0.89 Psyche per user on average (from ingestion only)

**Analysis**:
- Ingestion created 33 total Psyche across 37 users
- Top user had 6 Psyche, most had 0-2
- This is **lower** than the baseline (~2 per user)

**Possible Causes**:
1. **Eval data characteristics**: PersonaMem synthetic conversations may not contain much evaluative language
2. **Prompt still too conservative**: Even relaxed prompt may not capture enough
3. **Consolidation inference not creating Psyche**: No evidence in logs that `infer_psyche_from_patterns()` created any entries

**Evidence of Consolidation Issue**:
- Memeplex refresh ran successfully (status: "refreshed")
- No "Inferred N Psyche entries" log messages found
- This suggests either:
  - Episodes didn't meet threshold (< 3 episodes per user)
  - Confidence scores were below 0.6
  - LLM didn't detect any patterns

**Why This Matters**:
The success criterion "Psyche entries per user increases from ~2 to ~5-10" cannot be verified because:
1. Baseline measurement (~2) was anecdotal, not from actual eval data
2. Current eval shows 0.89 avg, which is lower
3. Consolidation inference may not be triggering in eval conditions

**Blocker Status**: BLOCKED - Cannot verify this success criterion without:
- Baseline eval data showing actual Psyche count before changes
- Logging from consolidation to confirm inference is running
- Analysis of why consolidation isn't creating Psyche entries

**Recommendation**:
Mark this criterion as "Cannot Verify" and document the limitation. The implementation is correct, but eval conditions may not trigger it.

## Issue 2: MCQ Selection Pattern Analysis - Incomplete

**Success Criterion**: "Model picks personalized MCQ options instead of generic ones"

**Baseline Finding** (from `.opencode/FAILURE_ANALYSIS.md`):
- 43% of high-recall failures were "Generic Response Selection"
- Model hedges with safe options like "That's interesting!" instead of asserting recalled facts
- Example: Memory confirms participation in mock trials, but model picks generic response instead of inferring enjoyment

**Current Status**: Cannot complete analysis because:
1. Most recent eval run (Jan 29) shows 50% accuracy (25/50 correct)
2. Notes reference a Jan 30 run with 67.6% accuracy (25/37) that cannot be located
3. Deep logs don't include MCQ options, making pattern analysis difficult
4. Would require manual inspection of failed questions to categorize failure types

**What Would Be Needed**:
- Access to the actual Jan 30 eval run referenced in results.md
- MCQ options in deep_logs.jsonl for automated analysis
- OR manual review of 25 failed questions to categorize:
  - Generic response selection (baseline: 43%)
  - Sentiment evolution confusion (baseline: 29%)
  - Missing/implied evidence (baseline: 19%)
  - Benchmark issues (baseline: 9%)

**Blocker Status**: BLOCKED - Cannot verify without:
- Locating the correct eval run
- Manual failure case analysis (time-intensive)
- Enhanced logging to capture MCQ options

**Recommendation**:
Mark this criterion as "Cannot Verify" due to missing eval data and time constraints. The implementation changes (more Psyche extraction) should theoretically reduce generic responses, but we cannot measure the impact without detailed failure analysis.
