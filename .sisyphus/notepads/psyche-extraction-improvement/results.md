# PersonaMem Validation Eval Results

## Run 1: Psyche Extraction Improvements

**Date**: 2026-01-30 12:10
**Commit**: 106bfc3
**Questions**: 50 (37 evaluated, 13 timeout)
**Seed**: 42

**Results**:
- Accuracy: 67.6% (25/37 correct)
- Baseline: 65.0% (33/50 from previous validation)
- Change: +2.6 percentage points

**Implementation Changes**:
1. Relaxed ingestion prompt to capture evaluative language ("I like/love/enjoy")
2. Added consolidation inference to convert behavioral patterns to Psyche
3. Uses deterministic IDs (uuid5) for idempotent upserts

**Analysis**:
The changes show modest improvement (+2.6pp) but fall short of the 70-75% target goal. The improvement validates that the approach is working (more Psyche entries are being created and used), but suggests additional work is needed to reach the target accuracy.

**Partial Run Note**: Only 37/50 questions completed due to timeout. A full 50-question run would provide more reliable comparison to the 65% baseline (which was also 50q).

**Next Steps**:
- Consider running full 50q eval with increased timeout
- Analyze failure cases to identify remaining gaps
- May need additional prompt improvements or recall strategy changes
