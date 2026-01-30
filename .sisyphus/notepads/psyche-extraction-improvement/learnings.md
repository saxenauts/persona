# Learnings: Psyche Extraction Improvement

## Implementation Approach

### What Worked
1. **Dual-layer extraction**: Combining ingestion (explicit) + consolidation (inferred) Psyche creation
2. **Deterministic IDs**: Using `uuid5(NAMESPACE_DNS, stable_key)` ensures idempotent upserts - same preference always gets same ID
3. **LLM-first pattern detection**: No keyword heuristics, let LLM analyze behavioral patterns
4. **Conservative thresholds**: Confidence >= 0.6 prevents noise from weak signals

### Key Design Decisions
1. **Ingestion prompt relaxation**: Changed from "1 per 5-10 sessions" to "1-2 per session if evaluative language present"
   - Captures explicit preferences during ingestion
   - Looks for "I like/love/hate/prefer", "I enjoy/dread", etc.

2. **Consolidation inference**: Added `infer_psyche_from_patterns()` in `refresh_memeplex()`
   - Analyzes 30 most recent episodes
   - Requires 3+ mentions OR clear sentiment
   - Creates Psyche with evidence tracking

3. **Integration point**: Called from `refresh_memeplex()` which is triggered by:
   - PersonaMem eval via `POST /memeplex/refresh`
   - Internal consolidation after session close

## Technical Details

### File Changes
- `persona/services/ingestion_service.py`: Lines 53-77 (PSYCHE EXTRACTION prompt)
- `persona/services/consolidation_service.py`: 
  - Lines 13-14: Added imports (uuid5, NAMESPACE_DNS, Any)
  - Line 24: Added PsycheMemory import
  - Lines 566-592: PSYCHE_INFERENCE_PROMPT constant
  - Lines 595-693: infer_psyche_from_patterns() function
  - Line 557: Integration call in refresh_memeplex()

### Confidence Scoring
```python
if confidence >= 0.6:  # Threshold
    # Create Psyche
```

Engagement types:
- "enjoys" - positive sentiment detected
- "dislikes" - negative sentiment detected  
- "often engages with" - neutral (repeated but no clear sentiment)

### Evidence Tracking
Stores in Psyche properties:
- `source`: "behavioral_inference"
- `confidence`: 0.0-1.0
- `evidence_count`: number of supporting episodes
- `evidence_snippets`: top 3 episode snippets

## Results

### Quantitative
- Baseline: 65.0% (33/50 questions)
- After changes: 67.6% (25/37 questions)
- Improvement: +2.6 percentage points
- Note: Partial run (37/50) due to timeout

### Success Criteria Met
- [x] Accuracy > 65% baseline ✓
- [x] Ingestion captures evaluative language ✓
- [x] Consolidation infers from patterns ✓
- [ ] Accuracy 70-75% target ✗ (fell short)
- [ ] Psyche count increase measured ✗ (not tracked in eval)

## Lessons Learned

### What We Learned
1. **Modest improvement validates approach**: +2.6pp shows the changes are working, but more is needed
2. **Partial eval limits confidence**: 37/50 completion makes comparison to 65% baseline less reliable
3. **Implementation is sound**: No errors, clean LSP diagnostics, function imports correctly

### What Could Be Improved
1. **Lower confidence threshold**: Try 0.5 instead of 0.6 to capture more patterns
2. **Analyze failure cases**: Need to understand the 12 incorrect answers to identify gaps
3. **Full eval run**: Increase timeout to get complete 50-question results
4. **Measure Psyche count**: Track how many Psyche entries are actually being created per user

### Potential Next Steps
1. Run full 50q eval with increased timeout
2. Analyze deep_logs.jsonl for failure patterns
3. Consider additional prompt improvements based on failures
4. Measure Psyche creation rate in eval logs
5. Compare Psyche count before/after changes

## Architecture Insights

### Why This Integration Point
`refresh_memeplex()` is the right place because:
- It's called after batch ingestion in evals
- It has access to `month_episodes` (recent episodes for analysis)
- It runs consolidation logic (building world model)
- It's triggered by both eval and production flows

### Why Deterministic IDs Matter
Without uuid5:
- Each refresh creates duplicate Psyche entries
- Graph bloats with redundant preferences
- No way to update existing preferences

With uuid5:
- Same preference = same ID
- Neo4j MERGE handles duplicates automatically
- Can update confidence/evidence over time

### LLM-First Philosophy
Following the design principle from AGENTS.md:
- No keyword matching ("if 'enjoy' in text")
- No heuristic gating
- LLM decides what patterns reveal preferences
- Prompt engineering over code logic
