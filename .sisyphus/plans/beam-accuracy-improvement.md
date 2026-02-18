# BEAM Accuracy Improvement Plan

## TL;DR

> **Quick Summary**: Fix BEAM benchmark accuracy from 69% to 85%+ by addressing prompt engineering gaps, tool selection enforcement, and retrieval policies. Focus on 5 weak question types: event_ordering (0%), knowledge_update (60%), contradiction_resolution (50%), temporal_reasoning (70%), multi_session_reasoning (50%).
> 
> **Deliverables**:
> - Updated system prompt with eval-strict rules, evolution verification, and multi-hop guidance
> - Enhanced tool schemas with stronger ordering mandates
> - Validation run showing accuracy improvements
> 
> **Estimated Effort**: Medium (1-2 days for prompt/schema changes + validation)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 (prompt changes) -> Task 5 (validation run)

---

## Context

### Original Request
Improve BEAM benchmark accuracy, specifically targeting weak categories: event_ordering (0%), knowledge_update (60%), contradiction_resolution (50%), temporal_reasoning (70%), and multi_session_reasoning (50%).

### Interview Summary
**Key Discussions**:
- Deep diagnostic analysis revealed root causes are primarily prompt/policy gaps, not architectural limitations
- Oracle consultation confirmed: "failures split cleanly into evaluation brittleness + retrieval-policy gaps"
- event_ordering 0% is primarily JUDGE_STRICT (semantic equivalence not recognized) + lack of extractive formatting

**Research Findings**:
- Agent DOES use correct tools (browse with order=asc) but outputs are paraphrased, causing judge failures
- Temporal reasoning fails when milestone dates aren't extracted - INDEX_MISSING at ingestion
- knowledge_update/contradiction_resolution fail because model picks "strongest" not "latest"
- multi_session_reasoning fails due to early-stop (no multi-hop retrieval)

### Oracle Review
**Identified Gaps** (addressed in this plan):
- No eval-strict formatting rules for deterministic, judge-aligned outputs
- No explicit evolution verification policy for tracking fact changes
- No multi-hop retrieval guidance for cross-session aggregation
- Tool schema doesn't mandate timeline() for ordering queries

---

## Work Objectives

### Core Objective
Increase BEAM benchmark accuracy from 69% to 85%+ by fixing prompt engineering and tool selection gaps.

### Concrete Deliverables
- `persona/llm/prompts.py` - Updated with eval-strict rules, evolution verification, multi-hop guidance
- `persona/tools/schemas.py` - Enhanced timeline() description with ordering mandate
- BEAM validation run showing accuracy improvements by question type

### Definition of Done
- [x] `bun test` passes (no regressions in persona core)
- [ ] ~~BEAM 100q run shows event_ordering > 40% (from 0%)~~ **FAILED: 0%**
- [ ] ~~BEAM 100q run shows knowledge_update > 70% (from 60%)~~ **FAILED: 50%**
- [ ] ~~BEAM 100q run shows contradiction_resolution > 60% (from 50%)~~ **FAILED: 10%**
- [ ] ~~BEAM 100q run shows multi_session_reasoning > 60% (from 50%)~~ **FAILED: 30%**
- [ ] ~~BEAM 100q run shows overall accuracy > 80% (from 69%)~~ **FAILED: 41%**

### Must Have
- Eval-strict formatting rules for ordering questions (extractive, not generative)
- Evolution verification policy for knowledge_update and contradiction_resolution
- Multi-hop retrieval guidance for multi_session_reasoning
- Strengthened timeline() tool schema

### Must NOT Have (Guardrails)
- NO changes to ingestion pipeline (too risky for v1)
- NO changes to tool implementations (only schemas and prompts)
- NO new tools (use existing recall/browse/timeline)
- NO breaking changes to existing prompt structure (additive only)
- Avoid over-engineering: simple prompt additions, not complex logic

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (bun test for persona, BEAM runner for eval)
- **User wants tests**: Manual verification via BEAM runs
- **Framework**: bun test (persona), python runner (memory-evals)

### Automated Verification

Each task includes verification commands that can be run without user intervention.

**Unit Test Verification:**
```bash
cd /Users/saxenauts/Documents/InnerNets\ AI\ Inc/persona/persona
bun test
```
Expected: All tests pass

**BEAM Validation:**
```bash
cd /Users/saxenauts/Documents/InnerNets\ AI\ Inc/persona/memory-evals
source .venv/bin/activate
python run_beam_100q.py --question-types event_ordering knowledge_update contradiction_resolution temporal_reasoning multi_session_reasoning
```
Expected: Accuracy improvements per type

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Add eval-strict rules to prompts.py
├── Task 2: Add evolution verification to prompts.py
└── Task 3: Add multi-hop retrieval guidance to prompts.py

Wave 2 (After Wave 1):
├── Task 4: Strengthen timeline() schema
└── Task 5: Run BEAM validation (depends on 1-4)

Critical Path: Task 1 → Task 5
Parallel Speedup: ~30% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 5 | 2, 3 |
| 2 | None | 5 | 1, 3 |
| 3 | None | 5 | 1, 2 |
| 4 | None | 5 | 1, 2, 3 |
| 5 | 1, 2, 3, 4 | None | None (final validation) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2, 3, 4 | delegate_task(category="quick", load_skills=[], run_in_background=true) for each |
| 2 | 5 | delegate_task(category="unspecified-high", load_skills=[], run_in_background=false) |

---

## TODOs

- [x] 1. Add Eval-Strict Formatting Rules to System Prompt

  **What to do**:
  - Open `persona/llm/prompts.py`
  - Find the `PERSONAL_AI_SYSTEM_PROMPT` variable
  - Add a new `<eval_strict_rules>` section BEFORE the `</system>` closing tag
  - Content to add:
  ```
  <eval_strict_rules>
  For ordering/sequence questions ("In what order...", "List the sequence...", "Walk me through the order..."):
  1. MUST use timeline(subject) or browse(order="asc") - recall() is NOT acceptable
  2. Output MUST be extractive: copy exact phrases from retrieved evidence, do not paraphrase
  3. Format as numbered list: "1) [exact phrase] 2) [exact phrase] 3) [exact phrase]"
  4. NO extra prose or explanation - just the numbered list
  5. When timestamps are identical, use discourse markers (first/then/before/after) from text to determine order
  
  For factual questions with possible updates:
  1. If multiple values exist for same fact, ALWAYS pick the LATEST by timestamp
  2. State the current value, not historical values
  </eval_strict_rules>
  ```

  **Must NOT do**:
  - Do not remove or modify existing prompt sections
  - Do not change the overall prompt structure
  - Do not add complex conditional logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file edit, clear instructions, low complexity
  - **Skills**: `[]`
    - No special skills needed for prompt editing

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Task 5 (validation)
  - **Blocked By**: None

  **References**:
  - `persona/llm/prompts.py:1-200` - System prompt definition, find PERSONAL_AI_SYSTEM_PROMPT
  - Oracle recommendation: "add eval-strict rule for ordering questions: use timeline/browse asc, copy exact step phrases from evidence"

  **Acceptance Criteria**:
  - [ ] `<eval_strict_rules>` section exists in PERSONAL_AI_SYSTEM_PROMPT
  - [ ] Section contains ordering rules (extractive format, timeline mandate)
  - [ ] Section contains factual update rules (pick latest by timestamp)
  ```bash
  grep -A 20 "eval_strict_rules" persona/llm/prompts.py
  # Assert: Returns the new section content
  ```

  **Commit**: YES
  - Message: `feat(prompts): add eval-strict rules for ordering and factual questions`
  - Files: `persona/llm/prompts.py`
  - Pre-commit: `bun test`

---

- [x] 2. Add Evolution Verification Policy to System Prompt

  **What to do**:
  - Open `persona/llm/prompts.py`
  - Find the `PERSONAL_AI_SYSTEM_PROMPT` variable
  - Add a new `<evolution_verification>` section AFTER `<eval_strict_rules>`
  - Content to add:
  ```
  <evolution_verification>
  When handling questions about facts that may have changed over time:
  
  KNOWLEDGE_UPDATE pattern (budget was $5k, then increased to $8k):
  1. Retrieve ALL mentions of the fact using recall()
  2. Build mini-timeline: sort by timestamp
  3. Answer with LATEST value only
  4. Older values are history, not candidates
  
  CONTRADICTION_RESOLUTION pattern (conflicting statements about same topic):
  1. Retrieve ALL statements about the topic
  2. Check timestamps - if one is clearly later, that's the current state
  3. If same timestamp: look for explicit markers ("now", "updated to", "changed my mind")
  4. If truly ambiguous: state both and note the ambiguity
  
  CRITICAL: Do not pick by similarity score alone. Recency trumps similarity for evolving facts.
  </evolution_verification>
  ```

  **Must NOT do**:
  - Do not remove existing contradiction handling (if any)
  - Do not add complex branching logic
  - Do not modify other prompt sections

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file edit, clear instructions
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:
  - `persona/llm/prompts.py` - Same file as Task 1
  - Oracle recommendation: "unified evolution verification policy: when multiple values exist, build mini-timeline and choose latest"
  - BEAM failures: knowledge_update picks "strongest" not "latest"

  **Acceptance Criteria**:
  - [ ] `<evolution_verification>` section exists in PERSONAL_AI_SYSTEM_PROMPT
  - [ ] Contains KNOWLEDGE_UPDATE and CONTRADICTION_RESOLUTION patterns
  - [ ] Contains "Recency trumps similarity" principle
  ```bash
  grep -A 25 "evolution_verification" persona/llm/prompts.py
  # Assert: Returns the new section content with both patterns
  ```

  **Commit**: YES (combine with Task 1 if done sequentially)
  - Message: `feat(prompts): add evolution verification policy for knowledge updates`
  - Files: `persona/llm/prompts.py`
  - Pre-commit: `bun test`

---

- [x] 3. Add Multi-Hop Retrieval Guidance to System Prompt

  **What to do**:
  - Open `persona/llm/prompts.py`
  - Find the `PERSONAL_AI_SYSTEM_PROMPT` variable
  - Add a new `<multi_hop_retrieval>` section AFTER `<evolution_verification>`
  - Content to add:
  ```
  <multi_hop_retrieval>
  For questions requiring aggregation across multiple sessions or time periods:
  
  1. DETECT: Question implies counting, totaling, or synthesis ("how many total", "across all sessions", "altogether")
  2. DO NOT stop after first recall - results may be incomplete
  3. ITERATE:
     - Run recall() with 2-3 alternate phrasings
     - Run browse() with wider date ranges
     - Continue until results span multiple dates/sessions
  4. VERIFY COVERAGE: Before answering, check if results seem complete
     - If thin/single-session: search again with different terms
     - Only compute totals after verifying breadth
  5. SYNTHESIZE: Combine information from all retrieved memories
  
  Example: "How many features did I mention across all sessions?"
  - First recall: finds 3 features from session 1
  - Check: only one session - incomplete!
  - Second recall (different terms): finds 2 more from session 2
  - Browse wider: finds 1 more from session 3
  - Answer: 6 features total
  </multi_hop_retrieval>
  ```

  **Must NOT do**:
  - Do not add retry limits that would cause early termination
  - Do not add complex state tracking
  - Keep it guidance-based, not algorithmic

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file edit, clear pattern
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:
  - `persona/llm/prompts.py` - Same file as Tasks 1-2
  - Oracle recommendation: "if question implies aggregation, do not stop after first recall; iterate with 2-3 alternate queries"
  - BEAM failures: multi_session_reasoning stops after one recall

  **Acceptance Criteria**:
  - [ ] `<multi_hop_retrieval>` section exists in PERSONAL_AI_SYSTEM_PROMPT
  - [ ] Contains iteration guidance (multiple queries, wider date ranges)
  - [ ] Contains coverage verification step
  - [ ] Contains synthesis instruction
  ```bash
  grep -A 25 "multi_hop_retrieval" persona/llm/prompts.py
  # Assert: Returns section with ITERATE and VERIFY steps
  ```

  **Commit**: YES (combine with Tasks 1-2 if done sequentially)
  - Message: `feat(prompts): add multi-hop retrieval guidance for cross-session queries`
  - Files: `persona/llm/prompts.py`
  - Pre-commit: `bun test`

---

- [x] 4. Strengthen Timeline Tool Schema

  **What to do**:
  - Open `persona/tools/schemas.py`
  - Find `TIMELINE_TOOL` definition
  - Update the description to make ordering mandate explicit
  - New description should include:
    - "MANDATORY for ordering/sequence questions"
    - "recall() is NOT acceptable for determining order"
    - "Returns results in chronological order (oldest first)"
    - "Use this for: 'in what order', 'list the sequence', 'what came first'"

  **Must NOT do**:
  - Do not change the tool's parameters
  - Do not change the tool's implementation
  - Do not remove existing functionality

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small schema text change
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:
  - `persona/tools/schemas.py:421-468` - TIMELINE_TOOL definition
  - Oracle recommendation: "strengthen timeline(subject) description to explicitly say it is mandatory for order/sequence queries"

  **Acceptance Criteria**:
  - [ ] TIMELINE_TOOL description contains "MANDATORY for ordering"
  - [ ] Description explicitly states "recall() is NOT acceptable for ordering"
  - [ ] Description lists trigger phrases ("in what order", "list the sequence")
  ```bash
  grep -A 10 "TIMELINE_TOOL" persona/tools/schemas.py | grep -i "mandatory\|not acceptable"
  # Assert: Returns lines with ordering mandate
  ```

  **Commit**: YES
  - Message: `feat(tools): strengthen timeline schema with ordering mandate`
  - Files: `persona/tools/schemas.py`
  - Pre-commit: `bun test`

---

- [x] 5. Run BEAM Validation and Verify Improvements

  **What to do**:
  - Ensure Persona API is running locally on port 8000
  - Navigate to memory-evals directory
  - Run BEAM evaluation targeting the 5 weak question types
  - Compare results against baseline (69% overall, 0% event_ordering, etc.)
  - Document improvements

  **Must NOT do**:
  - Do not modify BEAM benchmark code
  - Do not cherry-pick results
  - Do not skip any question type

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires running external evaluation, longer duration, analysis
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after all prompt changes)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 1, 2, 3, 4

  **References**:
  - `/Users/saxenauts/Documents/InnerNets AI Inc/persona/memory-evals/run_beam_100q.py` - BEAM runner
  - `/Users/saxenauts/Documents/InnerNets AI Inc/persona/memory-evals/results/beam_100q_minimal_prompt/run_20260126_113003/final_results.json` - Baseline results
  - Baseline: overall=69%, event_ordering=0%, knowledge_update=60%, contradiction_resolution=50%, temporal_reasoning=70%, multi_session_reasoning=50%

  **Acceptance Criteria**:

  **Pre-validation (ensure API running):**
  ```bash
  curl -s http://localhost:8000/health | jq '.status'
  # Assert: Returns "healthy"
  ```

  **Run BEAM validation:**
  ```bash
  cd /Users/saxenauts/Documents/InnerNets\ AI\ Inc/persona/memory-evals
  source .venv/bin/activate
  python run_beam_100q.py 2>&1 | tee results/beam_post_prompt_fix.log
  ```

  **Check results:**
  ```bash
  # Find latest run directory
  LATEST_RUN=$(ls -td results/run_* | head -1)
  cat $LATEST_RUN/final_results.json | jq '.beam.type_accuracies'
  # Assert: event_ordering.accuracy > 0.4
  # Assert: knowledge_update.accuracy > 0.7
  # Assert: contradiction_resolution.accuracy > 0.6
  # Assert: multi_session_reasoning.accuracy > 0.6
  # Assert: .beam.overall_accuracy > 0.8
  ```

  **Evidence to Capture:**
  - [ ] final_results.json from the run
  - [ ] Comparison table: baseline vs post-fix by question type
  - [ ] Any remaining failures for analysis

  **Commit**: NO (evaluation only, no code changes)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1, 2, 3 (combined) | `feat(prompts): add eval-strict, evolution verification, and multi-hop guidance` | persona/llm/prompts.py | bun test |
| 4 | `feat(tools): strengthen timeline schema with ordering mandate` | persona/tools/schemas.py | bun test |
| 5 | N/A (validation only) | N/A | BEAM run results |

---

## Success Criteria

### Verification Commands
```bash
# Unit tests pass
cd /Users/saxenauts/Documents/InnerNets\ AI\ Inc/persona/persona && bun test
# Expected: All tests pass

# BEAM accuracy improved
cd /Users/saxenauts/Documents/InnerNets\ AI\ Inc/persona/memory-evals
cat results/run_*/final_results.json | jq '.beam.overall_accuracy'
# Expected: > 0.80
```

### Final Checklist
- [x] All prompt sections added (eval_strict_rules, evolution_verification, multi_hop_retrieval)
- [x] Timeline schema strengthened
- [x] bun test passes
- [ ] ~~BEAM overall accuracy > 80%~~ **FAILED: 41%**
- [ ] ~~event_ordering > 40% (was 0%)~~ **FAILED: 0%**
- [ ] ~~knowledge_update > 70% (was 60%)~~ **FAILED: 50%**
- [ ] ~~contradiction_resolution > 60% (was 50%)~~ **FAILED: 10%**
- [ ] ~~multi_session_reasoning > 60% (was 50%)~~ **FAILED: 30%**

---

## PLAN COMPLETION SUMMARY

**Date Completed:** 2026-01-31
**Status:** COMPLETE BUT FAILED
**Tasks Completed:** 5/5 (100%)
**Success Criteria Met:** 0/5 (0%)

### Validation Results

Run ID: 20260131_123531
- Overall accuracy: 41% (target: 80%, baseline: 69%)
- event_ordering: 0% (target: 40%, baseline: 0%)
- knowledge_update: 50% (target: 70%, baseline: 60%)
- contradiction_resolution: 10% (target: 60%, baseline: 50%)
- multi_session_reasoning: 30% (target: 60%, baseline: 50%)

### Outcome

All implementation tasks were completed successfully, but the approach caused a **catastrophic -28% regression** instead of the targeted +11% improvement.

### Root Cause

The prompt engineering changes over-constrained the model and created conflicting instructions, making performance significantly worse.

### Recommendation

**REVERT all changes** (commits c25022c, 5753a2d, 56b8aca, 2667fa1) and restart with:
1. Deep log analysis first
2. Minimal, isolated changes
3. Validation after each change
4. Alternative approaches (retrieval, ingestion, tools)

### Lessons Learned

1. More prompt guidance ≠ better performance
2. Test changes in isolation before combining
3. Validate assumptions before implementing
4. Check for regressions early and often
5. Analyze actual failures before prescribing solutions

**This plan is COMPLETE but the approach FAILED. A new plan with a different strategy is required.**

### Post-Completion Action: REVERTED

**Date:** 2026-01-31 13:18:00
**Action:** Reverted all commits (600b76a, c281c9a, d457dd7, 1a9ac70)
**Reason:** Catastrophic -28% regression
**Result:** Codebase restored to baseline (69% accuracy)

The work plan is now COMPLETE and REVERTED. All changes have been undone to preserve baseline performance while a better approach is developed.
