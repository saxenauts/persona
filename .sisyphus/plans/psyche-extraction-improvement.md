# Psyche Extraction Improvement

## TL;DR

> **Quick Summary**: Improve Psyche extraction to capture preferences and behavioral patterns, fixing the 65% PersonaMem accuracy ceiling.
> 
> **Deliverables**:
> - Updated ingestion prompt with relaxed Psyche extraction rules
> - New consolidation function to infer Psyche from Episode patterns
> 
> **Estimated Effort**: Medium (2-3 hours)
> **Parallel Execution**: NO - sequential (ingestion first, then consolidation)
> **Critical Path**: Task 1 → Task 2 → Task 3

---

## Context

### Original Request
Improve Psyche extraction to fix PersonaMem benchmark accuracy. Current: 65%, Target: 70-75%.

### Root Cause Analysis
The system retrieves relevant memories (83% recall) but picks generic MCQ answers because:
- Episodes capture "what happened" but not "what they prefer"
- Psyche extraction is too conservative ("1 per 5-10 sessions is normal")
- Model hedges with generic responses when no explicit preference exists

### Research Findings
- Oracle consultation confirmed: fix BOTH ingestion (explicit Psyche) AND consolidation (inferred Psyche)
- Ingestion should capture evaluative language (like/love/hate/prefer)
- Consolidation should turn repeated behaviors into stable preferences

---

## Work Objectives

### Core Objective
Extract Psyche entries that capture user preferences and values, enabling the model to confidently assert preferences in MCQ answers.

### Concrete Deliverables
- Modified `INGESTION_SYSTEM_PROMPT` in `persona/services/ingestion_service.py`
- New `infer_psyche_from_patterns()` function in `persona/services/consolidation_service.py`
- Integration call in `refresh_memeplex()`

### Definition of Done
- [x] Ingestion prompt no longer says "1 per 5-10 sessions"
- [x] Consolidation infers Psyche from 3+ repeated behaviors
- [x] `python -c "from persona.services.consolidation_service import infer_psyche_from_patterns"` works
- [x] PersonaMem 50q validation shows improvement

### Must Have
- Evaluative language triggers Psyche extraction in ingestion
- Behavioral patterns trigger Psyche inference in consolidation
- Deterministic IDs (uuid5) for idempotent upserts

### Must NOT Have (Guardrails)
- No keyword heuristics - use LLM for pattern detection
- No over-extraction (still require evidence)
- No benchmark-specific hacks

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **User wants tests**: Manual verification via eval
- **QA approach**: Run PersonaMem 50q eval after changes

---

## TODOs

- [x] 1. Update Ingestion Prompt

  **What to do**:
  - Open `persona/services/ingestion_service.py`
  - Find lines 53-70 (PSYCHE EXTRACTION section)
  - Replace the restrictive prompt with the new guidance

  **Must NOT do**:
  - Don't change other sections of the prompt
  - Don't remove the psyche_type definitions

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - Reason: Simple text replacement in a single file

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Task 2, 3
  - **Blocked By**: None

  **References**:
  - `persona/services/ingestion_service.py:53-70` - Current PSYCHE EXTRACTION section to replace

  **Acceptance Criteria**:
  
  ```bash
  # Verify old text is gone
  grep -c "1 per 5-10 sessions" persona/services/ingestion_service.py
  # Expected: 0
  
  # Verify new text is present
  grep -c "CAPTURE THE WHY" persona/services/ingestion_service.py
  # Expected: 1
  ```

  **Old Text** (lines 53-70):
  ```
  ## PSYCHE EXTRACTION (BE VERY SELECTIVE)

  Psyche represents WHO THE PERSON IS at their core. Most sessions don't reveal new psyche.

  **ONLY extract psyche when you see SIGNIFICANT identity signals:**
  - Core values being stated: "Family comes first for me", "I believe in..."
  - Fundamental preferences: "I'm a morning person", "I've always loved..."
  - Personality traits: "I tend to be introverted", "I'm risk-averse"
  - Life philosophy: "I live by the rule...", "My approach to life is..."

  **DO NOT extract psyche for:**
  - Temporary preferences: "I want pizza tonight" (this is just episode context)
  - Situational feelings: "I'm stressed about the deadline" (episode, not psyche)
  - One-time opinions: "That movie was good" (episode context)
  - Facts about activities: "I went running" (episode, maybe entity for 'running')

  **Rule of thumb**: If you're unsure, DON'T extract psyche. Episodes capture the detail.
  Most sessions should have 0 psyche. Only 1 psyche per 5-10 sessions is normal.
  ```

  **New Text**:
  ```
  ## PSYCHE EXTRACTION (CAPTURE THE WHY)

  Psyche represents what drives behavior - preferences, values, beliefs, and identity.
  Extract Psyche when you see evaluative language revealing the person's inner landscape.

  **EXTRACT psyche when you see:**
  - Preferences: "I like/love/hate/prefer...", "I enjoy/dread..."
  - Identity: "I'm the kind of person who...", "I always/never..."
  - Values/beliefs: "I value...", "I believe...", "What matters to me is..."
  - Reactions revealing preference: "That was amazing/terrible", "I had so much fun"
  - Recurring patterns with sentiment: doing something repeatedly AND expressing feeling about it

  **DO NOT extract psyche for:**
  - Neutral activity descriptions: "I went to the store" (no sentiment = Episode only)
  - Situational states: "I'm tired today" (temporary, not identity)
  - Single mentions without evaluative language

  **Psyche types:**
  - trait: personality characteristics ("I'm introverted")
  - preference: likes/dislikes ("I love hiking")
  - value: what matters to them ("Family comes first")
  - belief: worldview ("I believe in hard work")

  **Guideline**: 1-2 Psyche per session is healthy if evaluative language is present.
  Skip Psyche only when the session is purely factual narration.
  ```

  **Commit**: YES
  - Message: `feat(ingestion): relax Psyche extraction to capture evaluative language`
  - Files: `persona/services/ingestion_service.py`

---

- [x] 2. Add Consolidation Psyche Inference Function

  **What to do**:
  - Open `persona/services/consolidation_service.py`
  - Add imports: `from uuid import uuid5, NAMESPACE_DNS` and `Any` to typing, `PsycheMemory` to models
  - Add `PSYCHE_INFERENCE_PROMPT` constant
  - Add `infer_psyche_from_patterns()` function after `refresh_memeplex()` (around line 560)

  **Must NOT do**:
  - Don't modify existing functions yet
  - Don't use `store.upsert()` (doesn't exist) - use `store.create()`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - Reason: Adding a new function with clear specification

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:
  - `persona/services/consolidation_service.py:1-30` - Current imports to extend
  - `persona/services/consolidation_service.py:462-557` - refresh_memeplex() to understand context
  - `persona/core/memory_store.py` - MemoryStore.create() signature

  **Acceptance Criteria**:
  
  ```bash
  # Verify function is importable
  python -c "from persona.services.consolidation_service import infer_psyche_from_patterns; print('OK')"
  # Expected: OK
  ```

  **Code to Add**:

  **Imports** (add to top of file):
  ```python
  # Line 13 - modify existing typing import:
  from typing import Optional, List, Any
  
  # Add after line 12 (after datetime imports):
  from uuid import uuid5, NAMESPACE_DNS
  
  # Line 17-23 - add PsycheMemory to existing import:
  from persona.models.memory import (
      UserCard,
      TemporalContext,
      Memory,
      Memeplex,
      EpisodeMemory,
      PsycheMemory,  # ADD THIS
  )
  ```

  **Constant and Function** (add after line 557, after refresh_memeplex):
  ```python
  # ============================================================================
  # Psyche Inference from Behavioral Patterns
  # ============================================================================

  PSYCHE_INFERENCE_PROMPT = """Analyze these episodes for recurring behavioral patterns that reveal preferences or interests.

  Episodes:
  {episodes}

  For each activity/topic that appears 3+ times OR has clear positive/negative sentiment:
  1. Identify the entity/activity
  2. Determine engagement type based on evidence:
     - "often engages with" (neutral - repeated but no clear sentiment)
     - "enjoys" (positive sentiment detected: "fun", "love", "excited", "great")
     - "dislikes" (negative sentiment: "hate", "dread", "boring", "awful")
  3. Cite the supporting evidence (episode snippets)

  Return JSON:
  {{
    "inferred_preferences": [
      {{
        "entity": "mock trial competitions",
        "engagement_type": "enjoys",
        "confidence": 0.85,
        "evidence": ["participated in mock trial - had a great time", "won regional competition"]
      }}
    ]
  }}

  Only include preferences with clear behavioral evidence. Skip if uncertain.
  """


  async def infer_psyche_from_patterns(
      user_id: str,
      store: MemoryStore,
      episodes: List[Any],
  ) -> int:
      """
      Infer Psyche entries from behavioral patterns across Episodes.
      
      Called from refresh_memeplex() after memeplex is built.
      Uses deterministic IDs (uuid5) for idempotent upserts.
      
      Returns count of Psyche entries created/updated.
      """
      if len(episodes) < 3:
          return 0  # Not enough data for pattern inference
      
      # Format episodes for LLM
      episode_text = "\n".join([
          f"[{e.event_time.strftime('%Y-%m-%d') if e.event_time else 'unknown'}] {e.content[:300]}"
          for e in episodes[:30]  # Limit to 30 most recent
      ])
      
      try:
          chat_client = get_chat_client()
          response = await chat_client.chat(
              messages=[
                  ChatMessage(
                      role="user",
                      content=PSYCHE_INFERENCE_PROMPT.format(episodes=episode_text),
                  )
              ],
              response_format={"type": "json_object"},
          )
          
          data = json.loads(response.content or "{}")
          preferences = data.get("inferred_preferences", [])
          
          if not preferences:
              return 0
          
          created_count = 0
          now = datetime.now(timezone.utc)
          
          for pref in preferences:
              entity = pref.get("entity", "").strip()
              engagement = pref.get("engagement_type", "often engages with")
              confidence = pref.get("confidence", 0.7)
              evidence = pref.get("evidence", [])
              
              if not entity or confidence < 0.6:
                  continue
              
              # Deterministic ID for idempotent upsert
              stable_key = f"{user_id}:preference:{entity.lower().replace(' ', '_')}"
              psyche_id = uuid5(NAMESPACE_DNS, stable_key)
              
              # Format content based on engagement type
              if engagement == "enjoys":
                  content = f"Enjoys {entity}"
              elif engagement == "dislikes":
                  content = f"Dislikes {entity}"
              else:
                  content = f"Often engages with {entity}"
              
              psyche = PsycheMemory(
                  id=psyche_id,
                  psyche_type="preference",
                  title="preference",
                  content=content,
                  user_id=user_id,
                  event_time=now,
                  observed_at=now,
                  day_id=now.strftime("%Y-%m-%d"),
                  properties={
                      "source": "behavioral_inference",
                      "confidence": confidence,
                      "evidence_count": len(evidence),
                      "evidence_snippets": evidence[:3],
                  },
              )
              
              # Create (Neo4j MERGE handles duplicates via name field)
              try:
                  await store.create(psyche, links=[])
                  created_count += 1
              except Exception as e:
                  logger.warning(f"Failed to create inferred psyche: {e}")
          
          if created_count > 0:
              logger.info(f"Inferred {created_count} Psyche entries for user {user_id}")
          
          return created_count
          
      except Exception as e:
          logger.warning(f"Psyche inference failed for {user_id}: {e}")
          return 0
  ```

  **Commit**: YES
  - Message: `feat(consolidation): add Psyche inference from behavioral patterns`
  - Files: `persona/services/consolidation_service.py`

---

- [x] 3. Integrate Inference into refresh_memeplex()

  **What to do**:
  - In `refresh_memeplex()` function, add call to `infer_psyche_from_patterns()` before the return
  - Find lines 552-553 (after `await store.save_memeplex(memeplex)`)

  **Must NOT do**:
  - Don't modify the memeplex building logic
  - Don't change the return type

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - Reason: Single line addition

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: None (final task)
  - **Blocked By**: Task 2

  **References**:
  - `persona/services/consolidation_service.py:550-557` - End of refresh_memeplex function

  **Acceptance Criteria**:
  
  ```bash
  # Verify the call is in refresh_memeplex
  grep -A2 "save_memeplex" persona/services/consolidation_service.py | grep "infer_psyche"
  # Expected: shows the infer_psyche_from_patterns call
  ```

  **Change**:
  
  **Current** (lines 552-553):
  ```python
          await store.save_memeplex(memeplex)
          return memeplex
  ```

  **New**:
  ```python
          await store.save_memeplex(memeplex)
          
          # Infer Psyche from behavioral patterns
          await infer_psyche_from_patterns(user_id, store, month_episodes)
          
          return memeplex
  ```

  **Commit**: YES
  - Message: `feat(consolidation): integrate Psyche inference into memeplex refresh`
  - Files: `persona/services/consolidation_service.py`

---

- [x] 4. Run Validation Eval

  **What to do**:
  - Run PersonaMem 50-question eval
  - Compare accuracy to baseline (65%)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocked By**: Tasks 1, 2, 3

  **Acceptance Criteria**:
  
  ```bash
  cd ../memory-evals
  
  # Fresh environment
  docker compose down -v && docker compose up -d
  
  # Wait for Neo4j
  sleep 30
  
  # Run eval
  PYTHONUNBUFFERED=1 .venv/bin/mem-eval run \
    --benchmark personamem --adapter persona --samples 50 --seed 42 \
    --output results/psyche_improvement_$(date +%Y%m%d_%H%M%S)
  ```

  **Expected**: Accuracy > 65% (baseline)

  **Commit**: NO (eval results only)

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `feat(ingestion): relax Psyche extraction to capture evaluative language` | ingestion_service.py |
| 2+3 | `feat(consolidation): add Psyche inference from behavioral patterns` | consolidation_service.py |

---

## Success Criteria

### Quantitative
- [x] PersonaMem accuracy > 65% (current baseline)
- [ ] Psyche entries per user increases from ~2 to ~5-10

### Qualitative
- [x] Ingestion prompt captures preferences from evaluative language
- [x] Consolidation infers preferences from behavioral patterns
- [ ] Model picks personalized MCQ options instead of generic ones
