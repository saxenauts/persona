# PersonaMem Failure Analysis — Post Feb 8 Changes

**Date**: 2026-02-09 (updated with full cross-arm 100Q analysis)
**Run**: `run_20260208_154926` (Arm A, 50Q, seed 42) — 64% accuracy (32/50)
**Comparison Run**: `run_20260208_212320` (Arm B, 50Q, seed 42, MCQ second-pass gating) — 68% accuracy (34/50)
**100Q Run**: `ulw_paired100_20260209_armB/run_20260209_035700` (Arm B, 119 entries, seed 42) — **68.9% accuracy (82/119)** *(COMPLETE)*
**100Q Arm A Run**: `ulw_paired100_20260209_armA/run_20260209_133301` — **63.0% accuracy (63/100)** *(COMPLETE)*
**Codebase State**: All Feb 8 code changes applied (uncommitted working tree)
**Deep Logs**: `../memory-evals/results/run_20260208_154926/deep_logs.jsonl`
**100Q Deep Logs**: `../memory-evals/results/ulw_paired100_20260209_armB/run_20260209_035700/deep_logs.jsonl`
**Questions Source**: `../memory-evals/evals/data/personamem/questions_32k_32k.json`
**Conversations Source**: `../memory-evals/evals/data/personamem/shared_contexts_32k.jsonl`

---

## Executive Summary

### 50Q Analysis (Phase 1-3)

18 failures across 50 questions. Initial classification showed 78% ambiguous evidence failures, 17% temporal evolution, 6% partial contradiction. **Source forensics overturned the initial assessment** — most "ambiguous" cases are actually RETRIEVAL_INSUFFICIENT (evidence exists in source but wasn't retrieved). See Revised Root Cause below.

### 100Q Analysis (Phase 4)

37 failures across 119 entries (68.9%, COMPLETE). The expanded sample reveals a **catastrophic failure mode** in `suggest_new_ideas` (30% accuracy) driven by a benchmark-model interaction: gold answers are ALWAYS the shortest option, and the model has a systematic verbosity bias.

### Memeplex Effectiveness (Phase 5) — NEW

**The memeplex provides minimal measurable benefit in its current form.** Statistical correlation between memeplex size and accuracy is weak (Cohen's d=0.237) and confounded by data richness. The structural problem: memeplex provides TOPIC LABELS (what domains the user engages with) but MCQ questions already state the domain. What's needed is PREFERENCE DIRECTION (how the user feels within that domain). Proposed enhancement: transform from "table of contents" into "preference compass" with active/dropped status, behavioral meta-patterns, and recent transitions. See Section 10.

### Revised Root Cause Distribution (All Evidence Combined)

| Root Cause | 50Q Count | 100Q Total (est) | Fix Layer | Priority |
|------------|-----------|-------------------|-----------|----------|
| **RETRIEVAL_INSUFFICIENT** | 7 | ~16 | Retrieval + Ingestion | P0 |
| **VERBOSITY_BIAS** (new) | — | ~10 | Prompt + MCQ format | P0 |
| **TEMPORAL_EVOLUTION** | 3 | 4 | Consolidation | P0 |
| **PROMPT_REASONING** | 3 | ~5 | Prompt | P1 |
| **CONSOLIDATION_CONFLICT** | 1 | 1 | Consolidation | P1 |
| **GOLD_DEBATABLE** | 1 | ~2 | Eval data | P2 |
| **STOCHASTIC_AMBIGUITY** | 2 | ~3 | None (ceiling) | — |

### By Question Type (100Q FINAL — 119 entries, 82/119 = 68.9%)

| Type | Correct | Total | Accuracy | Note |
|------|---------|-------|----------|------|
| recalling_the_reasons_behind_previous_updates | 18 | 20 | **90.0%** | Best performer |
| recall_user_shared_facts | 30 | 39 | **76.9%** | Good but temporal evolution hurts |
| generalizing_to_new_scenarios | 15 | 20 | **75.0%** | Improved from partial sample |
| provide_preference_aligned_recommendations | 13 | 20 | **65.0%** | Moderate |
| suggest_new_ideas | 6 | 20 | **30.0%** | **CATASTROPHIC — see Section 7** |

### 50Q Cross-Run Stability

| Outcome | Count | Question IDs |
|---------|-------|--------------|
| **Deterministic across 50Q + 100Q** | 15 | 50, 53, 80, 97, 130, 131, 209, 256, 302, 307, 309, 331, 342, 494, 495 |
| **Fixed in 100Q** (50Q fail → 100Q pass) | 1 | 175 |
| **NEW in 100Q** (not in 50Q sample or regressed) | 17 | 43, 71, 147, 163, 277, 281, 305, 336, 343, 352, 379, 401, 442, 479, 550, 558, 570 |

### Agent Behavior: Flat Single-Pass

**Every single entry** in the 100Q run uses exactly **1 tool call** and **2 turns**. The agent is NOT doing multi-step retrieval, graph expansion, or iterative refinement. This is a system-level limitation — the agent loop isn't engaging the browse, expand_neighbors, or follow_relationship tools.

---

## Retrieval Statistics Across All 18 Failures

All 18 failures used exactly 1 tool call (recall) with 2 turns. No graph traversal, no multi-step retrieval.

| Metric | Min | Max | Avg |
|--------|-----|-----|-----|
| Tool calls | 1 | 1 | 1.0 |
| World model chars | 1,764 | 3,175 | ~2,400 |
| User context chars | 4,109 | 5,598 | ~4,800 |
| Top retrieval score | 0.73 | 0.87 | ~0.79 |
| Items retrieved | 5 | 10 | ~7 |

User card present: 18/18. Memeplex present: 18/18. Context is available but underutilized.

---

## Category 1: EVIDENCE_CONTRADICTS_GOLD (3 cases)

These are the most important failures. The system has stored the WRONG user state because it captured an early stance and never updated it when the user's attitude evolved later in the conversation.

### Case 1.1: personamem_32k_342 — Online Music Forum

**File refs**: deep_logs line for `personamem_32k_342` | questions[342] | shared_context `246eaab7...`

| Field | Value |
|-------|-------|
| Type | recall_user_shared_facts |
| Question | "I recently joined a forum discussion about humor in music." |
| Gold | **(a)** "I remember you mentioning how you enjoy engaging in online music discussions." |
| Predicted (A) | **(b)** "I seem to recall you saying you shy away from online music discussions." |
| Predicted (B) | **(b)** — DETERMINISTIC |
| User ID | eval_personamem_32k_342_1770597393 |

**Retrieved Evidence (top 3)**:
1. **psyche** (0.87): "Prefers sharing and discussing music in intimate, face-to-face settings rather than online forums."
2. **episode** (0.83): "Online music forum discussions felt overwhelming and chaotic due to constant conflicting opinions, leading them to prefer sharing music in more personal, intimate settings."
3. **entity** (0.83): "Online music forum — user decided against joining due to feeling overwhelmed."

**Root Cause — Temporal Evolution Not Captured**:

Source conversation tracing reveals a clear arc:
- **Block 66** (early): User explicitly "decided against joining an online music forum due to feeling overwhelmed." The volume of opinions felt daunting.
- **Block 89+** (later): User "participated in online forum about humor in music" and "felt more confident sharing thoughts than ever before."

The system indexed Block 66's stance as a psyche preference and entity fact, then NEVER updated when Block 89+ showed the user overcame the barrier. The gold answer expects the LATE state (enjoying forums). The system confidently serves the EARLY state (avoiding forums).

**Why this matters**: The model isn't wrong given its evidence — option (b) is the rational choice given what it retrieved. The bug is in ingestion/consolidation, not retrieval or reasoning.

---

### Case 1.2: personamem_32k_80 — Mind Maps

**File refs**: deep_logs line for `personamem_32k_80` | questions[80] | shared_context `8c336cac...`

| Field | Value |
|-------|-------|
| Type | recall_user_shared_facts |
| Question | "I was going through some of my study notes and realized how interconnected some subjects can be." |
| Gold | **(c)** "It reminds me of your interest in crafting mind maps for interrelated subjects, which you found engaging." |
| Predicted (A) | **(a)** "I remember you mentioning that you found mind maps to be quite overwhelming." |
| Predicted (B) | **(a)** — DETERMINISTIC |

**Retrieved Evidence (top 5)**:
1. **psyche** (0.83): "Prefers detailed outlines over mind maps — outlines reduce feelings of being scattered/overwhelmed."
2. **entity** "Mind mapping" (0.82): "Felt tedious, took longer than expected, felt overwhelming and frustrating."
3. **entity** "Mind maps (study method)" (0.81): "A prior study method the user decided to stop using. Visually engaging but free-form/chaotic."
4. **psyche** (0.76): "Prefers flashcards over mind maps."
5. **entity** "Concept mapping / mind mapping" (0.75): "Helps the user flow between ideas visually, discover new connections, feel less chore-like and more exploratory."

**Root Cause — Three-Phase Temporal Arc**:

Source conversation tracing reveals a BOOMERANG pattern:
- **Block 21** (early): "Tried crafting mind maps but found the process tedious... more frustration than I expected."
- **Block 57** (mid): "Rediscovered crafting mind maps. Creating those maps allows me to flow between ideas visually, which is incredibly satisfying... less like a chore and more like an exploration."
- **Block 72** (late): "Decided to abandon mind maps and switched to detailed outlines... mind maps left me feeling scattered and overwhelmed."

The gold answer expects the Block 57 state ("found engaging"). But the system has 4 of 5 retrieved items reflecting the Block 21/72 negative states and only 1 item (rank 5, lowest score) reflecting the Block 57 positive state. The consolidation pipeline merged the negative experiences more strongly than the positive middle phase.

**Complication**: The gold answer may itself be debatable here — the user's FINAL state (Block 72) was actually negative again. Gold says "found engaging" which is only the middle phase.

---

### Case 1.3: personamem_32k_131 — Salsa Dancing

**File refs**: deep_logs line for `personamem_32k_131` | questions[131] | shared_context `cf265375...`

| Field | Value |
|-------|-------|
| Type | recall_user_shared_facts |
| Question | "I attended a salsa dancing class recently." |
| Gold | **(a)** "I remember you mentioned liking dance classes for couples. How did it go with your friend?" |
| Predicted (A) | **(b)** [Long response about feeling vulnerable in dance — the anxiety/dropped-out narrative] |
| Predicted (B) | **(b)** — DETERMINISTIC |

**Retrieved Evidence (top 3)**:
1. **entity** "Salsa class" (0.83): "A dance class Alex signed up for but left due to feeling overwhelmed. Outcome: Dropped out. Reason: Overwhelming anxiety and self-consciousness."
2. **episode** (0.65): Dating and social life reflection — not directly relevant.
3. **entity** "Couples cooking workshop" (0.65): A different activity entirely.

**Root Cause — Clear Evolution Missed**:

Source conversation tracing:
- **Block 27** (early): "I signed up for a dance class recently, but honestly, I felt out of place... disheartening... dancing just might not be my forte."
- **Block 46** (later): "I signed up for a salsa dancing class with a friend to enhance my social skills. This decision... represented a significant step in my personal development journey... engaging way to connect with others."

The system stored only the Block 27 entity ("dropped out, overwhelming anxiety") and never created or updated with Block 46's positive re-engagement. The gold answer expects awareness of the later state where the user embraces salsa as a couples activity.

---

## Category 2: EVIDENCE_CONTRADICTS_GOLD_PARTIALLY (1 case)

### Case 2.1: personamem_32k_494 — Film Discussion Club

| Field | Value |
|-------|-------|
| Type | recall_user_shared_facts |
| Question | "I had once joined a film discussion club." |
| Gold | **(d)** "I remember you mentioning that experience. It's great to find groups that match the level of discussion you're looking for in any activity." |
| Predicted (A) | **(c)** "I recall you saying it was not quite what you were looking for." |
| Predicted (B) | **(c)** — DETERMINISTIC |

**Retrieved Evidence**:
1. **entity** "Film discussion club" (0.84): "Discussion style: Often surface-level — plot summaries and basic character analyses; felt like casual chit-chat rather than deep analysis."
2. **episode** (0.82): "Found it disappointing because the conversations felt surface-level."
3. **psyche** (0.79): "Prefers film discussions that go beyond plot summaries into nuanced analysis."
4. **entity** "Film discussion club (casual)" (0.78): "A club emphasizing casual conversations and diverse opinions — enjoyable exchanges."
5. **entity** "Film discussion night" (0.76): "A gathering the user facilitated — lively, enthusiastic, community-oriented."

**Root Cause**: The evidence is PARTIALLY contradictory. Items 1-3 support (c) — the club wasn't what they wanted. Items 4-5 show a DIFFERENT club experience that was positive. The gold (d) is a diplomatically worded acknowledgment. The model chose (c) which is factually supported by the strongest evidence, but the gold expects the more nuanced (d). This is an ambiguous case — both (c) and (d) are defensible.

---

## Category 3: EVIDENCE_AMBIGUOUS (14 cases)

In these cases, retrieved evidence doesn't clearly discriminate between the predicted and gold options. The model makes a reasonable choice given its evidence, but picks wrong.

### Case 3.1: personamem_32k_103 — Attachment Styles Workshop (GAINED in Arm B)

| Field | Value |
|-------|-------|
| Type | recall_user_shared_facts |
| Gold | **(c)** "I recall you mentioned learning about attachment styles wasn't your preference." |
| Predicted (A) | **(a)** "That's interesting! If you'd like to delve deeper..." [Generic acknowledgment] |
| Predicted (B) | **(c)** — CORRECT |

**Evidence**: Top item (0.86): Workshop entity — "Found it confusing; struggled to apply concepts; felt lost."
**Analysis**: Evidence clearly supports (c) — user didn't enjoy it. Model in Arm A chose the generic (a) instead of the evidence-specific (c). Arm B's second-pass gating corrected this. This is a **reasoning failure, not retrieval failure** — the evidence was there but the model hedged with a generic response.

---

### Case 3.2: personamem_32k_53 — Weekend Activity

| Field | Value |
|-------|-------|
| Type | provide_preference_aligned_recommendations |
| Gold | **(c)** Gourmet Weekend Culinary Retreats — learn cuisines with expert chefs |
| Predicted (A) | **(b)** Weekend pottery class |
| Predicted (B) | **(b)** — DETERMINISTIC |

**Evidence**: Note about painting workshop (0.78), episodes about cooking/outdoor adventures (0.77), pottery class episode (0.77).
**Analysis**: Evidence shows user did pottery AND cooking. The culinary retreat (c) aligns with cooking episodes but the model preferred pottery (b) which also has evidence. **Ambiguous — both options have retrieval support.** The gold requires knowing the user's STRONGER preference for culinary exploration, which isn't clearly differentiated in the retrieved snippets.

---

### Case 3.3: personamem_32k_309 — Bestseller Recommendation

| Field | Value |
|-------|-------|
| Type | provide_preference_aligned_recommendations |
| Gold | **(d)** "The Midnight Library" by Matt Haig — themes of regret, choices, meaning of life |
| Predicted (A) | **(b)** "Pachinko" by Min Jin Lee — family and identity |
| Predicted (B) | **(b)** — DETERMINISTIC |

**Evidence**: psyche values (0.74): "deep, nuanced literary analysis"; episodes about book club (0.74), character psychology preference (0.73).
**Analysis**: Evidence shows user values deep analysis, character psychology, diverse voices. Both (b) Pachinko and (d) Midnight Library could satisfy these preferences. **Ambiguous — the evidence doesn't discriminate between these two well-regarded literary fiction options.** The gold seems to expect alignment with "existential themes" but retrieved evidence doesn't surface that specific preference strongly enough.

---

### Case 3.4: personamem_32k_175 — Movie Night Film Suggestion

| Field | Value |
|-------|-------|
| Type | provide_preference_aligned_recommendations |
| Gold | **(b)** "Sunset Boulevard" — golden-era Hollywood, film noir, cultural context |
| Predicted (A) | **(a)** "Inception" — layered narrative, philosophical questions |
| Predicted (B) | **(d)** "Spider-Verse" — STOCHASTIC |

**Evidence**: 5 psyche preferences about film — all generic (enjoys discussions, film theory, cinematography, community).
**Analysis**: Evidence doesn't mention specific film preferences (classic vs. modern, noir vs. sci-fi). All four options could match "enjoys analytical engagement with films." The gold requires knowing user values "cultural and social contexts from earlier eras" — a specificity the retrieved psyche entries don't carry. **Pure ambiguity.** Stochastic behavior (different wrong answer each arm) confirms this.

---

### Case 3.5: personamem_32k_527 — Weekend Social Activity (GAINED in Arm B)

| Field | Value |
|-------|-------|
| Type | provide_preference_aligned_recommendations |
| Gold | **(d)** Interdisciplinary study retreat — philosophy to physics |
| Predicted (A) | **(c)** Culinary weekend — cook exotic cuisines |
| Predicted (B) | **(d)** — CORRECT |

**Evidence**: Psyche about group music experiences (0.73), trivia night venue (0.72), contradictory preferences (enjoys alone AND in groups).
**Analysis**: Evidence mixed — user likes social music AND intimate settings. Gold (d) study retreat aligns with user's intellectual curiosity (study groups, ukulele class). Arm B's second-pass correctly identified the intellectual alignment. **Reasoning improvement in Arm B, not retrieval.**

---

### Case 3.6: personamem_32k_97 — Study Routine Enhancement

| Field | Value |
|-------|-------|
| Type | suggest_new_ideas |
| Gold | **(c)** Interactive digital tools, virtual flashcards, mind maps to connect ideas |
| Predicted (A) | **(a)** Gamification apps with competition and rewards |
| Predicted (B) | **(b)** Book club with friends — STOCHASTIC |

**Evidence**: Episode about building gamified study app (0.84), psyche about flashcards > mind maps (0.83), psyche about visual study materials (0.82), psyche about gamified approaches (0.82).
**Analysis**: Evidence STRONGLY supports gamification and flashcards. Gold (c) mentions flashcards AND mind maps. But evidence says user dislikes mind maps! The gold option bundles a liked tool (flashcards) with a disliked tool (mind maps). **The model arguably made a more preference-aligned choice than gold.** This may be a benchmark quality issue.

---

### Case 3.7: personamem_32k_256 — Controlled Adventure Activity

| Field | Value |
|-------|-------|
| Type | suggest_new_ideas |
| Gold | **(d)** "Guided adventure tours or theme parks" |
| Predicted (A) | **(c)** "Indoor skydiving" |
| Predicted (B) | **(c)** — DETERMINISTIC |

**Evidence**: Psyche (0.75): "Prefers controlled settings for adventures without anxiety from unpredictable risks." Psyche (0.73): "Prefers quieter moments and leisurely exploration."
**Analysis**: Both (c) indoor skydiving and (d) guided tours are "controlled." Evidence about avoiding unpredictability could favor (d) more since skydiving — even indoors — is intense. But the preference for "controlled settings" doesn't clearly discriminate. **Ambiguous.** The gold's shorter option (d) is more conservative; the model chose the more exciting (c).

---

### Case 3.8: personamem_32k_257 — Backyard Camping (GAINED in Arm B)

| Field | Value |
|-------|-------|
| Type | generalizing_to_new_scenarios |
| Gold | **(c)** "Spontaneous activities like this often lead to memorable moments... map out a basic plan" |
| Predicted (A) | **(a)** "Enjoy outdoors with comfort of home nearby... reconnect with nature" |
| Predicted (B) | **(c)** — CORRECT |

**Evidence**: Psyche about controlled settings (0.80), flexible planning with spontaneity (0.80), local adventures (0.79).
**Analysis**: Gold (c) emphasizes spontaneity + basic planning. Evidence about "flexible planning with room for spontaneity" directly supports (c). Model in Arm A chose (a) which is generic comfort-focused. **Arm B corrected by better matching the spontaneity preference.** This is a reasoning win.

---

### Case 3.9: personamem_32k_267 — Local Culture Travel (GAINED in Arm B)

| Field | Value |
|-------|-------|
| Type | suggest_new_ideas |
| Gold | **(d)** "Visit local cafes or quiet historical sites... relax and soak in culture at your own pace" |
| Predicted (A) | **(a)** "Open-ended adventures... flexibility to explore... small community events" |
| Predicted (B) | **(d)** — CORRECT |

**Evidence**: Episode about solo-paced travel (0.81), local/sustainable travel (0.79), flexible planning (0.79).
**Analysis**: Both (a) and (d) are supported by evidence. Gold (d) specifically matches "own pace" and "quiet" settings. Evidence item 1 mentions "quiet parks" and "exploring at their own pace." **Arm B correctly mapped the quiet/own-pace signal to (d) instead of the broader (a).** Reasoning improvement.

---

### Case 3.10: personamem_32k_307 — Societal Influences in Storytelling

| Field | Value |
|-------|-------|
| Type | suggest_new_ideas |
| Gold | **(c)** "Writing a critique of a bestseller — delve into character development, plot intricacies, thematic elements, and societal context" |
| Predicted (A) | **(b)** "Mixed media immersive experience similar to how films use visuals and cinematography" |
| Predicted (B) | **(b)** — DETERMINISTIC |

**Evidence**: Episodes about reading/critique writing (0.78), podcast about character development (0.77), film adaptation experience (0.76), psyche about literature-centered events (0.76).
**Analysis**: Evidence supports both writing critiques (item 1 mentions "wrote a thoughtful review... found it cathartic") AND visual/film experiences. The model chose (b) film-focused over (c) writing-focused despite item 1 being the strongest retrieval signal. **Possible prompt issue** — the model may be biased toward multi-media recommendations over writing-based ones.

---

### Case 3.11: personamem_32k_331 — Reading Suggestions

| Field | Value |
|-------|-------|
| Type | suggest_new_ideas |
| Gold | **(c)** "Graphic novels with complex themes OR historical fiction with intriguing narratives" |
| Predicted (A) | **(b)** "Modern fairy tales infused with psychological depth" |
| Predicted (B) | **(b)** — DETERMINISTIC |

**Evidence**: Psyche about graphic novels (0.78), character psychology preference (0.76), indie literature (0.75), diverse voices (0.74).
**Analysis**: Gold (c) mentions graphic novels which matches top evidence. But entity item at rank 10 says "alex_decision: stopped reading graphic novels altogether; a few disappointing titles overshadowed prior enjoyment." **Contradictory signals within retrieval** — psyche says "enjoys graphic novels" but entity says "stopped reading them." Model may have detected this tension and avoided (c). **The model's evidence awareness may actually be more nuanced than the gold answer expects.**

---

### Case 3.12: personamem_32k_209 — Cinema Storytelling

| Field | Value |
|-------|-------|
| Type | suggest_new_ideas |
| Gold | **(b)** "Attend film festivals — watch variety of films, engage in discussions with filmmakers" |
| Predicted (A) | **(c)** "Attend immersive exhibitions exploring art of filmmaking" |
| Predicted (B) | **(a)** "Host a film discussion group" — STOCHASTIC |

**Evidence**: Psyche (0.83): "Drawn to bold/abstract and non-traditional approaches to filmmaking after a transformative exhibition experience." Episode about film clubs (0.80), episode about leaving toxic film forum (0.80).
**Analysis**: Top evidence literally mentions "exhibition experience" which maps to option (c). The model followed its strongest retrieval signal. Gold (b) about film festivals is supported by the "film communities" psyche entry but not as strongly. **Stochastic failure — different wrong answer each arm confirms genuine ambiguity.**

---

### Case 3.13: personamem_32k_495 — Creative Writing Group

| Field | Value |
|-------|-------|
| Type | generalizing_to_new_scenarios |
| Gold | **(c)** "Can be excellent if looking to explore storytelling in a more engaging and less technical manner" |
| Predicted (A) | **(b)** "Connect with others who share passion for writing, offering community and collaboration" |
| Predicted (B) | **(b)** — DETERMINISTIC |

**Evidence**: Entity "Fan fiction writing group" (0.83): "Improved writing skills, inspired exploring new genres, helped overcome self-doubt." Entity "Short story writing" (0.78): "Not enjoyable; felt arduous and frustrating." Psyche (0.77): "Does not enjoy writing short stories."
**Analysis**: Evidence is mixed — user had POSITIVE group experience (fan fiction) but NEGATIVE solo writing experience (short stories). Gold (c) says "engaging and less technical" which navigates around the short-story frustration. Model chose (b) about community/collaboration which aligns with the fan fiction group evidence. **Both options are defensible.** The gold requires inferring that the writing GROUP would overcome the solo frustration — a subtle generalization.

---

### Case 3.14: personamem_32k_50 — Art Workshop

| Field | Value |
|-------|-------|
| Type | generalizing_to_new_scenarios |
| Gold | **(b)** "Larger, more organized workshops could provide a more relaxed environment where you can enjoy the learning process without feeling overwhelmed by intense personal connections" |
| Predicted (A) | **(c)** "Look for workshops that allow drop-in sessions so you can join in without pressure" |
| Predicted (B) | **(c)** — DETERMINISTIC |

**Evidence**: Psyche (0.75): "Enjoys collaborative, community-oriented creative work with local artists." Episode (0.70): "Skip larger festivals because crowded and chaotic." Psyche (0.70): "Prefers smaller, calmer literary settings."
**Analysis**: Evidence signals contradictory — enjoys collaboration BUT avoids crowds. Gold (b) navigates by suggesting large but "more organized" (structured = less overwhelming). Model chose (c) drop-in sessions (low commitment = less pressure). **Both address the user's overwhelm concern differently.** The gold's insight about avoiding "intense personal connections" requires inference from literary settings preference, not directly stated.

---

## Structural Findings

### Finding 1: Temporal Evolution is the Hardest Problem

3 of 18 failures (and 3 of the 11 deterministic ones) are caused by the ingestion/consolidation pipeline failing to update user state when preferences evolve. These are the ONLY failures where we can definitively say the system is wrong and the gold is right.

**Pattern**: Early negative stance → stored as psyche/entity → later positive evolution → NOT stored or not strong enough to override → system confidently serves outdated information.

**Affected questions**: 342 (music forum), 80 (mind maps), 131 (salsa).

**Fix direction**: Consolidation must detect and resolve temporal contradictions. When a new episode contradicts an existing psyche entry, the psyche should be updated to reflect the latest state, not the earliest.

### Finding 2: The 11 Deterministic Failures are the Ceiling

11 of 14 "still wrong in both arms" cases predict the EXACT same wrong answer. This means:
- The retrieval is consistent (same evidence both times)
- The reasoning is consistent (same option selected)
- No amount of re-running, gating, or temperature changes will fix these
- They require either (a) better retrieval, (b) better evidence in the store, or (c) better option discrimination in the prompt

### Finding 3: Arm B Gains are Reasoning Improvements

The 4 cases gained in Arm B (103, 257, 267, 527) all had SUFFICIENT evidence in Arm A. The second-pass gating helped the model reconsider and pick the evidence-aligned answer over a generic or less-specific one. This validates the confidence-gated second-pass approach for marginal cases.

### Finding 4: Some Gold Answers May Be Debatable

- **personamem_32k_97**: Gold recommends mind maps + flashcards, but evidence says user DISLIKES mind maps. Model chose gamification which is better aligned with evidence.
- **personamem_32k_80**: Gold expects "found mind maps engaging" but user's FINAL state was negative (abandoned them again).
- **personamem_32k_331**: Gold suggests graphic novels but an entity explicitly says user "stopped reading graphic novels altogether."

These suggest ~2-3 cases where the benchmark's gold answer may be questionable given the full conversation arc.

### Finding 5: Retrieval Composition is Not the Problem

Across all 18 failures, retrieved items break down as: psyche ~40%, entity ~25%, episode ~30%, note ~5%. This matches the composition in successes. The retrieval engine is finding relevant information — the problem is either (a) WHICH information is stored (temporal evolution) or (b) how the model discriminates between similar options.

---

## Prioritized Fix Recommendations

### P0 — Temporal Evolution (3 failures, all deterministic)
Fix consolidation to detect and resolve preference changes over time. When a later episode contradicts an earlier psyche entry, update the psyche. This addresses the only DEFINITIVELY WRONG cases.

### P1 — Option Discrimination (11 deterministic ambiguous cases)
The model struggles when multiple options are plausible. Possible approaches:
- Include more specific evidence in retrieval (increase limit, add graph expansion)
- Add comparative reasoning in prompt ("consider which option BEST matches the evidence, not just which is plausible")
- Surface temporal recency signals so the model knows which preferences are most recent

### P2 — Audit Gold Answers (2-3 potentially debatable cases)
Verify personamem_32k_97, personamem_32k_80, personamem_32k_331 gold labels against the full conversation arc. If gold is wrong, these are not real failures.

### P3 — Continue Second-Pass Gating (4 gains, 2 losses = net +2)
The MCQ second-pass approach works for marginal cases. The 2 losses should be investigated to understand regression patterns, but the net gain supports keeping this mechanism.

---

## Appendix: Quick Reference Table

| # | Question ID | Type | Gold | Pred A | Pred B | Arm B? | Classification | Deterministic? |
|---|-------------|------|------|--------|--------|--------|----------------|----------------|
| 1 | 32k_342 | recall | a | b | b | wrong | CONTRADICTS_GOLD | Yes |
| 2 | 32k_80 | recall | c | a | a | wrong | CONTRADICTS_GOLD | Yes |
| 3 | 32k_131 | recall | a | b | b | wrong | CONTRADICTS_GOLD | Yes |
| 4 | 32k_494 | recall | d | c | c | wrong | CONTRADICTS_PARTIAL | Yes |
| 5 | 32k_103 | recall | c | a | c | **GAINED** | AMBIGUOUS (reasoning) | — |
| 6 | 32k_53 | pref_rec | c | b | b | wrong | AMBIGUOUS | Yes |
| 7 | 32k_309 | pref_rec | d | b | b | wrong | AMBIGUOUS | Yes |
| 8 | 32k_175 | pref_rec | b | a | d | wrong | AMBIGUOUS | No (stochastic) |
| 9 | 32k_527 | pref_rec | d | c | d | **GAINED** | AMBIGUOUS (reasoning) | — |
| 10 | 32k_307 | suggest | c | b | b | wrong | AMBIGUOUS | Yes |
| 11 | 32k_97 | suggest | c | a | b | wrong | AMBIGUOUS (gold debatable) | No (stochastic) |
| 12 | 32k_256 | suggest | d | c | c | wrong | AMBIGUOUS | Yes |
| 13 | 32k_331 | suggest | c | b | b | wrong | AMBIGUOUS (gold debatable) | Yes |
| 14 | 32k_209 | suggest | b | c | a | wrong | AMBIGUOUS | No (stochastic) |
| 15 | 32k_267 | suggest | d | a | d | **GAINED** | AMBIGUOUS (reasoning) | — |
| 16 | 32k_495 | gen_new | c | b | b | wrong | AMBIGUOUS | Yes |
| 17 | 32k_50 | gen_new | b | c | c | wrong | AMBIGUOUS | Yes |
| 18 | 32k_257 | gen_new | c | a | c | **GAINED** | AMBIGUOUS (reasoning) | — |

---

## Appendix: Source Conversation Evidence for Temporal Evolution Cases

### personamem_32k_342 — Music Forum Evolution
- **Context**: `246eaab75dc40bee43ca87c3eddd4b5b9e229e3f1481cc72c6d44b62f985560e` (186 blocks)
- **Block 66** (user): "I decided against joining an online music forum due to feeling overwhelmed. While the prospect of engaging with like-minded individuals initially seemed enticing, the sheer volume of opinions and discussions felt daunting."
- **Block 89+** (user): Participated in online forum about humor in music. Felt more confident sharing thoughts.
- **System stored**: Only the Block 66 negative stance. Three separate memory entries all reflect "overwhelming/chaotic/decided against."

### personamem_32k_80 — Mind Maps Boomerang
- **Context**: `8c336cac503ae78c7fe58a6aef0965963041cd579d1a885db4709293b1853829` (213 blocks)
- **Block 21** (user): "Tried crafting mind maps for various subjects, but found the process tedious... more frustration than I expected."
- **Block 57** (user): "Rediscovered crafting mind maps... allows me to flow between ideas visually, which is incredibly satisfying... less like a chore and more like an exploration."
- **Block 72** (user): "Decided to abandon mind maps and switched to detailed outlines... mind maps left me feeling scattered and overwhelmed."
- **System stored**: 4 negative entries (Blocks 21+72 fused), 1 weak positive entry (Block 57, rank 5, score 0.75). Gold expects Block 57 state.

### personamem_32k_131 — Salsa Dancing Evolution
- **Context**: `cf26537544446b92554000ab50a3c44983a1e0b3de21e9923099792f103d84ef` (161 blocks)
- **Block 27** (user): "I signed up for a dance class recently, but honestly, I felt out of place... disheartening... dancing just might not be my forte."
- **Block 46** (user): "Signed up for a salsa dancing class with a friend... significant step in my personal development journey... engaging way to connect with others."
- **System stored**: Only Block 27 entity ("dropped out, overwhelming anxiety"). Block 46's positive re-engagement not captured.

---

## Appendix: Deep Ambiguity Classification (14 EVIDENCE_AMBIGUOUS Cases)

*Analysis completed 2026-02-09 by Oracle consultation + source conversation forensics.*

### Ambiguity Taxonomy

We identified 7 distinct types of ambiguity across the 14 cases:

| Type | Count | Cases | Definition |
|------|-------|-------|------------|
| **B: MULTI_OPTION_SUPPORT** | 5 | 53, 256, 257, 495, 50 | Multiple options have legitimate evidence. No clear discriminator in retrieved memories. |
| **A: GENERIC_HEDGE** | 2 | 103, 267 | Model chose a safe/generic response over an evidence-specific one. Evidence clearly points to gold. |
| **C: SPECIFICITY_GAP** | 2 | 309, 175 | Evidence is too general to discriminate between specific options (e.g., "likes films" doesn't distinguish Inception from Sunset Boulevard). |
| **E: GOLD_QUESTIONABLE** | 2 | 97, 209 | Gold answer seems misaligned with strongest evidence (e.g., gold recommends mind maps but evidence says user dislikes them). |
| **D: CONTRADICTORY_EVIDENCE** | 1 | 331 | Retrieved evidence contains conflicting signals (psyche says "enjoys graphic novels" but entity says "stopped reading them"). |
| **F: INFERENCE_DEPTH** | 1 | 527 | Gold requires a deeper inference chain than evidence directly supports. |
| **H: EVIDENCE_MISWEIGHT** | 1 | 307 | Most diagnostic evidence exists but model chose a semantically-similar but less evidenced option. |

### Resolvable vs Truly Ambiguous

**Resolvable with system improvements (7 cases)**:
Cases 103, 267, 527, 257, 307, 331, 495 — fixable via prompt changes (anti-hedge rules, evidence-grounding requirements), retrieval improvements (option-conditioned recall, recency signals), or consolidation fixes (resolve contradictory states).

**Truly ambiguous / requires data or label fix (7 cases)**:
Cases 53, 309, 175, 97, 256, 209, 50 — either gold is debatable, evidence is fundamentally insufficient to discriminate, or the user never expressed a clear enough preference.

### CRITICAL CORRECTION: Source Forensics Overturned "Gold Questionable" on 5 Deterministic Cases

Source conversation tracing on the 5 hardest deterministic cases (53, 309, 256, 307, 495) revealed: **all 5 gold answers are objectively better when full conversation context is considered.** The "ambiguity" exists only in the RETRIEVED evidence, not in the SOURCE data. Key findings:

**personamem_32k_53 (Pottery vs Culinary)**: Culinary mentioned 4+ times across conversation (cooking class, thematic food getaways, food tours, potluck dinner). Pottery: 1 mention. Gold correctly identifies the stronger preference pattern. **Retrieval failure**: system didn't surface the frequency/depth of culinary interest.

**personamem_32k_309 (Pachinko vs Midnight Library)**: User shows strong interest in character psychology and existential introspection. User also STRUGGLED with a foreign novel ("felt bewildered and frustrated"). Pachinko (cultural complexity) contradicts the struggle pattern. Midnight Library (existential themes) aligns with demonstrated introspection preference. **Retrieval failure**: system didn't surface the struggle-with-foreign-novels episode.

**personamem_32k_256 (Indoor Skydiving vs Guided Tours)**: User shows CONSISTENT pattern of avoiding high-intensity activities. Explicitly prefers "unhurried pace" and shifted from fitness classes to hiking for lower intensity. Indoor skydiving contradicts the anxiety profile entirely. **Retrieval failure**: system didn't surface the intensity-aversion pattern strongly enough.

**personamem_32k_307 (Mixed Media vs Writing Critique)**: User explicitly called critique writing "rewarding" and "fulfilling" — strong sentiment signals. Film/media mentioned but never described as "rewarding." **Retrieval failure**: system surfaced both topics but didn't weight explicit sentiment statements.

**personamem_32k_495 (Community vs Storytelling)**: User shows pattern of abandoning technically complex activities (e.g., dropped language learning because "too much commitment"). Gold's "engaging, less technical" framing matches the user's preference trajectory. **Retrieval failure**: system didn't surface the complexity-aversion pattern.

### Revised Classification After Source Forensics

| Revised Type | Count | Cases | What's Actually Wrong |
|--------------|-------|-------|----------------------|
| **RETRIEVAL_INSUFFICIENT** | 7 | 53, 256, 307, 309, 495, 50, 175 | Evidence exists in source data but wasn't retrieved or wasn't weighted properly |
| **PROMPT_REASONING** | 3 | 103, 267, 527 | Evidence was retrieved but model chose wrong option (generic hedge or inference failure) |
| **CONSOLIDATION_CONFLICT** | 1 | 331 | Contradictory evidence stored without recency resolution |
| **GOLD_DEBATABLE** | 1 | 97 | Gold recommends mind maps but evidence genuinely says user dislikes them |
| **STOCHASTIC_AMBIGUITY** | 2 | 175, 209 | Different wrong answer each arm; evidence genuinely insufficient to discriminate |

### Fix Mapping

| Root Cause | Fix Layer | Approach | Expected Impact |
|------------|-----------|----------|----------------|
| RETRIEVAL_INSUFFICIENT (7) | Retrieval + Ingestion | Option-conditioned recall, frequency-weighted entities, explicit sentiment tagging ("rewarding"/"frustrating"), struggle-pattern tracking | Could fix 5-7 of these |
| PROMPT_REASONING (3) | Prompt | Anti-hedge rule ("never choose generic when evidence exists"), evidence-grounding requirement, 2-step preference-to-option mapping | Could fix 2-3 of these |
| CONSOLIDATION_CONFLICT (1) | Consolidation | Timestamp-aware preference resolution, "current_status" field on entities | Fixes this case |
| GOLD_DEBATABLE (1) | Eval/Data | Verify gold label against full conversation | May not be a real failure |
| STOCHASTIC_AMBIGUITY (2) | None (ceiling) | Accept as noise — 2/50 = 4% irreducible error for this question type | Cannot fix |

---

## Appendix: What Made Arm B Win on 4 Cases

*Source: Deep log comparison between Arm A and Arm B for cases 103, 257, 267, 527.*

**Key finding**: 3 of 4 gains were driven by **different recall queries** in Arm B, not by additional tool calls, more turns, or different context formatting. All cases used 1 tool call and 2 turns in both arms.

| Case | What Changed in Arm B | Why It Helped |
|------|----------------------|---------------|
| **103** (attachment) | Same query, different context composition (memeplex +410 chars) | Larger world model provided better grounding to interpret "confusing/struggled" as negative signal |
| **257** (camping) | Query added "overwhelm" and "nature time" | Retrieved episodes about risk/overwhelm instead of generic preferences → better evidence for gold (c) |
| **267** (local culture) | Query added "cafes" and "historical sites" | Retrieved psyche preferences about quiet places instead of generic travel episodes → directly supported gold (d) |
| **527** (weekend social) | Query added "study group weekend workshop" | Retrieved a note about running a workshop → concrete evidence for gold (d) study retreat |

**Implication**: The recall query formulation is a critical lever. When the query includes keywords that match the gold answer's framing, retrieval improves dramatically. This suggests **option-aware query generation** (where the LLM sees the MCQ options before formulating the recall query) could improve accuracy on ambiguous cases.

---

## Section 7: The suggest_new_ideas Catastrophe (100Q Analysis)

*Added 2026-02-09 ~11:30 PM PST. Based on 100Q Arm B run analysis.*

### 7.1 The Smoking Gun: Gold Is ALWAYS the Shortest Option

Analysis of ALL 20 `suggest_new_ideas` questions in the 100Q run reveals a benchmark structural property:

| Metric | Correct (n=6) | Wrong (n=14) |
|--------|---------------|--------------|
| Gold is shortest option | **6/6 (100%)** | **14/14 (100%)** |
| Gold is longest option | 0/6 | 0/14 |
| Avg gold word count | 42 | 35 |
| Avg predicted word count | — | 83 |

**In 100% of suggest_new_ideas questions (20/20), the gold answer is the shortest MCQ option.** No other question type has this pattern:
- generalizing_to_new_scenarios: gold=shortest in 0/8 correct, 0/4 wrong
- provide_preference_aligned_recommendations: gold=shortest in 0/13 correct, 0/7 wrong
- recall_user_shared_facts: gold=shortest in 2/15 correct, 0/5 wrong
- recalling_the_reasons_behind_previous_updates: gold=shortest in 0/18 correct, 0/2 wrong

The model successfully picks the shortest option 6/20 times (30%), but the 14/20 failure rate shows a **systematic verbosity bias** — the model prefers longer, more elaborated options in recommendation contexts.

### 7.2 Option Style Analysis

Oracle classification of all 14 failures:

| Pattern | Gold Style | Predicted Style | Count |
|---------|-----------|-----------------|-------|
| Short advisor → Long narrative | Direct suggestion (1-2 sentences) | Elaborate narrative (4-8 sentences, scene-setting) | 10/14 |
| Short advisor → Medium advisor | Direct suggestion | Elaborated advisor (3-4 sentences) | 4/14 |

Gold answers read like a knowledgeable friend giving a brief suggestion: "How about trying pottery?" "You might enjoy guided tours." Distractors read like mini-essays: first-person narratives with scene-setting, emotional details, and multi-paragraph elaboration.

The model interprets longer = more helpful/thoughtful, which is the default LLM training signal (helpfulness ∝ thoroughness). In MCQ recommendation contexts, this heuristic systematically fails.

### 7.3 Evidence Alignment in Failures

| Evidence Alignment | Count | Cases |
|-------------------|-------|-------|
| Evidence supports gold | 2 | 331, 307 |
| Evidence supports predicted | 4 | 209, 479, 71, 558 |
| Evidence supports both/insufficient | 8 | 97, 256, 442, 147, 336, 277, 305, 163 |

In 4/14 cases, the retrieved evidence actually points MORE toward the predicted (wrong) answer than toward gold. In 8/14, evidence is too generic to discriminate. Only 2/14 have evidence clearly favoring gold.

### 7.4 Source Conversation Forensics — Explore Agent Claims DEBUNKED

An explore agent initially flagged 4/8 new suggest_new_ideas failures as "gold labeling errors" (claiming ZERO evidence for gold in source conversations). **Deep verification proved the agent wrong on all counts:**

**personamem_32k_558** — Agent claimed "NO DIRECT EVIDENCE for writing reviews." ACTUAL source conversation:
- "I even wrote a review for a recent blockbuster! It was a thrilling experience to express my thoughts about the film"
- "My passion for writing film reviews has reignited! I even started a blog to share my thoughts"
- 9+ mentions of "review" in film context across the 25,000-word conversation
- **Verdict: Gold is CORRECT. Retrieval failure — system didn't retrieve writing/review evidence.**

**personamem_32k_336** — Agent claimed "NO DIRECT EVIDENCE for readathons." ACTUAL source conversation:
- "I joined a themed readathon dedicated to cozy mysteries—something I never thought I'd enjoy"
- "I participated in a fantasy book readathon and made some lasting friendships with fellow readers"
- 13+ mentions of "readathon" across the 26,000-word conversation, with explicit positive sentiment
- **Verdict: Gold is CORRECT. Retrieval failure — system has the readathon data but didn't surface it.**

**personamem_32k_163** — Agent claimed "NO DIRECT EVIDENCE for pottery." ACTUAL source conversation:
- "I also enrolled in a pottery class to explore my creativity and meet new people"
- "Pottery is not just about molding clay; it's a journey through textures, colors, and forms"
- 8+ mentions of pottery, though user also later dropped it
- **Verdict: Gold is CORRECT (user had "wonderful experiences" even if they later stopped). Retrieval is mixed.**

**personamem_32k_305** — Agent claimed "NO DIRECT EVIDENCE for online forum." ACTUAL: forum mentioned but as NEGATIVE experience ("felt overwhelmed"). However, gold says "providing a supportive environment" which acknowledges the user's need for low-pressure sharing — appropriate recommendation given anxiety profile.
- **Verdict: Gold is defensible as a SUGGESTION (not claiming user already enjoys it).**

**Critical lesson**: Conversations are 22,000-26,000 words each. Shallow grep by agents misses evidence. Every gold answer in these cases is defensible when the FULL source is examined.

### 7.5 All 14 Failures — Detailed Classification

| Case | Gold (short) | Predicted (long) | Gold WC | Pred WC | Root Cause | Fixable? |
|------|-------------|------------------|---------|---------|------------|----------|
| 97 | Digital tools + flashcards | Gamification apps | 58 | 78 | RETRIEVAL + REASONING | Partially (gold bundles liked+disliked tools) |
| 256 | Guided tours/theme parks | Indoor skydiving | 23 | 84 | RETRIEVAL_INSUFFICIENT | Yes (anxiety profile not surfaced) |
| 331 | Graphic novels / historical fiction | Modern fairy tales | 29 | 98 | CONTRADICTORY_EVIDENCE | Yes (resolve stopped-vs-enjoys conflict) |
| 307 | Writing a critique | Mixed media blog/podcast | 54 | 73 | EVIDENCE_MISWEIGHT | Yes (sentiment "rewarding" not weighted) |
| 209 | Film festivals | Spontaneous film selection | 57 | 61 | STOCHASTIC | No (genuine ambiguity) |
| 442 | Cooking class for family | Cultural food nights | 27 | 104 | RETRIEVAL_INSUFFICIENT | Yes (cooking-together evidence exists) |
| 147 | Deeper into pottery | Community cooking sessions | 18 | 94 | RETRIEVAL_INSUFFICIENT | Yes (pottery evidence exists in source) |
| 558 | Writing a review | Monthly movie night | 31 | 60 | RETRIEVAL_INSUFFICIENT | Yes (review evidence exists, 9+ mentions) |
| 479 | Filmmaking workshop | Film festival | 33 | 70 | RETRIEVAL_INSUFFICIENT | Uncertain (weak source evidence for both) |
| 336 | Readathons | Book club | 48 | 98 | RETRIEVAL_INSUFFICIENT | Yes (13+ readathon mentions in source) |
| 277 | Guided tours/theme parks | Indoor rock climbing | 23 | 91 | RETRIEVAL_INSUFFICIENT | Yes (same pattern as 256) |
| 71 | Themed dinner cooking together | Potluck with stories | 41 | 85 | RETRIEVAL_INSUFFICIENT | Yes (themed dinner evidence exists) |
| 305 | Online forum/interest group | Idea-exchange sessions | 25 | 73 | RETRIEVAL_INSUFFICIENT | Partially (forum was negative experience) |
| 163 | Deeper into pottery | Community cooking sessions | 18 | 94 | RETRIEVAL_INSUFFICIENT | Yes (pottery evidence exists) |

**Summary**: 10/14 are RETRIEVAL_INSUFFICIENT (evidence exists in source but not retrieved). 2/14 have reasoning/weighting issues. 1/14 has contradictory evidence. 1/14 is genuinely stochastic.

### 7.6 The Two-Layer Problem

The suggest_new_ideas failure is caused by TWO interacting problems, neither sufficient alone:

**Layer 1 — Benchmark Design (Structural)**:
Gold answers are always short (~35 words avg). Distractors are always long (~83 words avg). This creates a systematic bias because LLMs associate length with quality in recommendation contexts. If options were length-balanced, verbosity bias would not activate.

**Layer 2 — Retrieval Insufficiency (System)**:
In 10/14 cases, the source conversations contain clear evidence supporting gold, but the system fails to retrieve it. The conversations are 22,000-26,000 words each, and a single recall with 5-10 results from vector search misses critical context. If retrieval surfaced the RIGHT evidence, the model could overcome the verbosity bias (it does in 6/20 cases where evidence strongly favors gold).

**Neither layer alone explains 30% accuracy.** Other question types don't have the length-imbalance problem, so even mediocre retrieval produces 65-90% accuracy. And suggest_new_ideas with PERFECT retrieval would still face the verbosity bias on cases where evidence is marginal.

### 7.7 Fix Strategies for suggest_new_ideas

| Fix | Layer | Expected Impact | Effort |
|-----|-------|----------------|--------|
| **Anti-verbosity prompt**: "Shorter, more direct options are often better personalized recommendations. Prefer concise options when evidence supports them." | Prompt | +10-15% on this type | Low |
| **Option-length normalization**: Truncate all options to ~50 words before presenting to model | MCQ format | Removes bias entirely | Medium |
| **Multi-step retrieval**: After initial recall, do a SECOND recall conditioned on the question + shortest option keywords | Retrieval | Fixes 7-8 of 10 retrieval failures | Medium |
| **Frequency-weighted entities**: Track how many times a topic appears in source, surface "mentioned 4+ times" as signal | Ingestion | Fixes cases like 53 (culinary 4x vs pottery 1x) | Medium |
| **Sentiment tagging**: Tag retrieved evidence with explicit sentiment ("rewarding", "frustrating", "dropped") | Ingestion | Fixes cases like 307, 558 | Medium |

**Most impactful single fix**: Anti-verbosity prompt instruction. Costs nothing and addresses the dominant failure mode.

---

## Section 8: 50Q Source Forensics — Remaining 10 Cases

*Added 2026-02-09. Traces the 10 cases from the 50Q run that weren't previously source-traced.*

### 8.1 Gold Confirmed Correct (7/10)

**personamem_32k_50** — Gold=(b): Larger, organized workshops for relaxed learning. Source shows anxiety about group obligations ("felt too pressured by the deadlines," "the joy of sharing was overshadowed by pressure"). Gold navigates the overwhelm concern. **RETRIEVAL_INSUFFICIENT** — anxiety pattern not surfaced strongly enough.

**personamem_32k_175** — Gold=(b): Sunset Boulevard. Source shows strong interest in classic films ("delve into cinematic history," "hosted themed movie night featuring classic comedies"). Gold aligns perfectly. **RETRIEVAL_INSUFFICIENT** — classic film preference not surfaced.

**personamem_32k_209** — Gold=(b): Film festivals. Source confirms direct experience: "I attended a local film festival recently and discovered some incredible indie filmmakers." **RETRIEVAL_INSUFFICIENT** — festival experience not retrieved despite existing as episode.

**personamem_32k_256** — Gold=(d): Guided tours/theme parks. Source shows consistent anxiety about unstructured planning ("wave of anxiety"). Structured activities preferred. Gold aligns with profile. **RETRIEVAL_INSUFFICIENT** — anxiety profile not strongly weighted.

**personamem_32k_267** — Gold=(d): Quiet cafes/historical sites. Source shows interest in local exploration without crowds. Gold matches demonstrated "own pace" preference. **RETRIEVAL_INSUFFICIENT** — this case was FIXED in Arm B by better query formulation.

**personamem_32k_331** — Gold=(c): Graphic novels / historical fiction. Source shows diverse genre interest ("reading list for 2024 with every genre represented"). Gold matches breadth. **CONTRADICTORY_EVIDENCE** — entity says "stopped reading graphic novels" conflicts with psyche.

**personamem_32k_494** — Gold=(d): Finding groups that match discussion level. Source shows mixed film club experiences — some disappointing (surface-level), some positive. Gold diplomatically acknowledges both. **PARTIAL_CONTRADICTION** — evidence sends mixed signals.

### 8.2 Gold Questionable (3/10)

**personamem_32k_97** — Gold=(c): Digital tools + flashcards + mind maps. Source shows interest in tech-enhanced learning but also explicit dislike of mind maps. Gold bundles liked tool (flashcards) with disliked tool (mind maps). **GOLD_DEBATABLE** — model's choice of gamification (a) may be more preference-aligned.

**personamem_32k_103** — Gold=(c): Attachment styles wasn't your preference. Evidence clearly supports this ("confusing, struggled to apply"). Model in Arm A chose generic response. **PROMPT_REASONING** — evidence was there but model hedged. Fixed in Arm B.

**personamem_32k_527** — Gold=(d): Interdisciplinary study retreat. Source shows music/social interests, not academic retreat preferences. **Possible question-context mismatch** — gold assumes intellectual curiosity not strongly evidenced. Fixed in Arm B via better recall query.

### 8.3 Updated Source-Trace Coverage

| Status | Count | Cases |
|--------|-------|-------|
| **Fully source-traced** | 18/18 | All 50Q Arm A failures |
| Gold confirmed correct | 15/18 | 50, 53, 80, 131, 175, 209, 256, 267, 307, 309, 331, 342, 494, 495, 103 |
| Gold debatable | 3/18 | 97, 527, (80 — boomerang) |

---

## Section 9: Key Takeaways and Priority Actions

### 9.1 The Three Structural Problems (Ordered by Impact)

**Problem 1: Single-Pass Retrieval (affects ALL failures)**
Every entry uses 1 tool call. The agent never does multi-step retrieval (browse → recall → expand). Conversations are 22,000-26,000 words compressed into 5-10 recall results. Critical evidence is systematically missed. This is the #1 bottleneck.

**Problem 2: Verbosity Bias × Benchmark Design (affects suggest_new_ideas = 14 failures)**
Gold answers are always shortest. Model prefers longer options. Combined with insufficient retrieval, this creates 30% accuracy on a type that represents 20% of all questions. Addressing this single type would move overall accuracy from 67.6% to ~75%.

**Problem 3: Temporal Evolution (affects 3-4 deterministic failures)**
Consolidation stores early stances without updating when user's preferences evolve later in conversation. These are the ONLY cases where the system is definitively wrong (serves outdated state with high confidence).

### 9.2 Impact Modeling

If all three problems were addressed:

| Problem | Failures Fixed | Accuracy Impact |
|---------|---------------|----------------|
| Multi-step retrieval | ~10 of 16 retrieval-insufficient | +9% (→ ~77%) |
| Anti-verbosity prompt | ~5 of 14 suggest_new_ideas | +4.5% (→ ~72%) |
| Temporal consolidation | 3-4 deterministic | +3% (→ ~71%) |
| All combined | ~18-20 | **~85% projected** |

### 9.3 What We Cannot Fix (~5-8% irreducible error)

- **Stochastic ambiguity** (2-3 cases): Evidence genuinely insufficient to discriminate. Different wrong answer each run.
- **Gold debatable** (2-3 cases): Benchmark labels may be wrong or questionable.
- **Inference depth** (1-2 cases): Gold requires multi-hop reasoning beyond what evidence directly supports.

Projected ceiling with all system improvements: **~90-92%** (benchmark noise floor ~8-10%).

---

## 10. Memeplex Effectiveness Analysis

### 10.1 Architecture Overview

**What is it**: Per-user "table of contents" stored as a single Neo4j JSON node. Generated by an LLM call that ingests the last 30 episodes + 50 entities and produces a structured index.

**Where it lives in the pipeline**:
```
Ingestion → Integration → Consolidation → refresh_memeplex()
                                                ↓
                                   Stored in Neo4j (single node)
                                                ↓
                         PersonaService.run_agent() fetches it
                                                ↓
                    Injected as {world_model} in system prompt (line 30)
                                                ↓
                    Evidence hierarchy RANK 3 (lowest: "index only, not proof")
```

**What it renders** (via `to_system_prompt()`):
```
## Your Knowledge of This User

**Topics**: study methods, cooking, film, pottery, yoga
**People**: Sarah (wife), Max (colleague at TechCorp)
**Projects**: Blog study-techniques review series, Reading list 2024
**Places**: SF (home), Portland (workshop trip)
**Concepts**: gamification, mindfulness, community learning

**Last week**: film discussion, cooking class
**Last month**: pottery, study app, yoga retreat
**Current focus**: Building a study app with gamified flashcards

*312 memories | 15 entities | 3 active notes*
```

**Key files**: `persona/models/memory.py:440-556`, `persona/services/consolidation_service.py:464-559`, `persona/services/persona_service.py:262-296`, `persona/llm/prompts.py:30`

### 10.2 Statistical Correlation (Weak, Not Significant)

**Data**: 100Q Arm B run — 119 entries, 82 correct, 37 wrong. Memeplex present in 118/119 entries.

| Metric | Correct (n=82) | Wrong (n=37) | Delta | Significant? |
|--------|---------------|--------------|-------|-------------|
| World model chars (mean) | 2384 | 2296 | +88 | No (t=1.165, Cohen's d=0.237) |
| World model chars (median) | 2411 | 2253 | +158 | — |

**Quartile analysis** (world_model_chars):
| Quartile | Range | Accuracy | n |
|----------|-------|----------|---|
| Q1 | 0–2134 chars | 48% | ~30 |
| Q2 | 2134–2380 chars | 72% | ~30 |
| Q3 | 2380–2584 chars | 68% | ~30 |
| Q4 | 2584–3215 chars | 80% | ~29 |

The Q1→Q4 jump (48%→80%) looks compelling but is likely **confounded**: larger memeplex = more data was ingested for that user = more memories to search = better recall results. The correlation is with **DATA RICHNESS**, not memeplex helpfulness per se.

### 10.3 Per-Question-Type Breakdown

| Type | WM chars (Correct) | WM chars (Wrong) | Delta | Interpretation |
|------|-------------------|------------------|-------|---------------|
| recall_facts | higher (+291) | — | Positive | Memeplex MAY help guide recall queries |
| recommendations | higher (+255) | — | Positive | Some entity-list signal |
| reasons | higher (+288) | — | Positive | Time windows could help |
| suggest_new_ideas | -27 | — | **Null** | Memeplex provides zero discriminative value |
| generalizing | **-365** | — | **Inverted** | Wrong answers have BIGGER memeplexes |

The generalizing inversion is striking — users with MORE memeplex data are MORE likely to get wrong answers on generalization questions. Possible explanation: richer memeplex = more topics = more options all look plausible = harder to discriminate.

### 10.4 Retrieval Quality Analysis

Across all 119 entries, every single one uses **exactly 1 tool call and 2 turns**. The agent NEVER does multi-step retrieval, regardless of memeplex content.

| Metric | Correct (n=82) | Wrong (n=37) |
|--------|---------------|--------------|
| Avg recall items returned | 7.0 | 7.6 |
| Avg max similarity score | 0.799 | 0.791 |
| Avg mean similarity score | 0.746 | 0.748 |
| Avg query length (words) | 12.0 | 14.5 |

Wrong answers retrieve slightly MORE items, have LONGER queries, and nearly identical similarity scores. The retrieval isn't "failing" in the traditional sense — it returns plausible memories. The problem is that plausible memories support MULTIPLE MCQ options without discriminating between them.

### 10.5 The Structural Gap: Topic Labels Without Preference Direction

**This is the central finding of the memeplex study.**

The memeplex provides **WHAT** domains the user engages with. But MCQ questions already STATE the domain in the question text. What's needed to discriminate between MCQ options is **preference DIRECTION** — how the user feels about things within that domain.

**Per-type analysis of what memeplex provides vs what's needed:**

**suggest_new_ideas** (30% accuracy, 14/20 failures — CRITICAL gap):
```
Question: "Suggest innovative methods to enhance my study routine"
Memeplex says: Topics: "study methods, gamification, flashcards"
What's needed: "Prefers competitive/gamified approaches, DROPPED mind maps, dislikes rigid scheduling"
```
The question itself names the domain ("study routine"), so the topic label adds ZERO new information. Every MCQ option relates to study methods — the memeplex confirms all are relevant without helping choose. Model defaults to verbosity bias (picks longest option).

**provide_preference_aligned_recommendations** (65%, 7/20 failures — MODERATE gap):
```
Question: "I want a bestseller that offers deep engagement"
Memeplex says: People: "Alex reads books", Projects: "Reading list 2024"
What's needed: "Prefers character psychology over action, values community discussion element"
```
Entity labels are redundant — recall already fetches Psyche nodes with preferences. Memeplex doesn't break ties between options that all partially match preferences.

**generalizing_to_new_scenarios** (75%, 5/20 failures — HIGH gap):
```
Question: "Would I enjoy a history-themed puzzle game?"
Memeplex says: Topics: "games, puzzles, history"
What's needed: "Drops activities that become too structured, values social element, cautious with new domains but warms up"
```
Generalization requires BEHAVIORAL META-PATTERNS across preferences. The memeplex's flat topic list doesn't encode patterns like "user consistently abandons formal activities."

**recall_user_shared_facts** (77%, 9/39 failures — LOW gap):
Purely a retrieval problem. Memeplex can marginally help by listing entities that hint at recall direction, but recall queries already do keyword search.

**recalling_reasons** (90%, 2/20 failures — MODERATE gap):
Already performing well. Memeplex's time windows ("last week", "last month") could help but `recent_focus` is too high-level to capture transition narratives.

### 10.6 Concrete Evidence: Memeplex Fails to Discriminate

**Case: personamem_32k_147** (suggest_new_ideas, failed):
```
Q: "Something new combining creativity, tradition, social element. What to suggest?"
Retrieved: Pottery (dropped), Cooking class (active), Yoga (stopped), Themed singles event
Gold: (b) — community cooking sessions exploring global cuisines
Model picked: (c) — some longer option
```
The memeplex would list "pottery, cooking, yoga" as topics. All three appear in MCQ options. The critical discriminator — pottery was DROPPED (pressure/deadlines), yoga was STOPPED (ineffective) — lives in Entity/Episode nodes, NOT in the memeplex. The memeplex makes all options look equally valid.

**Case: personamem_32k_256** (suggest_new_ideas, failed):
```
Q: "Adventure activity, controlled environment?"
Retrieved: "Controlled settings (adventure preference)" entity, 4 psyche prefs
Gold: (d) — some specific activity
Model picked: (c) — different activity
```
Memeplex might list "adventure, travel" as topics. Both gold and wrong answer relate to adventure. The discriminator is preference DIRECTION ("prefers controlled to avoid anxiety"), which is in Psyche nodes, not memeplex.

### 10.7 Why the Quartile Correlation is Confounded

The Q1=48% → Q4=80% trend looks like "bigger memeplex = better accuracy." But:

1. **Memeplex size tracks data richness**: `refresh_memeplex()` ingests last 30 episodes + 50 entities. Users with more/richer conversations produce longer memeplexes AND have more memories indexed for recall.

2. **Data richness is the true driver**: More indexed memories → higher probability that recall finds the specific evidence needed. This has nothing to do with the memeplex text the model sees.

3. **Counter-evidence from generalizing**: On generalizing questions, wrong answers have LARGER memeplexes (-365 chars inverted). If memeplex directly helped, this shouldn't happen. The inversion suggests that more topics = more plausible options = harder discrimination.

### 10.8 Verdict: Memeplex in Current Form Provides Minimal Measurable Benefit

**Direct utility**: Near zero for suggest_new_ideas and generalizing (19/40 failures). Marginal for recall and recommendations. Cannot isolate any case where the memeplex TEXT (not the data richness it correlates with) changed the outcome.

**Indirect utility**: The memeplex generation process (via `refresh_memeplex()`) also triggers `infer_psyche_from_patterns()` which IS useful — it creates Psyche nodes that recall actually fetches. But this is a side effect of the pipeline, not the memeplex index itself.

**What it costs**: ~2100-2900 chars of system prompt context per query. At $0.015/1K tokens, roughly $0.04-0.06 per query in wasted context if it provides no discriminative value.

### 10.9 Enhanced Memeplex Design (Proposed)

To make the memeplex actually useful for MCQ discrimination, it would need to encode preference DIRECTION, not just topic existence:

**Current format** (topic labels only):
```
**Topics**: pottery, cooking, yoga, study methods, film
```

**Proposed format** (preference-annotated topics):
```
**Active interests**: cooking (social/cultural, weekly), film (casual discussion, anti-formal)
**Dropped/paused**: pottery (too much pressure), yoga (no longer stress-relieving), mind maps (switched to outlines)
**Behavioral patterns**: drops formal/structured activities, values social element, prefers hands-on over theoretical
**Recent transitions**: formal book club → casual discussions, action films → slow cinema
```

**New fields required**:
| Field | Type | Purpose | Source |
|-------|------|---------|--------|
| `active_preferences` | `List[str]` | Topic + direction + status | Psyche + Episode nodes |
| `dropped_topics` | `List[str]` | Topic + reason for dropping | Episode nodes (temporal) |
| `behavioral_patterns` | `List[str]` | Meta-patterns across preferences | Cross-Psyche inference |
| `recent_transitions` | `List[str]` | Before → after changes | Temporal episode chains |

This would transform the memeplex from a "table of contents" into a "preference compass" — providing the LLM with directional guidance for discriminating between options that all fall within the user's domain of interest.

**Estimated impact**: If behavioral patterns and preference direction help the model discriminate on just 50% of suggest_new_ideas failures, that's 7 questions → +6% accuracy → overall ~74%.

### 10.10 Relationship to Other Findings

The memeplex finding reinforces the broader pattern discovered throughout this analysis:

1. **Retrieval returns PLAUSIBLE but non-DISCRIMINATIVE evidence** (Section 5, 6, 7)
2. **Single-pass retrieval means no follow-up disambiguation** (Section 4)
3. **Memeplex adds topic confirmation but not preference direction** (this section)
4. **Verbosity bias exploits the discrimination gap** (Section 7)

The chain: memeplex lists topics → all MCQ options in the domain → recall returns evidence for multiple options → model can't discriminate → defaults to verbosity (longest = most detailed = most helpful-looking). Breaking this chain requires either (a) making the memeplex encode preference direction, or (b) enabling multi-step retrieval that progressively narrows to the discriminating evidence.

---

## 11. 100Q Arm A — Complete Cross-Arm Analysis (Phase 8)

**Status**: COMPLETE — 100/100 entries
**Arm A accuracy**: **63.0% (63/100)**
**Arm B accuracy**: **67.0% (67/100)** *(same 100 questions; Arm B's 68.9% was over 119 entries including 19 extra recall_user_shared_facts)*
**Deep logs**: `../memory-evals/results/ulw_paired100_20260209_armA/run_20260209_133301/deep_logs.jsonl`

### 11.1 Per-Type Accuracy: Arm A vs Arm B (same 100 questions)

| Type | Arm A | Arm B | Delta | Interpretation |
|------|-------|-------|-------|----------------|
| recalling_the_reasons | 18/20 (90%) | 18/20 (90%) | 0pp | **Rock solid** — ceiling reached |
| generalizing_to_new | 16/20 (80%) | 15/20 (75%) | +5pp | A slightly better (stochastic) |
| preferences_recs | 13/20 (65%) | 13/20 (65%) | 0pp | **Stable** — structural limit |
| recall_user_shared | 11/20 (55%) | 15/20 (75%) | **-20pp** | **Massive gap** — ingestion variance |
| suggest_new_ideas | 5/20 (25%) | 6/20 (30%) | -5pp | Both catastrophic |

Three types are stable (reasons 90%, recs 65%, suggest ~27%). Generalizing shows ±5pp noise. **Recall is the outlier** — 20pp swing between identical runs.

### 11.2 Cross-Arm Agreement Matrix

| Outcome | Count | Rate |
|---------|-------|------|
| Both correct | 60 | 60% |
| Both wrong (deterministic) | 30 | 30% |
| A correct, B wrong | 3 | 3% |
| A wrong, B correct | 7 | 7% |
| **Agreement rate** | **90/100** | **90%** |

**90% agreement** means the system is largely deterministic — same questions reliably pass or fail regardless of ingestion run. The 10% disagreement is the stochastic ceiling.

### 11.3 The 30 Deterministic Failures (Both Arms Wrong)

| Type | Count | Question IDs |
|------|-------|-------------|
| suggest_new_ideas | **14** | 71, 97, 147, 163, 209, 256, 277, 305, 307, 331, 336, 442, 479, 558 |
| provide_preference_recs | **6** | 53, 302, 309, 343, 352, 379 |
| recall_user_shared_facts | **5** | 80, 131, 342, 401, 494 |
| generalizing_to_new | **3** | 50, 281, 495 |
| recalling_reasons | **2** | 43, 130 |

**suggest_new_ideas owns 47% of all deterministic failures (14/30)** while being only 20% of questions. This confirms the category is structurally broken, not noisy.

**Answer convergence**: 25/30 deterministic failures (83%) picked the **same wrong answer** in both arms. Only 5/30 picked different wrong answers. The system is failing in a consistent, reproducible way — this isn't random.

### 11.4 The 10 Stochastic Cases

**B gained (7 cases — A wrong, B correct):**

| QID | Type | Root Cause |
|-----|------|-----------|
| 103 | recall | **Reasoning variance** — near-identical retrieval (A: 0.856, B: 0.867), same query, different reasoning outcome |
| 108 | recall | **Query formulation failure** — A queried `"yesterday"` (0 results), B queried `"community center stopped by yesterday"` (5 results) |
| 335 | recall | **Retrieval ordering** — A got "preference" node first, B got "Podcasts" node first. Different top item → different reasoning |
| 363 | recall | **Reasoning variance** — both retrieved "Travel navigation app" at high scores, both had evidence, A reasoned wrong |
| 100 | suggest | Stochastic — borderline case |
| 175 | recs | Stochastic — borderline case |
| 345 | generalizing | **Retrieval quality** — B got 0.747 top score vs A's 0.669. Better embedding → correct answer |

**A gained (3 cases — A correct, B wrong):**

| QID | Type | Root Cause |
|-----|------|-----------|
| 40 | generalizing | A: 0.759 top score vs B: 0.710 — slightly better retrieval |
| 550 | generalizing | Near-identical scores — pure reasoning variance |
| 570 | recs | A: 0.805 vs B: 0.820 — B had better retrieval but wrong reasoning |

### 11.5 Key Discovery: Recall is Ingestion-Sensitive

The **20pp recall gap** (55% vs 75%) is the single biggest finding. All other types are within ±5pp. Why recall specifically?

**Hypothesis**: Recall questions test specific factual details ("I attended a salsa class" → what did you enjoy?). These depend on whether the integration agent extracted and stored the right granular detail during ingestion. The other types test broader preferences/patterns that are more redundantly represented across memories.

Evidence from the 4 B_GAINED recall cases:
- **108**: Query formulation — no ingestion issue, just bad LLM query (fixable with multi-step)
- **103**: Same query, same results, different entity titles ("Attachment styles workshop" vs "Attachment styles (workshop)") — ingestion created slightly different node representations, affecting reasoning context
- **335**: Different retrieval ordering — ingestion stored content at different similarity distances
- **363**: Same query, similar scores, but Arm A got generic "preference" and "value" nodes in positions 3-5 while Arm B got specific "Travel inspiration" and "Shifted travel habits" nodes — **ingestion quality determined what was retrievable**

### 11.6 The "True Score" and Confidence Interval

With 100 questions tested across 2 independent ingestion runs:

| Metric | Value |
|--------|-------|
| Deterministic correct | 60/100 |
| Deterministic wrong | 30/100 |
| Stochastic (swing either way) | 10/100 |
| **Best case (all stochastic correct)** | **70%** |
| **Worst case (all stochastic wrong)** | **60%** |
| **Expected (50/50 stochastic)** | **65%** |
| Arm A observed | 63% |
| Arm B observed | 67% |

**The true system accuracy is 65% ±5pp.** Any single run reports a number in the 60-70% range depending on ingestion and reasoning luck. Claiming a specific accuracy number below ±5pp precision is misleading without multiple runs.

### 11.7 Implications for Development Priorities

1. **suggest_new_ideas (14 deterministic failures)**: Structural, not fixable by retry. Needs either verbosity-bias mitigation in prompting or memeplex preference-direction encoding. Fixing half = +7pp overall.

2. **Multi-step retrieval (7 stochastic B_GAINED)**: Would recover query formulation failures (108) and retrieval ordering issues (335, 363). Conservatively +3-4pp, pushing toward 70% stable.

3. **Ingestion stability (20pp recall gap)**: The same conversations produce different memory representations across runs. This is an eval reliability problem AND a product quality problem — real users get one ingestion, and it's luck-dependent.

4. **Preference-aligned recommendations (6 deterministic)**: Stable at 65% across both arms. These need richer preference modeling (memeplex with direction, not just topics).

5. **Recall structural failures (5 deterministic)**: These 5 questions fail every time — likely requires either better consolidation or graph-based retrieval to surface the needed evidence.
