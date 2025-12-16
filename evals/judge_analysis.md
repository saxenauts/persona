# Deep Analysis: Judge Behavior Discovery

**Critical Finding**: The LLM judge rewards "I don't know" answers over incorrect attempts.

---

## The Problem

Looking at examples where Mem0 "wins", we discovered a judge calibration issue:

| System | Answer | Grade | Reality |
|--------|--------|-------|---------|
| Persona | "74 rare items" (partial) | 3 | Actually retrieved 3 of 4 data sources |
| Mem0 | "No information" | 5 | Retrieved NOTHING |

**Mem0 gets higher grades by NOT answering.**

---

## Example 1.2: "How many rare items do I have?"

### Ground Truth (Gold = 99)

| Item Type | Count | Session Found In |
|-----------|-------|------------------|
| Rare vinyl records | 57 | Sessions 8, 11 |
| Rare coins | 25 | Session 16 |
| Rare figurines | 12 | Session 8 |
| Rare books | 5 | Session 10 |
| **TOTAL** | **99** | Across 4 sessions |

### What Each System Did

**Persona (Grade 3)**:
```
Found: 57 records + 12 figurines + 5 books = 74
Missed: 25 rare coins from Session 16
```
- Correctly identified 3 of 4 item categories
- Got 74/99 = 75% of items
- Made a calculation attempt

**Mem0 Graph (Grade 5)**:
```
"The information provided does not include any details about rare items"
```
- Found NOTHING
- Made no calculation attempt
- Judge rewarded this with full marks

### The Judge's Logic

The judge prompt appears to reward:
- "Correct decline" when uncertain → Grade 5
- Incorrect but partial answer → Grade 2-3

This means:
- **Mem0's "wins" may be false positives**
- Mem0 retrieves less data, says "don't know"
- Judge rewards the "safe" non-answer

---

## Example 1.3: Wedding Guest Relation

### Ground Truth

Persona found specific wedding details but got wrong relationship.
Mem0 said "no information" and got higher grade.

Same pattern: **Decline to answer beats wrong answer.**

---

## Implications for Benchmark Results

### What We Thought

| Metric | Persona | Mem0 Graph |
|--------|---------|------------|
| Multi-session | 3.08 | **3.27** |
| Conclusion | Mem0 better | |

### What's Actually Happening

| Real Behavior | Persona | Mem0 |
|--------------|---------|------|
| Retrieves data | More | Less |
| Makes attempts | Yes | Often declines |
| Judge rewards | Penalized | Rewarded |

**Mem0's higher scores may reflect less retrieval, not better quality.**

---

## Recommendations

### 1. Fix Judge Prompt

Add: "Declining to answer when data exists should be penalized, not rewarded."

### 2. Separate Metrics

Track separately:
- Retrieval quality (did it find the data?)
- Answer quality (was the answer correct?)
- Decline rate (how often did it say "don't know"?)

### 3. Re-evaluate Results

With fixed judging, Persona may actually outperform Mem0 on multi-session.

---

## Files for Further Analysis

| File | Content |
|------|---------|
| `evals/raw_data_analysis.md` | Full session transcripts |
| `evals/analysis_examples.json` | All 15 extracted examples |
| `evals/data/longmemeval/sampled_benchmark_data.json` | Source data with sessions |
