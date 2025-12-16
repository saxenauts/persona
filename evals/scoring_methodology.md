# LongMemEval Scoring Methodology Analysis

## Key Discovery: Two Different Scoring Systems

| Aspect | EverMem/LongMemEval Standard | Our Benchmark |
|--------|------------------------------|---------------|
| **Scoring** | Binary CORRECT/WRONG | 1-5 Grades |
| **Runs** | 3 passes, mean±std | Single run |
| **Output** | % Accuracy | Average grade |

---

## EverMem's Judge Prompt (From prompts.yaml)

```
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.

The generated answer might be much longer, but you should be 
GENEROUS with your grading - as long as it touches on the same 
topic as the gold answer, it should be counted as CORRECT.
```

**Key Points**:
1. Binary scoring: CORRECT or WRONG only
2. "Generous grading" - partial matches count
3. Time references: flexible format matching

---

## LongMemEval Tests 5 Abilities

| Ability | Description | Example |
|---------|-------------|---------|
| **IE** | Information Extraction | Recall explicit details |
| **MR** | Multi-session Reasoning | Aggregate across conversations |
| **TR** | Temporal Reasoning | Time/date understanding |
| **KU** | Knowledge Updates | Track changed info |
| **ABS** | **Abstention** | Correctly refuse when info absent |

**Abstention (ABS) is KEY**: The benchmark explicitly tests whether systems correctly say "I don't know" when information is not present.

---

## Why "I Don't Know" Gets High Scores

**Scenario with Abstention questions**:
- If info IS present → "I don't know" = WRONG
- If info NOT present → "I don't know" = CORRECT

**Our Example 1.2 Issue**:
- Info WAS present (57+25+12+5=99 in sessions)
- Mem0 said "I don't know" → Should be WRONG
- But our judge gave grade 5

**Problem**: We're not distinguishing ABS questions from non-ABS questions.

---

## Ground Truth Verification: Example 1.2

**Question**: How many rare items do I have?

| Item | Count | Session |
|------|-------|---------|
| Vinyl records | 57 | Session 8, 11 |
| Coins | 25 | Session 16 |
| Figurines | 12 | Session 8 |
| Books | 5 | Session 10 |
| **Total** | **99** | |

**Results**:
- Persona: Found 74/99 (missed coins) → Grade 3
- Mem0: "No info" (found 0/99) → Grade 5

**Problem**: Mem0 scored higher despite retrieving nothing.

---

## Recommendations

### 1. Align with LongMemEval Standard
Use binary CORRECT/WRONG instead of 1-5 grades for consistency.

### 2. Fix Judge for Non-ABS Questions
When info IS in sessions, "I don't know" should be WRONG.

### 3. Track Retrieval Separately
Measure what each system retrieves, not just final answer.

### 4. Run Multiple Passes
Like EverMem, run 3 judgment passes and report mean±std.

---

## Files to Reference

| File | Content |
|------|---------|
| `evals/external/EverMemOS/evaluation/config/prompts.yaml` | EverMem's judge prompts |
| `evals/external/EverMemOS/evaluation/src/evaluators/llm_judge.py` | Binary scoring logic |
| `evals/benchmark_runner.py:77-97` | Our 1-5 grading logic |
