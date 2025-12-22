# Evaluation Research Notes (Internal - Do Not Push)

> Date: 2024-12-21
> Purpose: Design reference for Persona memory evaluation framework

---

## 1. LongMemEval (ICLR 2025)

**Paper**: https://arxiv.org/abs/2410.10813
**GitHub**: https://github.com/xiaowu0162/LongMemEval
**Project Page**: https://xiaowu0162.github.io/long-mem-eval/

### Question Types (7 types testing 5 abilities)

| Type | Ability | What It Tests |
|------|---------|---------------|
| `single-session-user` | Information Extraction | Recall user-stated details from one session |
| `single-session-assistant` | Information Extraction | Recall assistant-stated details |
| `single-session-preference` | Information Extraction | Infer preferences from conversation |
| `multi-session` | Multi-Session Reasoning | Aggregate/compare across sessions (e.g., "How many X total?") |
| `temporal-reasoning` | Temporal Reasoning | Date calculations, ordering, duration |
| `knowledge-update` | Knowledge Updates | Track changed information over time |
| `*_abs` suffix | Abstention | Correctly refuse when info is absent |

### Dataset Variants

| Variant | Context Size | Sessions | Purpose |
|---------|--------------|----------|---------|
| **LongMemEval_S** | ~115k tokens | ~40 | Standard evaluation |
| **LongMemEval_M** | ~1.5M tokens | ~500 | Stress test |
| **LongMemEval_Oracle** | Evidence only | Variable | Tests reading in isolation (perfect retrieval) |

### Scoring: Binary (CORRECT / WRONG)

Task-specific LLM judge prompts (from evaluate_qa.py):
- Standard: "answer yes if the response contains the correct answer"
- Temporal: Off-by-one errors tolerated
- Knowledge-update: Previous + updated answer = correct if update is right
- Preference: "correct as long as it recalls and utilizes the user's personal information correctly"
- Abstention: "answer yes if the model correctly identifies the question as unanswerable"

---

## 2. PersonaMem (COLM 2025)

**Paper**: https://arxiv.org/abs/2504.14225
**GitHub**: https://github.com/bowen-upenn/PersonaMem
**Dataset**: https://huggingface.co/datasets/bowen-upenn/PersonaMem

### Question Types (7 personalization skills)

| Type | What It Tests |
|------|---------------|
| `recall_user_shared_facts` | Recall static events/interests shared previously |
| `suggest_new_ideas` | Recommend novel items NOT in history |
| `acknowledge_latest_user_preferences` | Recognize most recent preference expressed |
| `track_full_preference_evolution` | Track how preferences shift over time |
| `recalling_the_reasons_behind_previous_updates` | Recall why preferences changed |
| `provide_preference_aligned_recommendations` | Proactive personalized suggestions |
| `generalizing_to_new_scenarios` | Transfer learning across domains |

### Dataset Variants

| Variant | Tokens | Samples |
|---------|--------|---------|
| 32k | ~26k | 589 |
| 128k | ~145k | 2,730 |
| 1M | ~152k | 2,670 |

### Scoring: Multiple-Choice Accuracy

- Format: 4 options (a, b, c, d) with human-annotated correct answer
- Metric: Exact match accuracy

---

## 3. Key Differences Between Benchmarks

| Aspect | LongMemEval | PersonaMem |
|--------|-------------|------------|
| **Focus** | Memory recall & reasoning | Personalization quality |
| **Format** | Open-ended QA | Multiple-choice |
| **Scoring** | LLM judge (binary) | Exact match |
| **Temporal** | Strong focus (dates, durations) | Preference evolution |
| **Aggregation** | Multi-session counting | N/A |
| **Abstention** | Explicit testing | N/A |

---

## 4. Oracle Dataset

**Location**: `evals/data/longmemeval_oracle.json`

**Purpose**: Contains ONLY the evidence sessions (not distractors). Used to:
1. Diagnose: Is failure due to retrieval or reading?
2. Ceiling: What's the best possible accuracy given perfect retrieval?
3. A/B test: Compare reading/generation improvements

**Usage**: Run eval with Oracle data to measure "reading accuracy" in isolation.

---

## 5. Current Persona Eval Status

**Already Implemented** (in benchmark_runner.py):
- Binary scoring via LongMemEval prompts
- Task-specific judge prompts
- Parallel question processing

**Gaps**:
- Abstention questions not explicitly handled (no `_abs` suffix detection)
- Only 3 question types used (SS, MS, TR) - missing KU, ABS, preferences
- PersonaMem not integrated
- Single run (should do 3 passes for mean±std)

---

## 6. Sources

- LongMemEval Paper: https://arxiv.org/abs/2410.10813
- LongMemEval GitHub: https://github.com/xiaowu0162/LongMemEval
- LongMemEval Project: https://xiaowu0162.github.io/long-mem-eval/
- PersonaMem Paper: https://arxiv.org/abs/2504.14225
- PersonaMem GitHub: https://github.com/bowen-upenn/PersonaMem
- PersonaMem Dataset: https://huggingface.co/datasets/bowen-upenn/PersonaMem
- Dria mem-agent: https://huggingface.co/driaforall/mem-agent
