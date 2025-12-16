# Memory Systems Evaluation

This directory contains benchmark results comparing different memory systems on the LongMemEval dataset.

## Systems Tested

| System | Description |
|--------|-------------|
| **Persona** | Our personal memory system with date-aware indexing and graph-based retrieval |
| **Mem0 (Vector)** | Mem0 open-source with Qdrant vector storage |
| **Mem0 (Graph)** | Mem0 with Neo4j graph memory enabled |

---

## Grading System

Each answer is graded by an LLM Judge (GPT-4.1-mini) on a **1-5 scale**:

| Grade | Meaning | Description |
|-------|---------|-------------|
| **5** | Perfect | Answer is completely correct and complete |
| **4** | Good | Answer is mostly correct with minor issues |
| **3** | Partial | Answer is partially correct but missing key details |
| **2** | Poor | Answer is mostly incorrect or incomplete |
| **1** | Wrong | Answer is completely incorrect or irrelevant |

**"Correct" = Grade ≥ 4**

---

## Benchmark 1: Sampled Comparison (80 Questions)

**Date**: December 15, 2024  
**Dataset**: LongMemEval (40 multi-session + 40 temporal-reasoning)

### Overall Results

| System | Avg Grade | Correct Rate (≥4) |
|--------|-----------|-------------------|
| **Mem0 Vector** | **3.35** | **51.2%** |
| **Persona** | 3.33 | 50.0% |
| Mem0 Graph | 3.25 | 48.8% |

### By Question Type

| Type | Persona | Mem0 Vector | Mem0 Graph | Best |
|------|---------|-------------|------------|------|
| Multi-session | 3.08 | 3.15 | **3.27** | Mem0 Graph |
| Temporal-reasoning | **3.58** | 3.55 | 3.23 | Persona |

### Result Files

- `results/benchmark_persona_sampled_valid.json` - Persona (80 questions)
- `results/benchmark_mem0_vector_final.json` - Mem0 Vector (80 questions)
- `results/benchmark_run_20251215_195307.json` - Mem0 Graph (80 questions)

---

## Benchmark 2: Single-Session-User (70 Questions)

**Date**: December 15, 2024  
**Dataset**: LongMemEval — single-session-user questions only

### Results (Updated - Fixed Embeddings)

| System | Avg Grade | Correct Rate (≥4) |
|--------|-----------|-------------------|
| **Mem0 (Vector)** | **3.84** | **65.2%** |
| Persona | 3.70 | 64.9% |

### Key Finding

**Persona and Mem0 perform nearly identically on single-session-user questions** (within 0.3%).

> **Note**: Initial Mem0 run (10.3%) was broken due to embedding configuration. Re-run shows 65.2%.

### Result Files

- `results/benchmark_run_20251215_211702.json` - Mem0 single-session (CORRECTED)
- `results/benchmark_run_20251215_074839.json` - Old broken run (DELETE)


---

## Benchmark 3: Full Persona Evaluation (498 Questions)

**Date**: July 2024  
**Dataset**: Complete LongMemEval (all 6 question types)

### Results by Question Type

| Question Type | Correct | Total | Accuracy |
|---------------|---------|-------|----------|
| single-session-user | 41 | 70 | **58.6%** |
| knowledge-update | 43 | 78 | **55.1%** |
| multi-session | 35 | 132 | 26.5% |
| temporal-reasoning | 35 | 133 | 26.3% |
| single-session-assistant | 10 | 55 | 18.2% |
| single-session-preference | 0 | 30 | 0.0% |
| **OVERALL** | **164** | **498** | **32.9%** |

### Key Findings

1. **Best at user-stated info** — single-session-user (58.6%)
2. **Good at knowledge updates** — knowledge-update (55.1%)
3. **Struggling with preferences** — single-session-preference (0%)

### Result Files

- `results/per_question_diagnostics.json` - Full benchmark (498 questions)

---

## Summary

| Benchmark | Persona | Mem0 Vector | Mem0 Graph | Winner |
|-----------|---------|-------------|------------|--------|
| Sampled (80Q) | 50.0% | **51.2%** | 48.8% | Mem0 Vector |
| Single-Session (78Q) | **64.9%** | 10.3% | — | Persona |
| Multi-session | 42.5% | 45.0% | **47.5%** | Mem0 Graph |
| Temporal-reasoning | **57.5%** | 57.5% | 50.0% | Persona |

### Recommendations

- **Use Persona** for: Temporal reasoning, single-session recall, consistent performance
- **Use Mem0 Vector** for: Fast, simple memory retrieval
- **Use Mem0 Graph** for: Multi-entity relationships across sessions

---

## Running Benchmarks

```bash
# Run benchmark
python -m evals.benchmark_runner

# Analyze results
python evals/analyze_results.py
```

## Configuration

All benchmarks use:
- **Model**: gpt-4.1-mini (Azure OpenAI)
- **Embeddings**: text-embedding-3-small (Azure OpenAI)
- **Vector Store**: Qdrant (local Docker)
- **Graph Store**: Neo4j (for Mem0 Graph)