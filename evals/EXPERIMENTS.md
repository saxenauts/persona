# Experiment Handover Log

**Date**: 2025-12-16
**Status**: Zep Graphiti Benchmarking Complete

## 1. Overview
This document tracks the LongMemEval benchmarking experiments conducted for **Persona** (Baseline), **Mem0** (Vector/Graph), and **Zep** (Graphiti).

### Key Findings
| System | Single Session (Recall) | Multi-Session (Recall) | Temporal Reasoning | Speed (Query) |
|---|---|---|---|---|
| **Persona** | 42.9% | 40.0% | 60.0% | ~2.65s |
| **Mem0** | **55.0%** | 42.5% | 60.0% | **0.70s** |
| **Zep** | 45.0% | **60.0%** | **85.0%** | ~1.05s |

> **Zep Strength**: Dominates in complex **Temporal Reasoning** (85%) and **Multi-Session** context bridging.
> **Zep Weakness**: Lower recall on **Single Session** simple queries due to strict "Fail-Safe" graph ingestion (refuses to hallucinate on zero-edge extraction).

---

## 2. Data & File Locations
**IMPORTANT**: Raw result files are in `evals/results/` (Gitignored). Do NOT delete them.

### A. Single Session Experiments
*Methodology*: Questions 0-24 of `single_session_user_benchmark.json` (Sequential).
*Status*: **Valid Comparison on Intersection (n=24)**.

| System | Source File (`evals/results/`) | Questions (n) | Notes |
|---|---|---|---|
| **Zep** | `benchmark_checkpoint.jsonl` | 24 | Contains `Zep (Graphiti)_ans` |
| **Persona**| `benchmark_checkpoint.jsonl` | 24 | Contains `Persona_ans` (Co-run with Zep) |
| **Mem0** | `benchmark_run_20251215_211702.json` | 70 | Superset. Use intersection logic to compare. |

### B. Multi-Session & Temporal Reason Experiments
*Methodology*: Sampled 80 Questions (40 Multi + 40 Temporal).
*Status*: **Exact Match (n=80)** across all systems.

| System | Source File (`evals/results/`) | Questions (n) | Notes |
|---|---|---|---|
| **Zep** | `benchmark_run_20251216_113346.json` | 80 | Full run. |
| **Persona**| `benchmark_persona_sampled_valid.json`| 80 | Valid baseline. |
| **Mem0** | `benchmark_mem0_vector_final.json` | 80 | Vector implementation. |

---

## 3. Analysis Scripts
Use these scripts in `evals/scripts/` to reproduce stats:

1.  **`generate_breakdown.py`**:
    - Aggregates the disparate files above.
    - Prints the Success Rate tables.
    - *Usage*: `python evals/scripts/generate_breakdown.py`

2.  **`analyze_benchmark_deep.py`**:
    - Performs the "Hardcore" deep dive.
    - Calculates Latency stats.
    - Extracts specific Success/Failure examples (Safety vs Hallucination).
    - *Usage*: `python evals/scripts/analyze_benchmark_deep.py`

3.  **`verify_full_benchmark_integrity.py`**:
    - Verifies that the Question Sets overlap correctly.
    - *Usage*: `python evals/scripts/verify_full_benchmark_integrity.py`

---

## 4. Qualitative Notes
- **Zep Integration**: Found in `evals/adapters/zep_adapter.py`. Uses `graphiti-core` with Azure OpenAI. Note the `ResponseWrapper` monkey-patch to handle Azure API quirks.
- **Latency**: Zep query time is fast (~1s) because complex graph construction happens asynchronously during `add_sessions` (Ingestion), not at query time.
- **Safety**: Zep has a high "Refusal Rate" (Grade 5/Refusal) for empty contexts, making it safer than Persona (Grade 1/Hallucination) but lower recall.
