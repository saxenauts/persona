# Persona Memory System - Evaluation Framework Design

**Version**: 1.0
**Date**: December 21, 2024
**Status**: Design Phase
**Owner**: Persona Team

---

## Implementation Status

### TODO Checklist

- [ ] **Phase 1: Data Acquisition**
  - [ ] Download PersonaMem 32k dataset from HuggingFace
  - [ ] Verify LongMemEval Oracle dataset integrity
  - [ ] Create PersonaMem data loader
  - [ ] Create LongMemEval data loader
  - [ ] Create unified loader interface

- [ ] **Phase 2: Sampling & Golden Set**
  - [ ] Implement stratified sampling algorithm
  - [ ] Generate LongMemEval golden set (250 questions)
  - [ ] Generate PersonaMem golden set (120 questions)
  - [ ] Save sample manifests with reproducibility info

- [ ] **Phase 3: Deep Logging**
  - [ ] Create logging schema (Pydantic models)
  - [ ] Implement deep logger utility
  - [ ] Enhance Persona adapter with logging hooks
  - [ ] Test logging with sample questions

- [ ] **Phase 4: Enhanced Runner**
  - [ ] Create CLI interface
  - [ ] Create config parser
  - [ ] Refactor runner for modularity
  - [ ] Add PersonaMem scoring
  - [ ] Test with Persona adapter
  - [ ] Test with Graphiti (Zep) adapter

- [ ] **Phase 5: Analysis & Reporting**
  - [ ] Create analysis engine
  - [ ] Implement failure pattern detection
  - [ ] Create retrieval quality analyzer
  - [ ] Generate result interpretation documentation
  - [ ] Test full pipeline

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Current State Analysis](#current-state-analysis)
3. [Benchmark Research](#benchmark-research)
4. [Proposed Solution](#proposed-solution)
5. [Golden Set Design](#golden-set-design)
6. [Sampling Strategy](#sampling-strategy)
7. [Evaluation Methodology](#evaluation-methodology)
8. [Implementation Plan](#implementation-plan)
9. [Success Metrics](#success-metrics)
10. [References](#references)

---

## 1. Problem Statement

### Current Evaluation Gaps

Our existing evaluation setup has several critical limitations:

**Limited Coverage**:
- Only testing 3 out of 7 question types from LongMemEval
- ~120 questions total (40 each of single-session, multi-session, temporal)
- No testing of knowledge updates, abstention, or preference inference

**Incomplete Scoring**:
- Binary scoring implemented correctly for LongMemEval
- No evaluation of personalization quality (preferences, identity tracking)
- Missing abstention detection (when to say "I don't know")

**Insufficient Analysis**:
- Limited visibility into what's being indexed (nodes, relationships)
- Basic retrieval logging but not queryable
- No breakdown of failure modes (retrieval vs reasoning vs generation)

**Weak Areas Identified**:
| Question Type | Current Accuracy | Sample Size |
|---------------|-----------------|-------------|
| Multi-session reasoning | 33% | 40 |
| Temporal reasoning | 30% | 40 |
| Single-session user | 58% | 40 |

### Business Impact

Without comprehensive evaluation:
- Cannot identify root causes of failures (retrieval? reasoning? generation?)
- Cannot validate improvements to memory architecture
- Cannot compare against baselines with statistical confidence
- Cannot measure personalization quality

---

## 2. Current State Analysis

### Existing Infrastructure

**Strengths**:
- ✅ Binary scoring (LongMemEval-compliant)
- ✅ Parallel execution with checkpointing
- ✅ Adapter pattern for multiple memory systems
- ✅ Task-specific evaluation prompts
- ✅ Basic retrieval logging

**Architecture**:
```
evals/
├── benchmark_runner.py       # Main orchestrator (340 lines)
├── adapters/
│   ├── base.py              # Abstract interface
│   ├── persona_adapter.py   # Our system (HTTP-based)
│   ├── mem0_adapter.py      # Baseline comparison
│   └── zep_adapter.py       # Graphiti comparison
├── longmemeval/
│   └── evaluate_qa.py       # Binary evaluation with 5 prompts
├── data/longmemeval/
│   ├── longmemeval_oracle.json           # 500 questions (full)
│   └── sampled_benchmark_data.json       # 80 questions (subset)
└── results/
    ├── benchmark_checkpoint.jsonl        # Incremental saves
    └── retrieval_logs.jsonl              # Query logging
```

### Data Scientist Handoff Points

**What's Working**:
1. Checkpoint-based resumption (can restart failed runs)
2. Thread-safe parallel execution (5 workers)
3. Retry logic with exponential backoff
4. Multiple adapter comparison in single run

**What Needs Enhancement**:
1. Stratified sampling (currently ad-hoc)
2. Deep logging (capture intermediate steps)
3. Unified data loader (supports multiple benchmarks)
4. Analysis tooling (failure pattern detection)

---

## 3. Benchmark Research

### LongMemEval (ICLR 2025)

**Paper**: https://arxiv.org/abs/2410.10813
**GitHub**: https://github.com/xiaowu0162/LongMemEval

**Dataset**: 500 questions across 7 types testing 5 core memory abilities

| Question Type | Count | Ability Tested | Description |
|---------------|-------|----------------|-------------|
| `single-session-user` | 70 | Information Extraction | Recall user-stated facts from one session |
| `single-session-assistant` | 56 | Information Extraction | Recall assistant-stated facts |
| `single-session-preference` | 30 | Information Extraction | Infer implicit preferences |
| `multi-session` | 133 | Multi-Session Reasoning | Aggregate across sessions (e.g., "How many total?") |
| `temporal-reasoning` | 133 | Temporal Reasoning | Date calculations, ordering, duration |
| `knowledge-update` | 78 | Knowledge Updates | Track changed information over time |
| Abstention (`*_abs`) | ~30 | Abstention | Correctly refuse when info absent |

**Scoring**: Binary (CORRECT/WRONG) via LLM judge (GPT-4o)
- Task-specific prompts (5 different templates)
- Off-by-one tolerance for temporal reasoning
- "Generous grading" - partial matches count

**Variants**:
- **LongMemEval_S**: ~115k tokens, ~40 sessions (standard)
- **LongMemEval_M**: ~1.5M tokens, ~500 sessions (stress test)
- **LongMemEval_Oracle**: Evidence sessions only (tests reading ability in isolation)

**Key Finding**: Commercial assistants show 30% accuracy drop on sustained interactions

---

### PersonaMem (COLM 2025)

**Paper**: https://arxiv.org/abs/2504.14225
**Dataset**: https://huggingface.co/datasets/bowen-upenn/PersonaMem

**Dataset**: Multi-variant benchmark testing personalization quality

| Variant | Token Length | Samples | Question Types |
|---------|--------------|---------|----------------|
| 32k | ~26k | 589 | 7 personalization skills |
| 128k | ~145k | 2,730 | 7 personalization skills |
| 1M | ~152k | 2,670 | 7 personalization skills |

**Question Types** (7 personalization skills):

| Type | What It Tests |
|------|---------------|
| `recall_user_shared_facts` | Recall static events/interests |
| `suggest_new_ideas` | Recommend novel items NOT in history |
| `acknowledge_latest_user_preferences` | Recognize most recent preference |
| `track_full_preference_evolution` | Track how preferences shift over time |
| `recalling_the_reasons_behind_previous_updates` | Recall why preferences changed |
| `provide_preference_aligned_recommendations` | Proactive personalized suggestions |
| `generalizing_to_new_scenarios` | Transfer learning across domains |

**Scoring**: Multiple-choice accuracy (a/b/c/d exact match)

**Key Finding**: Frontier models (GPT-4.1, o1, Gemini-2.0) score ~43% overall
- 60-70% on fact recall and tracking
- 30-50% on applying preferences to new scenarios

---

### Why Two Benchmarks?

They test **different capabilities**:

| Aspect | LongMemEval | PersonaMem |
|--------|-------------|------------|
| **Focus** | Memory recall & aggregation | Personalization quality |
| **Format** | Open-ended QA | Multiple-choice |
| **Temporal** | Strong (dates, durations) | Preference evolution |
| **Cross-session** | Counting, comparison | Identity tracking |
| **Abstention** | Explicit testing | N/A |

**Recommendation**: Run both separately, report both scores

---

## 4. Proposed Solution

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Unified Eval CLI                        │
│  python -m evals.run --benchmark [longmemeval|personamem]│
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │    Unified Data Loader          │
        │  - Stratified sampling          │
        │  - Format normalization         │
        │  - Reproducible random seeds    │
        └─────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │    Enhanced Benchmark Runner    │
        │  - Deep logging (per-step)      │
        │  - Multi-system comparison      │
        │  - Checkpoint resumption        │
        └─────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │    Dual Scoring System          │
        │  - LongMemEval: Binary judge    │
        │  - PersonaMem: Exact match      │
        └─────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │    Analysis & Reporting         │
        │  - Failure pattern detection    │
        │  - Retrieval/generation split   │
        │  - Comparative analysis         │
        └─────────────────────────────────┘
```

### Core Principles

1. **Keep It Simple**: No over-engineering, follow data science fundamentals
2. **Reproducible**: Random seeds, config persistence, deterministic sampling
3. **Queryable**: All logs in structured format (JSONL)
4. **Extensible**: Easy to add new benchmarks/adapters
5. **Out-of-Box**: CLI-driven, minimal setup

---

## 5. Golden Set Design

### Stratified Sampling Approach

**LongMemEval Golden Set** (250 questions):

| Type | Sample Size | Rationale |
|------|-------------|-----------|
| `single-session-user` | 35 | Basic recall (baseline capability) |
| `multi-session` | 60 | **Weak point** (33% accuracy) - need more signal |
| `temporal-reasoning` | 60 | **Weak point** (30% accuracy) - need more signal |
| `knowledge-update` | 40 | Not currently tested - critical for memory |
| `abstention` | 30 | Not currently tested - safety critical |
| `single-session-preference` | 25 | Currently 0% - diagnostic value |
| **Total** | **250** | Balanced coverage with weak-area focus |

**PersonaMem Golden Set** (120 questions, 32k variant):

| Type | Sample Size | Rationale |
|------|-------------|-----------|
| `recall_user_shared_facts` | 30 | Maps to LongMemEval single-session |
| `track_full_preference_evolution` | 30 | Unique to PersonaMem, critical for identity |
| `acknowledge_latest_user_preferences` | 20 | Recency detection |
| `generalizing_to_new_scenarios` | 20 | Transfer learning capability |
| `provide_preference_aligned_recommendations` | 20 | Proactive personalization |
| **Total** | **120** | Focus on preference/identity tracking |

### Sample Size Justification

**Statistical Power**:
- Per-stratum minimum: 25-30 samples (industry standard for CI estimation)
- Weak areas: 60 samples (higher signal-to-noise for diagnosis)
- Total: 370 questions (vs current 120)

**Research Backing**:
- SubLIME paper (arXiv:2406.15527): 10% sampling achieves 0.85-0.95 correlation with full dataset
- Medical research standards: n≥30 per group for statistical power
- LLM eval research: Small samples (n<30) suffer from high variance

---

## 6. Sampling Strategy

### Stratified Random Sampling

**Algorithm**:
```python
def create_golden_set(
    dataset: List[Question],
    sample_sizes: Dict[str, int],
    random_seed: int = 42
) -> List[Question]:
    """
    Stratified sampling with reproducibility

    Args:
        dataset: Full question list
        sample_sizes: {"type_name": count, ...}
        random_seed: For reproducibility

    Returns:
        Sampled question list
    """
    np.random.seed(random_seed)
    sampled = []

    for qtype, n_samples in sample_sizes.items():
        # Get all questions of this type
        questions_of_type = [q for q in dataset if q['question_type'] == qtype]

        # Random sample (without replacement)
        sampled_questions = np.random.choice(
            questions_of_type,
            size=min(n_samples, len(questions_of_type)),
            replace=False
        )
        sampled.extend(sampled_questions)

    return sampled
```

**Reproducibility Guarantees**:
1. Fixed random seed (default: 42)
2. Sample manifest saved with results (question IDs + types)
3. Config file persisted (sample_sizes, seed, dataset version)

### Handling Dataset Variants

**LongMemEval**:
- Source: `longmemeval_oracle.json` (500 questions)
- Filter by `question_type` field
- Handle abstention via `_abs` suffix in `question_id`

**PersonaMem**:
- Source: HuggingFace `bowen-upenn/PersonaMem`
- Use 32k variant (manageable context size)
- Filter by `question_type` column in CSV
- Preserve `correct_answer` (a/b/c/d) and `all_options` JSON

---

## 7. Evaluation Methodology

### LongMemEval Scoring

**Existing Implementation** (keep as-is):
```python
def evaluate_answer(question, gold, hypothesis, task_type) -> dict:
    # Generate task-specific prompt
    prompt = get_anscheck_prompt(task_type, question, gold, hypothesis)

    # LLM judge (GPT-4o)
    response = judge_llm.invoke(prompt)

    # Binary result
    correct = 'yes' in response.lower()

    return {
        "correct": correct,
        "raw_response": response
    }
```

**Task-Specific Prompts** (5 templates):
1. Standard: "answer yes if response contains/equivalent to correct answer"
2. Temporal: Same + "don't penalize off-by-one errors"
3. Knowledge-update: "previous info + updated answer OK if update correct"
4. Preference: "correct if recalls and utilizes personal info correctly"
5. Abstention: "answer yes if correctly identifies as unanswerable"

### PersonaMem Scoring

**New Implementation** (exact match):
```python
def evaluate_personamem(question, user_answer, correct_answer) -> dict:
    # Normalize both answers
    user_normalized = user_answer.strip().lower()
    correct_normalized = correct_answer.strip().lower()

    # Exact match
    correct = (user_normalized == correct_normalized)

    return {
        "correct": correct,
        "expected": correct_answer,
        "received": user_answer
    }
```

**Adapter Integration**:
- PersonaMem adapter converts multiple-choice to prompt
- Extracts single-letter answer (a/b/c/d) from model response
- Compares to `correct_answer` field

### Dual Reporting

**Results Schema**:
```json
{
  "run_id": "20241221_143052",
  "config": {
    "benchmarks": ["longmemeval", "personamem"],
    "sample_sizes": {...},
    "random_seed": 42
  },
  "longmemeval": {
    "overall_accuracy": 0.52,
    "task_accuracies": {
      "single-session-user": {"accuracy": 0.63, "count": 35},
      "multi-session": {"accuracy": 0.38, "count": 60},
      ...
    }
  },
  "personamem": {
    "overall_accuracy": 0.47,
    "task_accuracies": {
      "recall_user_shared_facts": {"accuracy": 0.67, "count": 30},
      ...
    }
  }
}
```

---

## 8. Implementation Plan

### Phase 1: Data Acquisition & Preparation

**Tasks**:
1. Download PersonaMem 32k variant from HuggingFace
2. Verify LongMemEval Oracle dataset integrity
3. Create unified schema mapper

**Deliverables**:
- `evals/data/personamem_32k.csv` (589 questions)
- `evals/data/personamem_32k_contexts.jsonl` (conversation histories)
- `evals/data/dataset_manifest.json` (metadata)

**Files to Create**:
- `evals/loaders/personamem_loader.py`
- `evals/loaders/unified_loader.py`

---

### Phase 2: Enhanced Logging Infrastructure

**Deep Logging Schema** (JSONL per question):

```json
{
  "question_id": "gpt4_abc123",
  "timestamp": "2024-12-21T14:30:52Z",
  "user_id": "Persona_q15_a8f3",

  "ingestion": {
    "duration_ms": 15420,
    "sessions_count": 47,
    "memories_created": {
      "episodes": 52,
      "psyche": 18,
      "goals": 7
    },
    "nodes_created": 77,
    "relationships_created": 134,
    "embeddings_generated": 77,
    "errors": []
  },

  "retrieval": {
    "query": "How many weeks between...",
    "duration_ms": 1847,
    "vector_search": {
      "top_k": 5,
      "seeds": [
        {"node": "episode_42", "score": 0.94, "type": "episode"},
        {"node": "episode_38", "score": 0.87, "type": "episode"},
        ...
      ]
    },
    "graph_traversal": {
      "max_hops": 2,
      "nodes_visited": 23,
      "relationships_traversed": 45,
      "final_ranked_nodes": ["episode_42", "episode_38", ...]
    },
    "context_size_tokens": 3452
  },

  "generation": {
    "duration_ms": 2310,
    "model": "gpt-4.1-mini",
    "temperature": 0.7,
    "prompt_tokens": 3580,
    "completion_tokens": 87,
    "answer": "The generated answer text..."
  },

  "evaluation": {
    "gold_answer": "1 week",
    "correct": true,
    "judge_response": "yes",
    "judge_model": "gpt-4o"
  }
}
```

**Files to Modify**:
- `evals/adapters/persona_adapter.py` (add logging hooks)
- `persona/core/rag_interface.py` (enhance retrieval logging)
- `persona/services/ingestion_service.py` (log ingestion stats)

**Files to Create**:
- `evals/logging/deep_logger.py` (centralized logging utility)
- `evals/logging/log_schema.py` (Pydantic models for validation)

---

### Phase 3: Unified Benchmark Runner

**Enhanced CLI**:
```bash
# LongMemEval run
python -m evals.run \
  --benchmark longmemeval \
  --types multi-session,temporal-reasoning \
  --samples 60 \
  --seed 42 \
  --adapters persona,mem0

# PersonaMem run
python -m evals.run \
  --benchmark personamem \
  --variant 32k \
  --samples 100 \
  --seed 42

# Combined run
python -m evals.run \
  --benchmark longmemeval,personamem \
  --config configs/full_eval.yaml
```

**Config File Format** (`configs/full_eval.yaml`):
```yaml
longmemeval:
  source: data/longmemeval/longmemeval_oracle.json
  sample_sizes:
    single-session-user: 35
    multi-session: 60
    temporal-reasoning: 60
    knowledge-update: 40
    abstention: 30
    single-session-preference: 25

personamem:
  source: huggingface://bowen-upenn/PersonaMem
  variant: 32k
  sample_sizes:
    recall_user_shared_facts: 30
    track_full_preference_evolution: 30
    acknowledge_latest_user_preferences: 20
    generalizing_to_new_scenarios: 20
    provide_preference_aligned_recommendations: 20

global:
  random_seed: 42
  adapters: [persona, mem0]
  parallel_workers: 5
  checkpoint_dir: evals/results
  deep_logging: true
```

**Files to Create**:
- `evals/cli.py` (CLI entry point)
- `evals/runner.py` (refactored from benchmark_runner.py)
- `evals/config.py` (config parser)

**Files to Modify**:
- `evals/benchmark_runner.py` (refactor into modular runner.py)

---

### Phase 4: Analysis Tooling

**Analysis CLI**:
```bash
# Summary report
python -m evals.analyze \
  --run results/run_20241221_143052.json \
  --summary

# Deep dive by question type
python -m evals.analyze \
  --run results/run_20241221_143052.json \
  --type multi-session \
  --failures

# Retrieval analysis
python -m evals.analyze \
  --run results/run_20241221_143052.json \
  --retrieval \
  --min-score 0.7

# Comparative analysis
python -m evals.analyze \
  --runs results/run_a.json results/run_b.json \
  --compare
```

**Analysis Capabilities**:

1. **Failure Pattern Detection**:
   - Group failures by error type (retrieval miss, wrong reasoning, hallucination)
   - Extract common patterns in failed questions
   - Identify systematic biases

2. **Retrieval Quality Analysis**:
   - Did top-k seeds contain answer? (recall metric)
   - Average similarity score for successful vs failed questions
   - Graph traversal depth effectiveness

3. **Timing Analysis**:
   - Breakdown: ingestion / retrieval / generation
   - Identify bottlenecks
   - Compare across question types

4. **Comparative Analysis**:
   - System A vs System B accuracy by type
   - Win/loss matrix
   - Statistical significance tests

**Files to Create**:
- `evals/analysis/analyzer.py` (main analysis engine)
- `evals/analysis/patterns.py` (failure pattern detection)
- `evals/analysis/retrieval_stats.py` (retrieval-specific analysis)
- `evals/analysis/visualizations.py` (optional: charts/plots)

---

### Phase 5: PersonaMem Adapter

**New Adapter Implementation**:

```python
class PersonaMemAdapter(MemorySystem):
    """Adapter for PersonaMem multiple-choice evaluation"""

    def query(self, user_id: str, query: str, options: List[str]) -> str:
        """
        Query with multiple-choice options

        Args:
            user_id: User identifier
            query: Question text
            options: ["Option A text", "Option B text", ...]

        Returns:
            Single letter answer: "a", "b", "c", or "d"
        """
        # Get context from memory system
        context = await self.rag_interface.get_context(query)

        # Format prompt with options
        prompt = self._format_mc_prompt(query, options, context)

        # Generate answer
        response = await self.llm.chat(prompt)

        # Extract letter (a/b/c/d)
        answer_letter = self._extract_letter(response.content)

        return answer_letter

    def _format_mc_prompt(self, question, options, context):
        return f"""
        Given the following context about a user:

        {context}

        Question: {question}

        Options:
        (a) {options[0]}
        (b) {options[1]}
        (c) {options[2]}
        (d) {options[3]}

        Please select the best answer (a, b, c, or d):
        """

    def _extract_letter(self, response: str) -> str:
        """Extract a/b/c/d from model response"""
        # Look for standalone letter
        match = re.search(r'\b([a-d])\b', response.lower())
        if match:
            return match.group(1)

        # Look for pattern like "(a)" or "a)"
        match = re.search(r'[(]?([a-d])[)]', response.lower())
        if match:
            return match.group(1)

        # Default: return first letter found
        for char in response.lower():
            if char in 'abcd':
                return char

        # Fallback
        return 'a'
```

**Files to Create**:
- `evals/adapters/personamem_adapter.py`

---

## 9. Success Metrics

### Quantitative Goals

**Coverage**:
- ✅ 7/7 LongMemEval question types tested (vs 3/7 currently)
- ✅ 7 PersonaMem personalization skills tested (vs 0 currently)
- ✅ 370 total questions (vs 120 currently)

**Accuracy Targets** (after system improvements):
| Question Type | Current | Target |
|---------------|---------|--------|
| Multi-session | 33% | 55% |
| Temporal-reasoning | 30% | 55% |
| Knowledge-update | N/A | 50% |
| Abstention | N/A | 70% |
| PersonaMem overall | N/A | 45% |

**Observability**:
- ✅ Deep logs for 100% of questions
- ✅ Retrieval recall metrics (seed node quality)
- ✅ Timing breakdowns (ingestion/retrieval/generation)
- ✅ Failure pattern detection

### Qualitative Goals

**Usability**:
- Data scientist can run full eval in single CLI command
- Results are human-readable (markdown reports)
- Easy to compare multiple systems
- Reproducible (fixed seeds, config persistence)

**Debugging**:
- Can identify if failure is retrieval or generation
- Can inspect what was indexed for any question
- Can trace retrieval path (vector → graph → context)

---

## 10. References

### Papers

1. **LongMemEval**: Benchmarking Chat Assistants on Long-Term Interactive Memory
   - arXiv: https://arxiv.org/abs/2410.10813
   - GitHub: https://github.com/xiaowu0162/LongMemEval
   - ICLR 2025

2. **PersonaMem**: Know Me, Respond to Me
   - arXiv: https://arxiv.org/abs/2504.14225
   - GitHub: https://github.com/bowen-upenn/PersonaMem
   - Dataset: https://huggingface.co/datasets/bowen-upenn/PersonaMem
   - COLM 2025

3. **SubLIME**: Less Is More for Evaluation
   - arXiv: https://arxiv.org/abs/2406.15527
   - On data-efficient LLM evaluation via adaptive sampling

### Internal Documents

- `docs/eval_research_notes.md` - Detailed research findings
- `evals/README.md` - Current evaluation setup
- `evals/analysis/scoring_methodology.md` - Scoring analysis

### External Resources

- LongMemEval Project Page: https://xiaowu0162.github.io/long-mem-eval/
- Dria mem-agent: https://huggingface.co/driaforall/mem-agent

---

## Appendix A: File Structure

```
evals/
├── cli.py                          # [NEW] CLI entry point
├── runner.py                       # [NEW] Refactored runner
├── config.py                       # [NEW] Config parser
├── benchmark_runner.py             # [MODIFY] Legacy (keep for compatibility)
│
├── loaders/                        # [NEW]
│   ├── __init__.py
│   ├── longmemeval_loader.py      # LongMemEval data loader
│   ├── personamem_loader.py       # PersonaMem data loader
│   └── unified_loader.py          # Unified interface
│
├── logging/                        # [NEW]
│   ├── __init__.py
│   ├── deep_logger.py             # Centralized logging
│   └── log_schema.py              # Pydantic schemas
│
├── adapters/
│   ├── base.py                    # [KEEP] Abstract base
│   ├── persona_adapter.py         # [MODIFY] Add logging
│   ├── mem0_adapter.py            # [KEEP]
│   ├── zep_adapter.py             # [KEEP]
│   └── personamem_adapter.py      # [NEW] Multiple-choice support
│
├── analysis/                       # [NEW]
│   ├── __init__.py
│   ├── analyzer.py                # Main analysis engine
│   ├── patterns.py                # Failure pattern detection
│   ├── retrieval_stats.py         # Retrieval quality metrics
│   └── visualizations.py          # Charts (optional)
│
├── configs/                        # [NEW]
│   ├── full_eval.yaml             # Complete eval config
│   ├── quick_test.yaml            # Small test config
│   └── longmemeval_only.yaml      # LongMemEval-only config
│
├── data/
│   ├── longmemeval/
│   │   └── longmemeval_oracle.json       # [EXISTS]
│   └── personamem/                        # [NEW]
│       ├── questions_32k.csv
│       └── shared_contexts_32k.jsonl
│
└── results/
    ├── run_20241221_143052/              # [NEW] Per-run directory
    │   ├── config.yaml                   # Run config
    │   ├── results.json                  # Aggregated results
    │   ├── deep_logs.jsonl               # Deep logs
    │   └── sample_manifest.json          # Question IDs used
    └── retrieval_logs.jsonl              # [EXISTS] Legacy logs
```

---

**End of Document**
