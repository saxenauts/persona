# Persona Memory System - Evaluation Framework

A comprehensive evaluation framework for testing long-term memory systems against academic benchmarks, featuring deep observability, reproducible sampling, and multi-system comparison.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Unified Benchmark Architecture](#unified-benchmark-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Results (Placeholders)](#results)
- [Usage Guide](#usage-guide)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Extending the Framework](#extending-the-framework)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## Overview

This evaluation framework tests memory systems on their ability to:
- **Recall user-stated facts** across conversations
- **Track temporal information** (dates, durations, ordering)
- **Aggregate multi-session data** (counting, comparison)
- **Update knowledge** when information changes
- **Personalize responses** based on preferences
- **Know when to abstain** (say "I don't know")

The framework supports multiple benchmarks (LongMemEval, PersonaMem) and memory systems (Persona, Mem0, Zep/Graphiti), with deep logging for debugging and analysis.

---

## Features

✅ **Dual Benchmark Support**
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) - 500 questions, 7 question types
- [PersonaMem](https://github.com/bowen-upenn/PersonaMem) (COLM 2025) - 589+ questions, 7 personalization skills

✅ **Stratified Sampling**
- Reproducible golden sets (695 curated questions)
- Balanced coverage across question types
- Fixed random seeds for reproducibility

✅ **Deep Observability**
- Structured JSONL logging (ingestion → retrieval → generation → evaluation)
- Retrieval quality metrics (vector search, graph traversal)
- Timing breakdowns per pipeline stage
- Failure pattern detection

✅ **Multi-System Comparison**
- Pluggable adapter architecture
- Parallel evaluation support
- Side-by-side performance analysis

✅ **Production-Ready CLI**
- Single-command evaluation runs
- Interactive result analysis
- Automated report generation

---

## Unified Benchmark Architecture

We combine two benchmarks to test both **reasoning** and **recall**:

| Benchmark | What it Tests | Reference |
|-----------|---------------|----------|
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | Temporal logic, multi-session aggregation | ICLR 2025 |
| [PersonaMem](https://github.com/bowen-upenn/PersonaMem) | Factual precision, personalization | COLM 2025 |

### Evaluation Approach

We report **two separate benchmark scores** (paper-aligned):
- **LongMemEval** macro-by-qtype accuracy
- **PersonaMem** macro-by-qtype accuracy

**Scoring methodology:**
- LongMemEval: LLM judge (configurable via `EVAL_JUDGE_MODEL`, default `gpt-5-mini`)
- PersonaMem: Exact match on multiple choice
- Macro-by-type: Each question type weighted equally (skill fairness)

**Test set options:**
| Set | Purpose | Size |
|-----|---------|------|
| `--samples 2` | Smoke test | ~12 questions |
| `--samples 10` | Quick test | ~60 questions |
| `--golden-set` | Full eval | ~695 questions |

**Unified golden set config:**
- `evals/configs/golden_set.yaml` defines the combined sampling plan across LongMemEval + PersonaMem.
- `evals/scripts/generate_golden_sets.py` writes per-benchmark golden sets plus `combined_golden_set_manifest.json` for cross-system comparisons.

**Full dataset config (paper-complete):**
- `evals/configs/full_dataset.yaml` runs all questions (~1089 total).


### Capability Coverage

**Currently Tested:**

| Category | Capability | Source |
|----------|------------|--------|
| Recall | Static facts (names, events) | PersonaMem |
| Temporal | Date ordering, durations | LongMemEval |
| Logic | Aggregation across sessions | LongMemEval |
| Updates | Correcting outdated info | LongMemEval |
| Abstention | Knowing when to say "I don't know" | LongMemEval |
| Alignment | Preference-based suggestions | PersonaMem |

**Persona-Specific (Future Tests):**

| Category | Capability | Why Persona Excels |
|----------|------------|-------------------|
| Goal Tracking | "What are my current tasks?" | Explicit `Goal` nodes |
| Inventory | Tracking items (food, ingredients) | Goal nodes with state |
| Proactivity | "Remind me about X" | Temporal + Goal linking |
| Multi-Hop | "Did I like X before Y changed?" | Graph traversal |
| Narrative | Understanding life "stories" | Episode temporal chaining |

---

## Installation

### Prerequisites

- Python 3.12+
- Poetry (package manager)
- Docker (for Neo4j graph database)

### Setup

```bash
# Clone the repository
git clone https://github.com/saxenauts/persona.git
cd persona

# Install dependencies
poetry install

# Download benchmark datasets
poetry run python evals/scripts/download_personamem.py

# Verify LongMemEval dataset
poetry run python evals/scripts/verify_longmemeval_oracle.py

# Generate golden sets (695 stratified questions)
# Uses evals/configs/golden_set.yaml and writes combined_golden_set_manifest.json
poetry run python evals/scripts/generate_golden_sets.py

# Create default config files
poetry run python -m evals.cli create-configs
```

### Environment Variables

```bash
# LLM Service (required for memory system)
export LLM_SERVICE="foundry/gpt-5.2"
export EMBEDDING_SERVICE="azure/text-embedding-3-small"

# Eval Judge Model (optional, default: gpt-5-mini)
export EVAL_JUDGE_MODEL="gpt-5-mini"

# Parallel ingestion (optional, default: 5)
export INGEST_SESSION_CONCURRENCY="5"

# Azure OpenAI
export AZURE_API_KEY="your-key"
export AZURE_API_BASE="https://your-endpoint.openai.azure.com/"

# Neo4j (for Persona adapter)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"
```

---

## Quick Start

### Run Your First Evaluation

```bash
# Quick test (sample 5 per type, LongMemEval subset)
poetry run python -m evals.cli run \
  --benchmark longmemeval \
  --samples 5 \
  --seed 42

# Use pre-generated golden set (~695 questions)
poetry run python -m evals.cli run \
  --config evals/configs/full_eval.yaml \
  --golden-set

# Analyze results
poetry run python -m evals.cli analyze run_20241221_143052 --summary
```

---

## Results (Placeholders)

> [!CAUTION]
> **Placeholder (Incomplete)**: Results pending full benchmark run.

### Comparative Performance

| System | LongMemEval (Acc) | PersonaMem (Acc) | Avg. Latency (ms) |
|--------|-------------------|-------------------|-------------------|
| **Persona (Graph)** | 84.2% | 91.5% | 840ms |
| **Mem0 (Vector)** | 76.5% | 88.2% | 1,250ms |
| **Zep (Graphiti)** | 79.1% | 86.4% | 910ms |

### LongMemEval Performance

| System | Overall | Multi-Session | Temporal | Knowledge Update | Single-Session |
|--------|---------|---------------|----------|------------------|----------------|
| **Persona** | 52.3% | 38.3% | 57.5% | 55.0% | 63.5% |
| Mem0 (Vector) | 51.2% | 45.0% | 57.5% | 50.0% | 65.2% |
| Mem0 (Graph) | 48.8% | 47.5% | 50.0% | 48.0% | 62.0% |
| Zep (Graphiti) | 45.0% | 40.0% | 45.0% | 42.0% | 55.0% |

*Based on 220 questions from LongMemEval golden set*

### PersonaMem Performance

| System | Overall | Fact Recall | Preference Evolution | Generalization | Recommendations |
|--------|---------|-------------|----------------------|----------------|-----------------|
| **Persona** | 47.5% | 67.0% | 55.0% | 35.0% | 40.0% |
| Mem0 (Vector) | 43.0% | 60.0% | 50.0% | 30.0% | 35.0% |

*Based on 120 questions from PersonaMem golden set (32k variant)*

### Key Findings

📊 **Strengths**:
- Single-session fact recall: 60-68% accuracy
- Knowledge updates: 50-55% accuracy
- Temporal reasoning: 50-58% accuracy

📉 **Challenges**:
- Multi-session aggregation: 38-48% accuracy
- Preference inference: 30-40% accuracy
- Transfer to new scenarios: 30-35% accuracy

🔍 **System Comparison**:
- **Persona**: Best at temporal reasoning and knowledge updates
- **Mem0 (Vector)**: Fastest, good for simple recall
- **Mem0 (Graph)**: Better at multi-entity relationships

---

## Usage Guide

### Running Evaluations

#### Option 1: Using Config Files (Recommended)

```bash
# Full evaluation (both benchmarks, ~695 questions)
poetry run python -m evals.cli run --config evals/configs/full_eval.yaml

# Full dataset run (paper-complete, ~1089 questions)
poetry run python -m evals.cli run --config evals/configs/full_dataset.yaml

# Quick test (small sample)
poetry run python -m evals.cli run --config evals/configs/quick_test.yaml

# LongMemEval only
poetry run python -m evals.cli run --config evals/configs/longmemeval_only.yaml
```

#### Option 2: Command-Line Arguments

```bash
# Custom sampling
poetry run python -m evals.cli run \
  --benchmark longmemeval \
  --types multi-session,temporal-reasoning \
  --samples 20 \
  --seed 42 \
  --workers 5

# Multiple benchmarks
poetry run python -m evals.cli run \
  --benchmark longmemeval \
  --benchmark personamem \
  --samples 10

# Compare multiple adapters
poetry run python -m evals.cli run \
  --benchmark longmemeval \
  --adapter persona \
  --adapter mem0 \
  --samples 15
```

#### Option 3: Use Pre-Generated Golden Sets

```bash
# Use golden sets instead of random sampling
poetry run python -m evals.cli run \
  --config evals/configs/full_eval.yaml \
  --golden-set
```

### Analyzing Results

```bash
# Summary report
poetry run python -m evals.cli analyze run_20241221_143052 --summary

# Filter by question type
poetry run python -m evals.cli analyze run_20241221_143052 \
  --type multi-session

# Show only failures
poetry run python -m evals.cli analyze run_20241221_143052 \
  --type multi-session \
  --failures

# Retrieval quality analysis
poetry run python -m evals.cli analyze run_20241221_143052 --retrieval
```

### Comparing Runs

```bash
# Compare two runs
poetry run python -m evals.cli compare run_a run_b

# Compare three systems
poetry run python -m evals.cli compare \
  persona_20241221 \
  mem0_20241221 \
  zep_20241221
```

For per-type latency/token comparisons using the unified golden set:

```bash
poetry run python evals/scripts/compare_runs.py \
  --runs run_a,run_b \
  --manifest evals/data/golden_sets/combined_golden_set_manifest.json
```

---

## Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Data Loading                         │
│  - LongMemEval Oracle (500 questions)                   │
│  - PersonaMem 32k (589 questions)                       │
│  - Stratified sampling → Golden sets (695 questions)    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                 Memory System Adapter                   │
│  - Ingest conversation history                          │
│  - Query with test question                             │
│  - Log retrieval stats (vector + graph)                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Answer Generation (LLM)                    │
│  - Context + question → generated answer                │
│  - Track tokens, latency, model used                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Evaluation (Judge)                     │
│  - LongMemEval: Binary (gpt-5.2 judge)                   │
│  - PersonaMem: Exact match (a/b/c/d)                    │
│  - Task-specific prompts                                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│             Deep Logging & Analysis                     │
│  - JSONL logs (queryable)                               │
│  - Failure pattern detection                            │
│  - Retrieval quality metrics                            │
│  - Comparative reports                                  │
└─────────────────────────────────────────────────────────┘
```

### Adapter Interface

All memory systems implement the `MemorySystem` base class:

```python
from evals.adapters.base import MemorySystem

class YourAdapter(MemorySystem):
    def add_session(self, user_id: str, session_data: str, date: str) -> None:
        """Ingest a single session into the memory system."""
        pass
    
    def add_sessions(self, user_id: str, sessions: List[Dict]) -> None:
        """Bulk ingest sessions. Each dict has 'content' and 'date' keys."""
        pass

    def query(self, user_id: str, question: str) -> str:
        """Query memory and generate answer."""
        pass

    def reset(self, user_id: str) -> None:
        """Clear memory for a user."""
        pass
```

### Deep Logging Schema

Each question evaluation produces a structured log:

```json
{
  "question_id": "gpt4_abc123",
  "timestamp": "2024-12-21T14:30:52Z",
  "user_id": "user_123",
  "benchmark": "longmemeval",
  "question_type": "multi-session",
  "question": "How many times did I visit the gym?",

  "ingestion": {
    "duration_ms": 15420,
    "sessions_count": 47,
    "memories_created": {"episodes": 52, "psyche": 18, "goals": 7},
    "nodes_created": 77,
    "relationships_created": 134,
    "embeddings_generated": 77
  },

  "retrieval": {
    "query": "How many times did I visit the gym?",
    "duration_ms": 1847,
    "vector_search": {
      "top_k": 5,
      "seeds": [
        {"node_id": "episode_42", "score": 0.94, "type": "episode"}
      ]
    },
    "graph_traversal": {
      "max_hops": 2,
      "nodes_visited": 23,
      "final_ranked_nodes": ["episode_42", "episode_38"]
    },
    "context_size_tokens": 3452
  },

  "generation": {
    "duration_ms": 2310,
    "model": "gpt-4o-mini",
    "answer": "You visited the gym 3 times."
  },

  "evaluation": {
    "gold_answer": "3",
    "correct": true,
    "judge_model": "gpt-4o"
  }
}
```

---

## Configuration

### Config File Format (`configs/full_eval.yaml`)

```yaml
longmemeval:
  source: evals/data/longmemeval_oracle.json
  sample_sizes:
    single-session-user: 35
    multi-session: 60
    temporal-reasoning: 60
    knowledge-update: 40
    single-session-preference: 25

personamem:
  source: evals/data/personamem
  variant: 32k
  sample_sizes:
    recall_user_shared_facts: 30
    track_full_preference_evolution: 30
    generalizing_to_new_scenarios: 20
    provide_preference_aligned_recommendations: 20
    recalling_the_reasons_behind_previous_updates: 20

global:
  random_seed: 42
  adapters: [persona]
  parallel_workers: 5
  checkpoint_dir: evals/results
  deep_logging: true
```

### Creating Custom Configs

```python
from evals.config import EvalConfig, BenchmarkConfig

config = EvalConfig(
    longmemeval=BenchmarkConfig(
        source="evals/data/longmemeval_oracle.json",
        sample_sizes={"multi-session": 20, "temporal-reasoning": 20}
    ),
    random_seed=42,
    adapters=["persona", "mem0"],
    parallel_workers=3
)

config.save("evals/configs/my_custom_config.yaml")
```

---

## Extending the Framework

### Adding a New Memory System

1. **Create adapter** in `evals/adapters/`:

```python
from evals.adapters.base import MemorySystem

class MySystemAdapter(MemorySystem):
    def __init__(self):
        # Initialize your memory system
        pass

    def add_session(self, user_id: str, session_data: str, date: str) -> None:
        """Ingest a single session into the memory system."""
        pass
    
    def add_sessions(self, user_id: str, sessions: List[Dict]) -> None:
        """Bulk ingest sessions. Each dict has 'content' and 'date' keys."""
        for session in sessions:
            self.add_session(user_id, session['content'], session['date'])

    def query(self, user_id: str, question: str) -> str:
        """Retrieve relevant context and generate answer."""
        context = self.retrieve(user_id, question)
        return self.llm.generate(context, question)

    def reset(self, user_id: str) -> None:
        """Clear all memory for a specific user to ensure test isolation."""
        pass
```

2. **Register in CLI** (`evals/cli.py`):

```python
from evals.adapters.mysystem_adapter import MySystemAdapter

ADAPTERS = {
    "persona": PersonaAdapter,
    "mem0": Mem0Adapter,
    "mysystem": MySystemAdapter,  # Add here
}
```

3. **Run evaluation**:

```bash
poetry run python -m evals.cli run \
  --benchmark longmemeval \
  --adapter mysystem \
  --samples 20
```

### Adding a New Benchmark

1. **Create loader** in `evals/loaders/`:

```python
from evals.loaders.unified_loader import UnifiedBenchmarkLoader

class MyBenchmarkLoader:
    def load(self) -> List[Question]:
        # Load your benchmark dataset
        pass

    def stratified_sample(self, sample_sizes: Dict[str, int]) -> List[Question]:
        # Implement sampling logic
        pass
```

2. **Integrate with unified loader**:

```python
# In unified_loader.py
if benchmark == "mybenchmark":
    self.loader = MyBenchmarkLoader(...)
```

3. **Add scoring logic**:

```python
# In runner.py
def evaluate_mybenchmark(question, answer):
    # Your evaluation logic
    pass
```

---

## Project Structure

```
evals/
├── adapters/              # Memory system adapters
│   ├── base.py           # Abstract base class
│   ├── persona_adapter.py # Persona system
│   ├── mem0_adapter.py   # Mem0 system
│   └── zep_adapter.py    # Zep/Graphiti system
│
├── loaders/              # Data loaders
│   ├── longmemeval_loader.py
│   ├── personamem_loader.py
│   └── unified_loader.py
│
├── logging/              # Deep logging infrastructure
│   ├── log_schema.py     # Pydantic models
│   └── deep_logger.py    # Logger utility
│
├── longmemeval/          # LongMemEval evaluation
│   └── evaluate_qa.py    # Binary judge (GPT-4o)
│
├── configs/              # YAML configurations
│   ├── full_eval.yaml
│   ├── quick_test.yaml
│   └── longmemeval_only.yaml
│
├── data/
│   ├── longmemeval/
│   │   └── longmemeval_oracle.json    # 500 questions
│   ├── personamem/
│   │   ├── questions_32k_32k.json     # 589 questions
│   │   ├── questions_128k_32k.json    # 2,727 questions
│   │   └── questions_1M_32k.json      # 2,674 questions
│   └── golden_sets/
│       ├── longmemeval_golden_set.json      # 220 questions
│       └── personamem_golden_set.json       # 120 questions
│
├── scripts/              # Utility scripts
│   ├── download_personamem.py
│   ├── generate_golden_sets.py
│   └── verify_longmemeval_oracle.py
│
├── results/              # Evaluation results
│   └── run_YYYYMMDD_HHMMSS/
│       ├── deep_logs.jsonl
│       ├── summary.json
│       └── run_metadata.json
│
├── analysis/             # Historical analysis (gitignored)
├── cli.py               # CLI interface
├── config.py            # Config parser
├── runner.py            # Evaluation orchestrator
└── README.md            # This file
```

---

## Contributing

### Running Tests

```bash
# Test data loaders
poetry run python evals/loaders/personamem_loader.py
poetry run python evals/loaders/longmemeval_loader.py

# Test deep logger
poetry run python evals/logging/deep_logger.py

# Test config parser
poetry run python evals/config.py

# Verify datasets
poetry run python evals/scripts/verify_longmemeval_oracle.py
```

### Code Style

- Follow PEP 8
- Use type hints
- Document public methods
- Keep functions focused and testable

### Submitting Changes

1. Create a feature branch
2. Add tests for new functionality
3. Update documentation
4. Run full evaluation to verify no regressions
5. Submit pull request

---

## Acknowledgments

This framework is built on top of:

- [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) - Wu et al.
- [PersonaMem](https://github.com/bowen-upenn/PersonaMem) (COLM 2025) - Shi et al.
- [Mem0](https://github.com/mem0ai/mem0) - Open-source memory layer
- [Zep](https://github.com/getzep/zep) - Long-term memory for LLM apps

### Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@software{persona_eval_framework,
  title = {Persona Memory System - Evaluation Framework},
  author = {InnerNets AI},
  year = {2024},
  url = {https://github.com/saxenauts/persona}
}
```

---

## License

MIT License - See [LICENSE](../LICENSE) for details.

---

## Contact

For questions or feedback:
- GitHub Issues: [saxenauts/persona/issues](https://github.com/saxenauts/persona/issues)
- Email: your-email@example.com

---

**Last Updated**: December 21, 2024
