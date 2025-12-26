# Persona Memory System - Evaluation Framework

A modular evaluation framework for testing long-term memory systems against academic benchmarks, featuring deep observability, reproducible sampling, and multi-system comparison.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This evaluation framework tests memory systems on their ability to:

| Capability | Description | Benchmark |
|------------|-------------|-----------|
| **Recall** | User-stated facts across conversations | PersonaMem |
| **Temporal** | Date ordering, durations | LongMemEval |
| **Aggregation** | Multi-session data (counting, comparison) | LongMemEval |
| **Updates** | Knowledge changes over time | LongMemEval |
| **Personalization** | Preference-based suggestions | PersonaMem |
| **Abstention** | Knowing when to say "I don't know" | LongMemEval |

---

## Features

**Dual Benchmark Support**: [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025, 500 questions) and [PersonaMem](https://github.com/bowen-upenn/PersonaMem) (COLM 2025, 589 questions)

**Modular Architecture**: Protocol-based adapters, composable metrics, async-first execution engine

**Deep Observability**: Structured JSONL logging with retrieval quality metrics, timing breakdowns, failure pattern detection

**Multi-System Comparison**: Pluggable adapter architecture with parallel evaluation support

---

## Quick Start

```bash
# Install dependencies
poetry install

# Download benchmark datasets
poetry run python evals/scripts/download_personamem.py

# Run a quick evaluation
poetry run python -m evals.cli run \
  --benchmark longmemeval \
  --samples 5 \
  --seed 42

# Explore results in browser
poetry run python -m evals.cli explore
```

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-openai-key"

# Optional
export LLM_SERVICE="openai/gpt-4o-mini"
export EMBEDDING_SERVICE="openai/text-embedding-3-small"
export EVAL_JUDGE_MODEL="gpt-4o"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Engine                               │
│  - Loads benchmarks, registers metrics                      │
│  - Orchestrates: reset → ingest → query → evaluate → report │
│  - Emits events to EventStore                               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  v
┌─────────────────────────────────────────────────────────────┐
│                     AsyncExecutor                           │
│  - Semaphore concurrency control                            │
│  - TokenBucket rate limiting                                │
│  - Retry with exponential backoff                           │
│  - TaskGroup structured concurrency                         │
└─────────────────────────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        v                   v
┌───────────────┐   ┌───────────────┐
│   Adapters    │   │    Metrics    │
│ (Protocol)    │   │  (8 built-in) │
└───────────────┘   └───────────────┘
```

### Key Design Decisions

**Protocol-based Interfaces**: Structural subtyping via Python `Protocol`. Any class implementing `ingest` and `retrieve` works as an adapter without inheriting from a base class.

**Structured QueryResult**: Every query returns a rich object with answer, retrieved nodes, usage stats, and timing data.

**Binary Pass/Fail**: No noisy 1-5 scales. Metrics return binary results, easier to calibrate and more consistent.

**Async-First Engine**: Built for high-concurrency runs with built-in retries and rate limiting per adapter.

**Composable Metrics**: Single-purpose metrics (`ContainsAnswer`, `OptionExtractor`) composed via `AllOf`/`AnyOf` wrappers.

---

## Benchmarks

| Benchmark | Questions | What it Tests | Scoring |
|-----------|-----------|---------------|---------|
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 500 | Temporal logic, multi-session aggregation | LLM judge |
| [PersonaMem](https://github.com/bowen-upenn/PersonaMem) | 589 | Factual precision, personalization | Exact match (a/b/c/d) |

**Test set options:**

| Set | Purpose | Size |
|-----|---------|------|
| `--samples 2` | Smoke test | ~12 questions |
| `--samples 10` | Quick test | ~60 questions |
| `--golden-set` | Full eval | ~695 questions |

---

## Metrics

| Metric | Type | Purpose |
|--------|------|---------|
| `llm_binary_judge` | Binary | Reference-based LLM judgment |
| `abstention_accuracy` | Binary | Correctly refuses when info absent |
| `semantic_similarity` | Continuous | Vector cosine similarity |
| `context_precision` | Continuous | Relevant chunks in retrieved context |
| `context_recall` | Continuous | Ground-truth info found in context |
| `binary_exact_match` | Binary | Strict string comparison |
| `option_extractor` | Binary | Multiple choice extraction (a/b/c/d) |
| `contains_answer` | Binary | Substring match |

---

## CLI Usage

### Running Evaluations

```bash
# Using config files (recommended)
poetry run python -m evals.cli run --config evals/configs/full_eval.yaml

# Command-line arguments
poetry run python -m evals.cli run \
  --benchmark longmemeval \
  --types multi-session,temporal-reasoning \
  --samples 20 \
  --seed 42 \
  --workers 5

# Compare multiple adapters
poetry run python -m evals.cli run \
  --benchmark longmemeval \
  --adapter persona \
  --adapter graphiti \
  --samples 15
```

### Analyzing Results

```bash
# Summary report
poetry run python -m evals.cli analyze run_20241221_143052 --summary

# Filter by question type
poetry run python -m evals.cli analyze run_20241221_143052 --type multi-session

# Show only failures
poetry run python -m evals.cli analyze run_20241221_143052 --failures

# Launch explorer UI
poetry run python -m evals.cli explore
```

---

## Extending the Framework

### Adding a Memory System

Implement the `MemorySystemAdapter` protocol:

```python
from evals.core.models import Session, QueryResult
from evals.core.interfaces import AdapterCapabilities

class MySystemAdapter:
    """Implements MemorySystemAdapter protocol."""
    
    @property
    def name(self) -> str:
        return "mysystem"
    
    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(supports_async=False, supports_retrieval_items=True)
    
    def reset(self, user_id: str) -> None:
        """Clear memory for test isolation."""
        pass
    
    def add_sessions(self, user_id: str, sessions: list[Session]) -> None:
        """Ingest conversation sessions into memory."""
        for session in sessions:
            self._ingest(user_id, session.content, session.date)
    
    def query(self, user_id: str, query: str, *, trace: bool = True) -> QueryResult:
        """Retrieve context and generate answer."""
        context = self._search(user_id, query)
        answer = self._generate(query, context)
        return QueryResult(answer=answer)
```

### Adding a Benchmark

Implement the `Benchmark` protocol:

```python
from evals.core.models import TestCase, Session

class MyBenchmark:
    """Implements Benchmark protocol."""
    
    @property
    def name(self) -> str:
        return "mybenchmark"
    
    @property
    def version(self) -> str:
        return "1.0"
    
    def load(self, *, variant: str | None = None) -> list[TestCase]:
        return [
            TestCase(
                id="q1",
                benchmark="mybenchmark",
                user_id="user_1",
                query="What is my name?",
                reference_answer="Alice",
                sessions=[Session(content="My name is Alice", date="2024-01-01")],
                question_type="recall",
            )
        ]
    
    def default_metrics(self) -> list[str]:
        return ["contains_answer"]
    
    def sample(self, sizes: dict[str, int], *, seed: int | None = None, variant: str | None = None) -> list[TestCase]:
        # Implement stratified sampling
        pass
```

### Adding a Metric

Extend `BaseMetric`:

```python
from evals.metrics.base import BaseMetric
from evals.core.models import TestCase, QueryResult, MetricResult

class MyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "my_metric"
    
    @property
    def kind(self) -> str:
        return "end_to_end"
    
    @property
    def score_type(self) -> str:
        return "binary"
    
    async def evaluate(
        self, test_case: TestCase, result: QueryResult, *, resources: dict
    ) -> MetricResult:
        passed = test_case.reference_answer.lower() in result.answer.lower()
        return MetricResult(
            metric=self.name,
            kind=self.kind,
            score_type=self.score_type,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="Answer contains reference" if passed else "Answer missing reference",
        )
```

---

## Project Structure

```
evals/
├── core/                 # Foundation types and interfaces
│   ├── models.py        # TestCase, QueryResult, MetricResult
│   ├── interfaces.py    # Protocol definitions
│   └── compat.py        # Legacy adapter wrapper
│
├── engine/              # Orchestration and execution
│   ├── engine.py        # Main Engine class
│   └── executor.py      # AsyncExecutor with rate limiting
│
├── metrics/             # Evaluation metrics
│   ├── base.py          # BaseMetric, AllOf, AnyOf
│   ├── exact_match.py   # String matching metrics
│   ├── retrieval.py     # Context precision/recall
│   └── llm_judge.py     # LLM-based evaluation
│
├── benchmarks/          # Benchmark loaders
│   ├── registry.py      # Benchmark registry
│   ├── personamem.py    # PersonaMem benchmark
│   └── longmemeval.py   # LongMemEval benchmark
│
├── adapters/            # Memory system adapters
│   ├── base.py          # Legacy base class
│   ├── persona_adapter.py
│   └── zep_adapter.py
│
├── explorer/            # Web UI for results
│   ├── app.py
│   └── templates/
│
├── tests/               # Test suite (80+ tests)
├── configs/             # YAML configurations
├── data/                # Benchmark datasets (gitignored)
├── results/             # Run outputs (gitignored)
├── cli.py               # CLI interface
└── README.md            # This file
```

---

## Results (December 2025)

### LongMemEval Performance

| System | Overall | Multi-Session | Temporal | Knowledge Update |
|--------|---------|---------------|----------|------------------|
| **Persona** | **64.1%** | **68.3%** | **36.7%** | **75.0%** |
| Graphiti | 53.2% | 29.6% | 22.5% | 59.8% |

**Key Findings:**
- Persona outperforms Graphiti by +10.9% overall
- Strongest advantage in multi-session aggregation (+38.7%)
- Both systems struggle with temporal reasoning (<40%)

### Latency Comparison

| System | Ingestion (avg) | Retrieval (avg) |
|--------|-----------------|-----------------|
| Persona | 61s | 12.8s |
| Graphiti | 949s (15.8m) | 1.2s |

---

## Contributing

```bash
# Run tests
poetry run pytest evals/tests/ -v

# Type check
poetry run mypy evals/

# Format
poetry run ruff format evals/
poetry run ruff check evals/ --fix
```

### Submitting Changes

1. Create a feature branch
2. Add tests for new functionality
3. Update documentation
4. Run full test suite
5. Submit pull request

---

## Acknowledgments

Built on top of:
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) - Wu et al.
- [PersonaMem](https://github.com/bowen-upenn/PersonaMem) (COLM 2025) - Shi et al.

Research informed by:
- [Hamel Husain](https://hamel.dev/blog/posts/eval-tools/) - Eval best practices
- [Eugene Yan](https://eugeneyan.com/writing/qa-evals/) - Long-context Q&A evaluation
- [Groq OpenBench](https://github.com/groq/openbench) - Benchmark framework patterns

---

## License

MIT License - See [LICENSE](../LICENSE) for details.

---

**Last Updated**: December 26, 2025
