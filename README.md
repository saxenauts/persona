# Persona - Intelligent User Memory

![Persona Banner](docs/assets/banner.svg)

**The memory layer for the next generation of intelligent agents.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-buildpersona.ai-green)](https://docs.buildpersona.ai)

## 📚 Documentation

- **Official Docs**: [docs.buildpersona.ai](https://docs.buildpersona.ai)
- **Deep Dive**: [The Philosophy of Persona (saxenauts.io)](https://saxenauts.io/blog/persona-graph)

---

## Why Persona?

Memory in AI today is often treated as simple storage—a static database of vectors or JSON blobs. Tools like **Mem0** and **Graphiti** provide excellent infrastructure for this, handling the "storage" aspect efficiently.

But true intelligence requires more than storage. It requires a dynamic, living system that evolves with the user.

**Persona aims to be that intelligent layer.**

- **Beyond Storage**: Just as OpenRouter revolutionized model access by adding intelligence on top of raw APIs, Persona adds intelligence on top of raw memory.
- **Memetic Organism**: We view the user's digital identity not as a table of rows, but as a living graph—a "memetic organism" that grows, forgets, and reinforces connections based on interaction.
- **Intelligent Features**:
  - **Associative Recall**: Like the human mind, retrieving one memory triggers related ones through graph connections.
  - **Narrative Continuity**: Automatically links events in temporal chains to understand "stories", not just facts.
  - **Psychological Profiling**: Explicitly models "Psyche" (traits, values) separate from "Episodes" (events).

## Features

- **Dynamic Knowledge Graph**: Automatically builds a graph from unstructured text.
- **Typed Memory System**: distinct `Episode`, `Psyche`, `Entity`, and `Note` nodes.
- **Temporal Chaining**: Narrative continuity for life-logging and long-term companions.
- **Context Retrieval**: Vector similarity + graph traversal for tool-based recall.
- **Structured Insights**: Ask questions and get JSON data, not just text.

---

## Benchmarks

Persona v0.3 includes critical bug fixes (H1-H5) validated through partial evaluation:

| Metric | v0.2 Baseline | v0.3 Partial Validation | Delta |
|--------|---------------|------------------------|-------|
| PersonaMem Accuracy | 65.3% | 66%* | +0.7% |
| Generic Response Rate | 43% | 0%* | -43% ✓ |

*Partial validation: 50 questions, seed 42. See [methodology](release_artifacts/methodology.md) for limitations.

**Key Improvements**:
- **Generic response problem solved**: 0% vs 43% baseline (Psyche inference working)
- **Temporal bugs fixed**: H1-H5 bug fixes improve chronological reasoning
- **Scientific integrity**: Full methodology + limitations documented

**PersonaMem** measures personal memory accuracy across 500 synthetic user profiles. Frontier models (GPT-4.5, o4-mini, Gemini-2.0) achieve ~50% per the [benchmark paper](https://arxiv.org/abs/2501.14260).

> **Note**: Partial validation with GPT-5.2 via Azure Foundry. Full methodology, limitations, and reproducibility package: [release_artifacts/methodology.md](release_artifacts/methodology.md).

---

## Quick Start

### Backend Support
Currently, Persona supports **Neo4j** (via Docker) as the primary graph backend.
> 🚧 **Coming Soon**: We are actively working on decoupling the storage layer to support **Qdrant**, **FalkorDB**, and generic vector/graph stores.

### Installation

```bash
git clone https://github.com/saxenauts/persona.git
cd persona
```

Create a `.env` file:

```env
# Graph Database (Neo4j)
URI_NEO4J=bolt://localhost:7687
USER_NEO4J=neo4j
PASSWORD_NEO4J=your_secure_password

# AI Services (OpenAI - easiest setup)
LLM_SERVICE=openai/gpt-4o
EMBEDDING_SERVICE=openai/text-embedding-3-small
OPENAI_API_KEY=your_openai_api_key

# Server runtime
UVICORN_WORKERS=1
UVICORN_RELOAD=true
```

Start the stack:

```bash
docker compose up -d
```

Explore the API at `http://localhost:8000/docs`.

---

## Roadmap Core Themes

- **Intelligent Forgetting**: Decay mechanisms for irrelevant memories.
- **Agentic Updates**: Self-healing graph that corrects its own contradictions.
- **Real-time Context**: Streaming updates for low-latency personalization.
- **Cross-User Intelligence**: (Planned) Privacy-preserving shared insights.



## License

MIT License. See [LICENSE](LICENSE) for details.
