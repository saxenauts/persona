# Persona — Graph-Vector Agent Memory

![Persona Banner](docs/assets/banner.svg)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Complete](https://img.shields.io/badge/status-complete-green.svg)]()

**State of the art in the graph-vector hybrid era of agent memory.** 4-pillar cognitive model, Neo4j graph storage, HNSW vector indices. 81.3% on LongMemEval. The only published benchmark comparison where Persona, Mem0, and Graphiti ran through the same evaluation framework under identical conditions.

Agents and reasoning systems outgrew dedicated retrieval infrastructure. Development continues as **[Syke](https://github.com/saxenauts/syke)**.

---

## The Origin

In 2020, a paper made the case that large language models are open knowledge graphs. At the same time, a startup was trying to model users' digital footprints across APIs — calendar, email, activity, preferences — to personalize their experience. Knowledge graphs were the right structure for that complexity. But they were too heavy. Maintaining them required hand-designed heuristics, rigid schemas, manually specified ontologies. The structure had to be decided upfront, which meant it was always wrong by the time the data arrived.

By 2023, LLMs could generate triplets reliably enough that the schema could be emergent — you didn't have to design the structure, you could let the model produce it from conversation. That changed everything about what a personal memory system could be. But the field wasn't there yet. Everyone was doing vector DB. Flat, stateless, no temporal awareness, no causal chains. It worked well enough for document retrieval. It missed everything that makes human context actually complex.

2024 to 2025 was the graph-RAG hybrid era. The argument was simple: semantic similarity alone can't tell you that someone changed their mind, that one event caused another, that a preference evolved over six months. Graphs can represent those relationships. Vectors find the relevant neighborhood. You need both. This was a contrarian position — the common critique was *"why do you need a graph, just use a vector database"* — and building Persona was in part an answer to that question, with data.

By the end of 2025, agents were reasoning well enough that the question changed. The causal chains and temporal links that graphs encoded explicitly could now be reasoned over from text. The associativity that required infrastructure now required a better loop.

---

## What We Built

A graph-vector hybrid memory system designed from cognitive science first principles. Four memory types with distinct semantics:

- **Episode** — what happened. Narrative evidence, append-only, anchored in time.
- **Psyche** — who they are. Enduring identity facets that consolidate and evolve.
- **Entity** — what exists. People, places, projects — upserted with conflict handling.
- **Note** — what to do. Tasks and intentions on a state machine.

No keyword routing, no intent classifiers, no heuristic gating. Every decision made by the model through prompt engineering — what we called LLM-first design. The tool layer budgeted ten calls per query: semantic recall, graph traversal, temporal chains, causal link following, entity expansion.

HiMem (arXiv 2601.06377), published January 2026 by a team that never saw our code, independently converged on Episode + Note as the fundamental memory dichotomy. Two of four pillars, validated by convergent evolution.

---

## The Results

| Benchmark | Score | Scope | Methodology |
|---|---|---|---|
| LongMemEval | **81.3%** | 300Q, 3 seeds | CI [78.5%, 84.2%], Docker-locked |
| PersonaMem (full) | **66.2%** | 589Q, seed 42 | Single-seed baseline |
| PersonaMem (subset) | **65.3%** | 150Q, 3 seeds | Audit-grade, Docker-locked |
| BEAM (10 abilities) | **69.0%** | 100Q, seed 1 | instruction/preference/summarization: 100% |

**Competitive comparison across all major benchmarks:**

| System | PersonaMem | LongMemEval | BEAM | Methodology |
|---|---|---|---|---|
| **Persona** | **65.3%** | **81.3%** | **0.69** | 3 seeds, CI [78.5%, 84.2%] on LongMemEval, Docker-locked |
| Mem0 | 61.9% | — | — | Run through our framework; no self-published score on either benchmark |
| Graphiti | — | 71.2% | — | Their published figure; single run, no CI |
| Honcho | — | 90.4%† | 0.630–0.649 | Single run, no CI; †raw model baseline (92.0%) outperforms their augmented score |

**Persona holds the highest published score by any memory system on PersonaMem.** The best LLM-only baseline is 52% (GPT-4.5, Gemini 1.5-Flash); an independent evaluation (arXiv:2603.04814) ran a Mem0-based system and found 62.48% — consistent with our 61.9% run of Mem0 and below our 65.3%.

On BEAM, Persona's 0.69 is scored on the ability-based variant; Honcho's 0.630–0.649 is on context-length variants of the same benchmark. On LongMemEval, Honcho's 90.4% uses Claude Haiku 4.5 — a configuration where Gemini 3 Pro alone, with no memory augmentation, scores 92.0% on the same test.

Audit artifacts with checksums: [`release_artifacts/audit_2026-01-31/`](release_artifacts/audit_2026-01-31/).

---

## The Finding

We gave the agent ten tool calls per query. Mean tool calls: **1.02**. Graph tool usage: **0%**.

Of every query the agent got wrong, **97.3% had the correct answer already present in the retrieved context.** Retrieval similarity gap between correct and incorrect answers: **0.009** (0.836 vs 0.827). The system found the right information. It couldn't tell which answer was right once it had both.

This is a reranking problem, not a retrieval problem. And it pointed at exactly what the memory field had not yet solved.

---

## What Was Being Tried — and Why It Changed

By late 2025, the path to fixing retrieval discrimination was becoming clear across the field. The approaches being developed:

**Frequency and recency weighting** — surface memories accessed often or recently above semantically similar but stale ones. MemGPT/Letta implemented access-count decay signals.

**Saliency scoring** — attach importance scores at ingestion time, use them as retrieval priors. A-MEM and EverMemOS both built importance fields into their memory nodes.

**Hybrid retrieval with RRF** — combine BM25 keyword matching with vector similarity via Reciprocal Rank Fusion, then rerank the fused set. EverMemOS and Zep both moved in this direction, achieving strong LoCoMo scores through multi-path fusion.

**Intent-aware retrieval planning** — before searching, reason about what kind of memory would answer the question. SimpleMem's intent planning stage directly targeted the query formulation failure that RAG-style systems had.

**RL-trained memory policies** — MemRL introduced Q-value scoring so each memory carries a learned utility score alongside its embedding. Memory-R1, AgeMem, and Mem-alpha all converged on the same idea: memory operations should be learned through reward signals, not engineered as heuristics.

**Then something changed the frame entirely.**

Mastra's Observational Memory achieved 94.87% on LongMemEval — the highest score ever recorded — with no vector database and no per-turn dynamic retrieval. Two background agents watch conversations and maintain a compressed observation log. The context window stays bounded. It beats the oracle. The intelligence moved from the retrieval layer to the agent layer.

The reranking problem that Persona couldn't solve wasn't fixed by better reranking. It became irrelevant when agent loops matured enough to reason over retrieved context rather than select from it. Recursive Language Models (arXiv 2512.24601) and RL-trained memory policies completed the picture: by 2026, the compute was in the agent loop, not in the retrieval infrastructure.

---

## What Continues

**[Syke](https://github.com/saxenauts/syke)** takes what Persona proved and rebuilds around what Persona revealed.

| Persona Insight | Syke Design |
|---|---|
| Agent loop prevents iteration (1.02 calls) | The agent loop IS the product. Iterate until done. |
| Graph tools at 0% usage | No graph database. Structure lives in text. |
| Vector DB = complexity for marginal gain | SQLite + FTS5. Single file. BM25. Zero ops. |
| Retrieval works but discrimination doesn't | Optimize for reasoning quality, not retrieval accuracy. |
| Consolidation silently broken | Fewer moving parts. What can't break silently won't. |

Same principles. Better computing paradigm.

**The principle that survived every iteration:** computers have to understand humans in whatever the fastest and best way possible — not by modeling identity, which is both computationally expensive and philosophically wrong, because identity is dynamic. It changes with every new project, tool, hobby, life phase. What's stable enough to learn from is the digital footprint: what actually happened, what was actually said, what was actually done. Benchmarks will always lag this because they measure recall against a fixed gold label. The real measure is whether the system understands you better tomorrow than it did today.

**A prediction:** the field is currently stuck in a debate between Markdown, freeform text, vector embeddings, structured schemas, and open-ended ontologies — each camp convinced its representation is the right substrate for memory. Syke went back to SQLite and plain text, with reasoning handled entirely by the model. The prediction is that everyone eventually arrives here. Not because simpler is better in principle, but because models are getting cheaper and smarter faster than retrieval infrastructure is. At some point you stop fighting the intelligence and let it work.

---

---

## Architecture (for reference)

```
persona/           # Core library
├── adapters/      # PersonaAdapter (single entry point)
├── core/          # Graph ops, retrieval, context formatting
├── llm/           # LLM clients, embeddings, prompts
├── models/        # Memory types (Episode/Psyche/Entity/Note)
├── services/      # Ingestion, persona service, consolidation
└── tools/         # recall/browse/expand/follow/record/update

server/            # FastAPI application
tests/             # Unit + integration tests
```

**LLM-First Design**: No keyword routing, no intent classifiers, no heuristic gating. Every decision made by the model through prompt engineering. The field is converging on this anyway — RL-trained memory policies (MemRL, Memory-R1, AgeMem, Mem-alpha) are all learned, not hardcoded.

---

## Running Persona

Still works if you want to explore or fork.

```bash
git clone https://github.com/saxenauts/persona.git
cd persona
```

Create `.env`:
```env
URI_NEO4J=bolt://localhost:7687
USER_NEO4J=neo4j
PASSWORD_NEO4J=your_secure_password
LLM_SERVICE=openai/gpt-4o
EMBEDDING_SERVICE=openai/text-embedding-3-small
OPENAI_API_KEY=your_openai_api_key
```

```bash
docker compose up -d        # Start Neo4j + API
# API at http://localhost:8000/docs
poetry run pytest tests/unit -v  # Run tests
```

---

## Deep Dive

**[`docs/CLOSURE.md`](docs/CLOSURE.md)** — the full story. What we built, what we measured, what failed, why the benchmarks are broken, where the field went, and what we'd tell you if you were starting today. 578 lines. Worth reading if you're building agent memory.

---

## Acknowledgments

306 commits. 3.5 years. None of it built alone.

**Claude Sonnet 3.5 + GPT o3 Pro** (early 2025) — the research phase. Long conversations about cognitive memory models, graph schema design, what episodic vs semantic memory means computationally. Claude was the thinking partner for the 4-pillar taxonomy. o3 Pro held the full architecture in context and stress-tested it.

**Cursor + GPT-4 + GPT-5 + Sonnet 4.5** (mid 2025) — the building phase. Cursor rewrote the graph design cleaner than I could have, restructured Neo4j operations, and carried out the first real evaluations. GPT-5 and Sonnet 4.5 came online mid-build and immediately became the workhorses — faster iteration, better code generation, deeper context. This is where I learned that evals with LLMs are a craft, not a checkbox.

**GPT-5.1 + GPT-5.2 + GLM-4.7 + OhMyOpenCode** (late 2025 – Jan 2026) — the push. OhMyOpenCode's agent harness (Sisyphus, Oracle, Explore, Librarian, Momus, Metis) turned a solo project into something with the throughput of a small team. GPT-5.2 did the hardcore final-push research and ran the precise data science experiments that produced the audit-grade numbers. GLM-4.7 from Zhipu brought a different perspective to the architecture debates. Every number in this document was verified during this sprint.

**Claude Opus 4.5 + Opus 4.6 + GLM-5 + Kimi 2.5** (Feb 2026) — the closure. Opus 4.5 did the early adapter layer work. Opus 4.6 showed up as an experiment for a week and ended up rewriting Persona's retrieval into a new-age agentic memory system — and wrote the closure document. GLM-5 and Kimi 2.5 were part of the broader model ecosystem tested during the final research push.

---

> *Still round the corner there may wait*
> *A new road or a secret gate,*
> *And though we pass them by today,*
> *Tomorrow we may come this way*
> *And take the hidden paths that run*
> *Towards the Moon or to the Sun.*
>
> — J.R.R. Tolkien

*Where language took over → forever.*

---

MIT License. See [LICENSE](LICENSE) for details.
