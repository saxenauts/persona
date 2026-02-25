# Persona — Intelligent User Memory

![Persona Banner](docs/assets/banner.svg)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Dormant](https://img.shields.io/badge/status-dormant-lightgrey.svg)]()

**A graph-vector hybrid memory system for AI agents.** Built mid-2022 to early 2026. Beat Mem0 on PersonaMem (65.3% vs 61.9%, audit-grade). Proved the paradigm — then watched agents outgrow it. Development continues through **[Syke](https://github.com/saxenauts/syke)**.

---

## What Happened

We built the best version of vector-DB-backed agent memory we could. A 4-pillar cognitive model (Episode, Psyche, Entity, Note), Neo4j graph storage, HNSW vector indices, a tool layer with ten calls budgeted per query — graph traversal, temporal chains, causal links, entity expansion. We beat Mem0 with audit-grade methodology: Docker-locked environments, three seeds, every number traced from artifact to headline.

Then we looked at how it actually performed — and found two problems stacked on top of each other.

**Problem 1: Retrieval works, but reranking doesn't.** Mean tool calls: 1.02 (budget: 10). Graph tool usage: 0%. Of queries the agent got wrong, 97.3% had the correct answer already in the retrieved context. The system finds the right information. Choosing between candidates once you have them is a reranking problem — basics — and our agent loop never got a chance to do it. A confidence stop at 0.88 fired on 100% of queries, terminating the loop before the agent could iterate, compare, or re-query. Every intervention we tried to fix this (psyche gates, temporal evolution, AttractorCards, iteration prompts) either had zero impact or made things worse: **-4.2pp net from baseline.**

**Problem 2: The evals themselves are broken.** When we stripped MCQ options and sent just the question, accuracy dropped from 64% to 30% — and the agent started storing questions instead of answering them. The MCQ format provides the retrieval signal that the agent loop should generate on its own. Four PersonaMem questions have duplicate options (literally unanswerable). In suggest_new_ideas, the gold answer is the shortest option in 19/20 cases — it's measuring verbosity resistance, not memory. Multiple teams have converged on this critique: BEAM, LoCoMo-Plus, Honcho, and Zep have all documented fundamental issues with how current benchmarks measure agent memory.

So the picture is: retrieval works (0.836 similarity), reranking is the gap (fixable in principle), but the benchmarks measuring the gap are themselves measuring the wrong thing. Meanwhile, Mastra's Observational Memory achieved 94.87% on LongMemEval with no vector database and no per-turn retrieval — just two background agents maintaining compressed observations. The intelligence moved from the retrieval layer to the agent layer.

**The full story — what we built, what we tried, why the benchmarks are broken, where the field went, and what we'd tell you if you were starting today — is in [`docs/CLOSURE.md`](docs/CLOSURE.md).**

---

## The Results

| Claim | Score | Scope | Methodology |
|-------|-------|-------|-------------|
| PersonaMem (subset) | 65.3% | 150Q, 3 seeds | Audit-grade, Docker-locked |
| PersonaMem (full) | 66.2% | 589Q, seed 42 | Single-seed baseline |
| BEAM (10 abilities) | 69.0% | 100Q, seed 1 | `event_ordering=0%` dragging average |
| vs Mem0 | +3.4pp | 65.3% vs 61.9% | Same benchmark, documented methodology gaps in Mem0 |

Frontier models (GPT-4.5, o4-mini, Gemini-2.0) achieve ~50% on PersonaMem per the [benchmark paper](https://arxiv.org/abs/2501.14260). Audit artifacts: [`release_artifacts/audit_2026-01-31/`](release_artifacts/audit_2026-01-31/).

---

## What We Learned

**The ceiling is in the agent loop, not the memory layer.** Retrieval accuracy was 0.836. The system finds the right information. What it can't do — because our loop short-circuits after one call — is reason over what it found, compare candidates, re-query, or follow links. Better storage cannot fix an agent that won't iterate.

**Benchmarks measure format parsing, not memory.** When we stripped MCQ options and sent just the question, accuracy dropped from 64% to 30% — and the agent started storing questions instead of answering them. The MCQ format provides the retrieval intent that the agent loop should generate. Four questions have duplicate options (literally unanswerable). Verbosity bias in suggest_new_ideas: gold answer is shortest in 19/20 cases.

**The field moved past dedicated retrieval infrastructure.** In 2024, tool calls matured. In 2025, agent loops matured. In 2026, agent loops absorbed retrieval. Each phase didn't improve retrieval — it made retrieval less necessary. The concepts that graphs and vectors provide survive, but they're migrating to text-native representations and agent-loop-native iteration.

**The 4-pillar model is cognitively valid.** HiMem (arXiv 2601.06377), published January 2026 by a team that never saw our code, independently converged on Episode + Note as the fundamental memory dichotomy. Two of four pillars, validated by convergent evolution.

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

**4-Pillar Memory Model**: Episode (what happened) → Psyche (who they are) → Entity (what exists) → Note (what to do). Each with distinct update semantics: append-only, consolidate, upsert, state machine.

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

## Acknowledgments

306 commits. 3.5 years. None of it built alone.

**Claude Sonnet 3.5 + GPT o3 Pro** (early 2025) — the research phase. Long conversations about cognitive memory models, graph schema design, what episodic vs semantic memory means computationally. Claude was the thinking partner for the 4-pillar taxonomy. o3 Pro held the full architecture in context and stress-tested it.

**Cursor + GPT-4 + GPT-5 + Sonnet 4.5** (mid 2025) — the building phase. Cursor rewrote the graph design cleaner than I could have, restructured Neo4j operations, and carried out the first real evaluations. GPT-5 and Sonnet 4.5 came online mid-build and immediately became the workhorses — faster iteration, better code generation, deeper context. This is where I learned that evals with LLMs are a craft, not a checkbox.

**GPT-5.1 + GPT-5.2 + GLM-4.7 + OhMyOpenCode** (late 2025 – Jan 2026) — the push. OhMyOpenCode's agent harness (Sisyphus, Oracle, Explore, Librarian, Momus, Metis) turned a solo project into something with the throughput of a small team. GPT-5.2 did the hardcore final-push research and ran the precise data science experiments that produced the audit-grade 65.3%. GLM-4.7 from Zhipu brought a different perspective to the architecture debates. Every number in this document was verified during this sprint.

**Claude Opus 4.5 + Opus 4.6 + GLM-5 + Kimi 2.5** (Feb 2026) — the closure. Opus 4.5 did the early adapter layer work. Opus 4.6 showed up as an experiment for a week and ended up rewriting Persona's retrieval into a new-age agentic memory system — and wrote the closure document. GLM-5 and Kimi 2.5 were part of the broader model ecosystem we tested against during the final research push. Same principles. Better computing paradigm.
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
