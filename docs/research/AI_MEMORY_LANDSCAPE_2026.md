# The 2026 AI Agent Memory Landscape: A Comprehensive Research Index

> **Purpose**: One document covering the entire 2026 AI memory research landscape — systems, benchmarks, paradigm shifts, and directional recommendations for Persona's architecture. Written for both humans and LLMs to read.
>
> **Date**: February 10, 2026
> **Author**: Saxenauts + Oracle synthesis
> **Status**: Research only — no implementation decisions committed

---

## Document Index

| # | Section | What It Covers |
|---|---------|----------------|
| 1 | [Executive Summary](#1-executive-summary) | The field in 60 seconds |
| 2 | [Persona Today](#2-persona-today) | Current architecture, strengths, known problems |
| 3 | [The RLM Paradigm](#3-the-rlm-paradigm) | Recursive Language Models — the 2026 compute shift |
| 4 | [RL for Memory](#4-rl-for-memory) | MemRL, Memory-R1, Mem-alpha, AgeMem — learned retrieval |
| 5 | [EverMemOS](#5-evermemos) | 93% on LoCoMo — the current SOTA memory OS |
| 6 | [ALMA](#6-alma) | Meta-learning memory designs — the philosophical challenge |
| 7 | [17 New Systems](#7-seventeen-new-systems-jan-feb-2026) | SimpleMem, TiMem, HiMem, CMA, SYNAPSE, and more |
| 8 | [4 New Benchmarks](#8-four-new-benchmarks) | BEAM+LIGHT, MEMORYBENCH, MemoryAgentBench, MemBench |
| 9 | [Surveys & Community](#9-surveys--community) | Key surveys, ICLR 2026 MemAgents workshop |
| 10 | [Founder's Research Notes](#10-founders-research-notes) | The 5-system landscape analysis (EverMemOS, MemRL, AgeMem, A-MEM, Mem0) |
| 11 | [7 Macro Shifts](#11-seven-macro-shifts) | Where the field is heading |
| 12 | [Oracle Synthesis](#12-oracle-synthesis) | Directional recommendations for Persona |
| 13 | [The 10-Year View](#13-the-10-year-view) | What personal AI memory looks like in 2036 |
| 14 | [Appendix A: Paper Catalog](#appendix-a-paper-catalog) | Full citation list with arxiv links |
| 15 | [Appendix B: Benchmark Comparison](#appendix-b-benchmark-comparison) | LoCoMo, PersonaMem, BEAM, MemoryAgentBench side-by-side |
| 16 | [Appendix C: Architecture Comparison Matrix](#appendix-c-architecture-comparison-matrix) | Side-by-side of all major systems |

---

## 1. Executive Summary

**The 2026 memory landscape is exploding.** In January-February 2026 alone, we found 17 new memory systems, 4 new benchmarks, and 3 major surveys — more than the previous two years combined. The field has shifted from "should agents have memory?" to "what kind of memory, and how should it evolve?"

**Three paradigm shifts define the moment:**

**Shift 1 — Recursive Compute Changes Everything**: RLMs (Recursive Language Models, MIT CSAIL) let models recursively call themselves to handle unbounded context. This doesn't replace external memory — it amplifies the need for a persistent, queryable substrate to recurse over. Persona becomes MORE important in an RLM world, not less.

**Shift 2 — RL-Trained Memory Management is Converging**: MemRL, Memory-R1, Mem-alpha, and AgeMem all converge on the same insight: memory operations (store, retrieve, update, summarize, discard) should be learned via reinforcement learning, not hardcoded via heuristics. Q-value scoring for retrieval beats pure semantic similarity. The question isn't whether to adopt RL — it's when.

**Shift 3 — The Bitter Lesson Looms but Hasn't Landed**: ALMA shows learned architectures beat hand-designed ones at scale. MemEvolve does meta-evolution of memory systems. But current production systems (EverMemOS at 93%, SimpleMem at 26.4% F1 improvement) all use hand-designed structures. The hedge: keep pillars as contracts, make everything else learned.

**Where Persona stands:** Our 4-pillar model (Episode/Psyche/Entity/Note) with graph-based storage and memeplex world index is architecturally sound and ahead of the curve (HiMem independently converged on Episode+Note dual memory). But our retrieval is broken (single-pass, links invisible) and our consolidation is weak (manual memeplex refresh). The highest-leverage fixes are known and tractable.

---

## 2. Persona Today

### 2.1 Architecture

```
Ingest → Integration Agent → Consolidation → Memeplex
                                    ↓
                            Neo4j Graph Store
                            (4-Pillar Nodes + Links + Vectors)
                                    ↓
                            PersonaService.run_agent()
                            (Tool Loop: recall/record/browse/expand/follow)
                                    ↓
                            Response to User
```

**4-Pillar Memory Model:**

| Pillar | Cognitive Function | What It Stores | Update Semantics |
|--------|-------------------|----------------|------------------|
| Episode | Episodic evidence | Events, experiences, conversations | Append-only (immutable) |
| Psyche | Self-schema | Traits, preferences, values, beliefs | Consolidate/evolve |
| Entity | Semantic referents | People, places, orgs, projects, concepts | Upsert with conflict handling |
| Note | Agent commitments | Tasks, goals, reminders, ideas | State machine (active→done) |

**Memeplex (World Model Index):** Per-user memory index providing the LLM with a "table of contents" — topics, people, projects, places, concepts, recent_focus, memory_stats. Injected as `{world_model}` in system prompt.

**Link Model:** LED_TO, CAUSED_BY, MENTIONED_IN, RELATES_TO relationships exist in Neo4j but are nearly invisible during retrieval. `expand_neighbors()` and `follow_relationship()` tools exist but are never called (111/111 queries = 1 tool call, 2 turns).

### 2.2 Current Performance

**PersonaMem Eval (MCQ format, 100Q paired arms):**
- Arm A: 63% (63/100), Arm B: 67% (67/100)
- True system accuracy: **65% ±5pp**
- 90% agreement rate between arms (60 both correct, 30 both wrong, 10 swing)
- 30 deterministic failures: 14 suggest_new_ideas, 6 recs, 5 recall, 3 generalizing, 2 reasons
- 83% of deterministic failures pick SAME wrong answer in both arms

### 2.3 Known Problems

**Problem 1 — Single-Pass Retrieval:** System prompt locks tool budget at 1. Agent never does multi-round exploration. Result: temporal queries fail ("yesterday" → 0 results), link-dependent queries impossible.

**Problem 2 — Links Are Invisible:** LED_TO, CAUSED_BY relationships exist in the graph but `expand_neighbors()` / `follow_relationship()` are never called. The entire relational structure is wasted.

**Problem 3 — Weak Consolidation:** Memeplex refresh is manual/scheduled. No automatic clustering, no conflict-aware reconsolidation, no temporal hierarchy. Memeplex has weak correlation with accuracy (Cohen's d=0.237).

**Problem 4 — No Temporal Intelligence:** No date resolution ("yesterday" not converted to actual date), no recency priors, no "what happened next" chaining.

**Problem 5 — Query Formulation:** Agent doesn't reformulate queries. If first recall() returns nothing, it gives up rather than trying alternative phrasings or time windows.

---

## 3. The RLM Paradigm

### 3.1 What Are RLMs?

**Paper**: "Recursive Language Models" — Alexander Zhang et al., MIT CSAIL, December 2025
**ArXiv**: 2512.24601
**Blog**: alexzhang13.github.io + primeintellect.ai/blog/rlm

Recursive Language Models are an inference paradigm where LLMs recursively call themselves via a Python REPL to handle unbounded context. Instead of cramming everything into one context window, the model writes code that calls itself on subsets of data, aggregates results, and recurses until it reaches an answer.

**Key results:**
- RLM(GPT-5-mini) outperforms GPT-5 on the OOLONG benchmark by **114%** at the same cost
- Handles input 2 orders of magnitude beyond the context window
- Core strategies: Peeking (random sampling), Grepping (targeted search), Partition+Map (divide and conquer), Summarization (progressive compression)

### 3.2 Why This Matters for Memory

RLMs are NOT a replacement for external memory. They're an amplifier.

**RLMs need persistent state for cross-session continuity.** RLM can recursively browse within a session, but it can't remember across sessions without an external store. Persona IS that external store.

**RLM-style recursive retrieval could replace single-pass recall.** Instead of one `recall()` call, an RLM-enhanced agent could: (1) recall broadly, (2) identify time anchors, (3) follow links from those anchors, (4) reconcile conflicts, (5) synthesize. This is exactly what Persona's agent loop SHOULD be doing but currently doesn't (single-pass problem).

**Prime Intellect (Jan 2026):** "RLMs are the paradigm of 2026." Planning RL training for long-horizon agents using recursive self-calls. Memory systems that support recursive access patterns will thrive.

### 3.3 Implication for Persona

Persona becomes MORE important in an RLM world. The right move: make Persona's tools (recall, browse, expand, follow) composable primitives that an RLM-style agent can recursively invoke. Remove the tool budget cap. Let the agent recurse until confident.

---

## 4. RL for Memory

### 4.1 The Core Insight

All four major RL-for-memory papers converge on the same insight: **memory operations should be learned via reinforcement learning, not hardcoded via heuristics.** The LLM should learn WHEN to store, WHAT to retrieve, HOW to update, and WHEN to forget — through reward signals, not prompt engineering.

### 4.2 MemRL — Q-Value Retrieval

**Core architecture**: Intent-Experience-Utility triplets `(z_i, e_i, Q_i)` where each memory carries a learned utility score.

**Two-phase retrieval:**
1. Semantic filter: retrieve top-k by embedding similarity
2. Value-aware selection: `score = (1-λ)·similarity + λ·Q_value`

**Key innovation**: Learns to ignore "distractor memories" — memories with high semantic similarity but low task utility. This directly addresses Persona's deterministic failures where recall returns plausible but wrong memories.

**Challenge for personal AI**: Task-based agents have clear reward signals (task success/failure). Personal AI reward is ambiguous — was the memory useful? Proxy metrics possible: user engagement, explicit feedback, downstream answer correctness.

### 4.3 Memory-R1

**Paper**: "Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning"
**Authors**: Sikuan Yan et al. (16 authors)
**ArXiv**: 2508.19828 (v5, revised Jan 2026)
**GitHub**: github.com/yansikuan/memory-r1

**Architecture**: Memory Manager (ADD/UPDATE/DELETE/NOOP) + Answer Agent for refined retrieval. Trained with PPO/GRPO.

**Results**: 28% F1 improvement over heuristic baselines.

**Key insight**: The Memory Manager learns to proactively maintain memory quality — it doesn't just store everything, it curates. It learns to UPDATE contradictory information and DELETE obsolete facts.

### 4.4 Mem-alpha

**Paper**: "Mem-α: Learning Memory Construction via Reinforcement Learning"
**Authors**: Yu Wang et al.
**ArXiv**: 2509.25911
**GitHub**: github.com/wangyu-ustc/Mem-alpha

**Core idea**: Formulates memory updates as a sequential Markov Decision Process. Multi-component external memory system with RL-driven construction.

**Innovation**: Instead of treating memory as a static store, treats each interaction as a state transition where the agent must decide what to add/modify in its memory to maximize future utility.

### 4.5 AgeMem v2

**Paper**: "Agentic Memory: Learning Unified LTM and STM Management for LLM Agents"
**ArXiv**: 2601.01885 (January 2026)

**Key innovation**: Exposes memory operations as tool-based actions (STORE, RETRIEVE, UPDATE, SUMMARIZE, DISCARD), enabling autonomous decision-making. Three-stage progressive RL with step-wise GRPO for sparse/discontinuous rewards.

**Why it matters**: This is the closest architecture to what Persona SHOULD become — tool-native memory with RL-trained policies.

### 4.6 Practical Path for Persona

**6-month recommendation (from Oracle):** Don't jump to full RL. Build the policy surface and data infrastructure first:
1. Add a lightweight utility score to each memory (start heuristic: access count, recency, link degree)
2. Log candidate sets, chosen memories, distractors, and outcomes for every retrieval
3. Implement two-phase retrieval: semantic filter → value-aware selection
4. Graduate to offline RL/GRPO once you have 10K+ retrieval traces with labels

---

## 5. EverMemOS

### 5.1 Overview

**Paper**: "EverMemOS: A Self-Organizing Memory Operating System"
**ArXiv**: 2601.02163 (January 2026)
**GitHub**: github.com/EverMind-AI/EverMemOS
**Result**: **93% on LoCoMo** (vs Mem0 69%, MemOS 85%, Zep 91%)

EverMemOS is the current SOTA memory operating system, achieving the highest published score on the LoCoMo benchmark through a three-phase engram-inspired lifecycle.

### 5.2 Architecture (6 Layers)

| Layer | Purpose |
|-------|---------|
| Agentic | Memory extraction agents, vectorization, retrieval orchestration, reranking |
| Memory | MemCell extraction, episodic management, type classification, LLM prompts |
| Retrieval | Vector search (Milvus), keyword search (Elasticsearch), hybrid RRF, reranking |
| Business | API endpoints, request handling, validation |
| Infrastructure | MongoDB, Redis, Elasticsearch, Milvus adapters |
| Core Framework | DI, lifecycle, middleware, queue, config |

### 5.3 Memory Abstraction

**MemCells → MemScenes → Episodes**

**MemCell**: Atomic memory unit extracted via boundary detection. One-to-many: each MemCell produces multiple downstream memories.
- Fields: event_id, original_data, timestamp, summary, participants, group_id

**MemScene**: Thematic cluster of related MemCells. Provides high-level, semantically coherent context.

**Episode Types:**
- **Group Episode** (user_id: None): Third-person, group-wide narration
- **Personal Episode** (user_id: specific): First-person view for participant

**Additional memory types**: Foresight (predictive, with validity windows), EventLog (atomic facts), Profiles

### 5.4 Retrieval: Fast vs Agentic

**Fast Mode** (latency-sensitive): BM25 keyword + embedding vector + RRF fusion → fast response

**Agentic Mode** (complex queries): LLM query expansion → multiple retrieval paths (parallel) → intelligent RRF fusion → enhanced coverage

### 5.5 Why 93%

Three factors drive the benchmark lead:

**Factor 1 — MemScene-guided retrieval**: Instead of isolated memory fragments, the agent reasons over coherent thematic narratives. This matters for multi-hop questions where fragments don't suffice.

**Factor 2 — Multi-path RRF fusion**: 2-3 complementary retrieval queries run in parallel, fused by Reciprocal Rank Fusion. Outperforms single-path on multi-hop.

**Factor 3 — Consistent temporal semantics**: EverMemOS avoids the timestamp confusion that plagues Mem0 (PDT issues) and Zep (event vs conversation timestamps).

### 5.6 Comparison with Persona

| Dimension | Persona | EverMemOS |
|-----------|---------|-----------|
| Memory org | 4-pillar graph (Neo4j) | 3-tier: MemCells → MemScenes → Episodes |
| Consolidation | Manual memeplex refresh | Automatic, time-bounded clustering |
| Consolidation output | Memeplex (topics, people, etc.) | MemScenes + Episodes + Profiles |
| Temporal | Individual timestamps | Foresight with validity windows |
| Retrieval | Single-pass recall() | Fast (RRF) + Agentic (multi-round) |
| Context format | Prose-based sections | MemScene-guided + multiple memory types |
| Storage | Neo4j + vectors | MongoDB + Milvus + Elasticsearch + Redis |
| Eval | 65% PersonaMem | 93% LoCoMo |

---

## 6. ALMA

### 6.1 Overview

**Paper**: "ALMA: Meta-Learning Memory Architectures for AI Agents"
**Authors**: Jeff Clune lab
**ArXiv**: 2602.07755 (February 2026)
**Predecessor**: ADAS (Automated Design of Agentic Systems, ICLR 2025)

ALMA uses evolutionary search with LLM-based crossover and mutation to automatically discover memory architectures. Instead of hand-designing memory systems, ALMA evolves them.

### 6.2 The Philosophical Challenge

**Key finding**: Learned memory designs beat ALL hand-designed baselines including graph-based ones in game environments.

**Caveats**: (1) Game environments, not personal memory. (2) Requires many evaluation cycles per evolution step. (3) Best designs still share structural features with hand-designed ones (hierarchical organization, temporal awareness). (4) Evolution happens offline, not at runtime.

### 6.3 Implications

ALMA doesn't invalidate Persona's 4 pillars — it suggests that the connectivity and retrieval WITHIN those pillars should be optimized by search rather than hand-tuned. The correct framing: **pillars are the contract, optimization is the variable.**

MemEvolve (Section 7) extends this with meta-evolution and EvolveLab (12 memory systems). Together with ALMA, they represent the "learned memory architecture" research direction that will likely dominate in 3-5 years at scale.

---

## 7. Seventeen New Systems (Jan-Feb 2026)

### 7.1 SimpleMem — Efficient Lifelong Memory

**ArXiv**: 2601.02553 (v3, Jan 29, 2026)
**Authors**: Jiaqi Liu et al. (8 authors)

Three-stage pipeline: (1) Semantic Structured Compression — distills interactions into compact, multi-view indexed memory units. (2) Online Semantic Synthesis — intra-session integration of related context. (3) Intent-Aware Retrieval Planning — infers search intent to determine retrieval scope.

**Results**: 26.4% F1 improvement on LoCoMo + 30-fold token reduction. Best balance of performance and efficiency.

**Relevance to Persona**: The "intent-aware retrieval planning" concept directly addresses our query formulation failure. Instead of blindly searching, the agent first reasons about what kind of memory would answer the question.

### 7.2 TiMem — Temporal-Hierarchical Consolidation

**ArXiv**: 2601.02845 (Jan 2026)
**Authors**: Kai Li et al. (12 authors)

Introduces the Temporal Memory Tree (TMT): conversations organized into a tree structure where raw observations progressively abstract into persona representations. Systematic consolidation without fine-tuning. Complexity-aware recall balances precision and efficiency.

**Results**: 75.30% LoCoMo, 76.88% LongMemEval-S, 52.20% reduction in recalled memory length.

**Relevance to Persona**: TMT directly maps to what our Episode hierarchy SHOULD look like — raw events → consolidated summaries → identity-level abstractions, all connected in a tree.

### 7.3 HiMem — Hierarchical Long-Term Memory

**ArXiv**: 2601.06377 (Jan 2026)
**Authors**: Ningning Zhang et al.

Dual memory types: Episode Memory (via Topic-Aware Event-Surprise Dual-Channel Segmentation) + Note Memory (stable knowledge through multi-stage extraction). Features conflict-aware reconsolidation for self-evolution.

**Relevance to Persona**: HiMem independently converged on Episode + Note as the fundamental memory dichotomy — same as two of our four pillars. Their conflict-aware reconsolidation is exactly what Persona's Psyche consolidation needs.

### 7.4 CMA — Continuum Memory Architecture

**ArXiv**: 2601.09913 (Jan 2026)
**Author**: Joe Logan (Mode7 GK, Japan)

Defines five architectural requirements for long-horizon agent memory: (1) Persistent storage, (2) Selective retention, (3) Associative routing, (4) Temporal chaining, (5) Consolidation.

**Key quote**: "RAG treats memory as a stateless lookup table: information persists indefinitely, retrieval is read-only, and temporal continuity is absent."

**Relevance to Persona**: CMA's 5 requirements serve as a checklist. Persona has 1 (persistent storage), partially has 3 (associative routing via links) and 5 (consolidation via memeplex), and is weak on 2 (selective retention) and 4 (temporal chaining).

### 7.5 Self-Consolidation

**ArXiv**: 2602.01966 (Feb 2026)
**Authors**: Hongzhuo Yu et al.

First framework for distilling non-parametric textual experience into compact learnable parameters. Key innovation: uses FAILED attempts (not just successes) for learning via contrastive reflection.

**Relevance to Persona**: The idea of learning from failures is powerful for memory consolidation. Most systems only consolidate successful interactions. Learning what NOT to store from failed retrievals could improve signal-to-noise.

### 7.6 SYNAPSE — Spreading Activation

**ArXiv**: 2601.02744 (Jan 2026, revised Jan 21)

Episodic-semantic memory fusion via spreading activation — the graph retrieval mechanism from cognitive science where activating one node spreads activation to connected nodes based on link strength.

**Relevance to Persona**: This is exactly what our link model SHOULD enable. LED_TO, CAUSED_BY connections should propagate activation during retrieval. SYNAPSE provides the theoretical framework for making Persona's graph retrievable.

### 7.7 E-mem — Episodic Context Reconstruction

**ArXiv**: 2601.21714 (Jan 2026)
**Authors**: Kaixiang Wang et al.

Shifts from "Memory Preprocessing" to "Episodic Context Reconstruction." Multi-agent ensemble reconstructs context rather than pre-processing memories into fixed structures.

**Key quote**: "Prevalent memory preprocessing paradigms suffer from destructive de-contextualization... By compressing complex sequential dependencies into pre-defined structures, these methods sever contextual integrity."

**Relevance to Persona**: Challenges our integration agent's approach. Instead of extracting and classifying memories upfront, E-mem suggests reconstructing context on-demand from raw episodic traces. Worth studying as an alternative.

### 7.8 CAST — Character-and-Scene Memory

**ArXiv**: 2602.06051 (Jan 2026)

Uses dramatic theory for multi-modal episodic memory. Characters and scenes as organizing principles.

**Relevance to Persona**: Novel framing. "Characters" map to our Entities, "scenes" map to our Episodes. The dramatic theory adds narrative structure that could enhance story-like memory retrieval.

### 7.9 MEM1 — Synthesizing Memory and Reasoning

**Venue**: ICLR 2026 (under review)

Addresses limitations of full-context prompting by synthesizing memory with reasoning. Most LLM systems append all past turns regardless of relevance → unbounded growth → degraded performance.

**Relevance to Persona**: Validates the need for selective, synthesized context rather than raw dump. Our prose-format context is already a step in this direction.

### 7.10 MemGen — Generative Latent Memory

**ArXiv**: 2509.24704 (ICLR 2026 accepted)
**GitHub**: github.com/KANABOON1/MemGen (303 stars)

Interleaves explicit memory synthesis with LLM reasoning via Memory Trigger (detecting key junctures) + Memory Weaver (generating latent tokens).

**Key quote**: "Neither parametric memory nor retrieval-based memory captures the fluid interweaving of reasoning and memory that underlies human cognition."

**Relevance to Persona**: MemGen's "memory trigger" concept — knowing WHEN to synthesize new memory during reasoning — could improve our record() tool's decision-making about when to persist new information.

### 7.11 MemEvolve — Meta-Evolution

**ArXiv**: 2512.18746 (Dec 2025)
**GitHub**: github.com/bingreeky/MemEvolve

Meta-evolution of inductive biases and memory architectures. Dual-evolution: inner loop accumulates experience, outer loop meta-learns. EvolveLab provides unified benchmarking across 12 memory systems.

**Relevance to Persona**: EvolveLab could serve as a comparison framework for benchmarking Persona against other systems. The meta-evolution approach is the medium-term future for architecture search.

### 7.12 AssoMem — Multi-Signal Associative Retrieval (Meta)

**ArXiv**: 2510.10397 (Oct 2025, ICLR 2026)
**Authors**: Meta Research (K. Zhang et al.)

Forms associative memory graph and adaptively fuses multi-dimensional retrieval signals for scalable QA. Addresses similarity-dense scenarios where semantic distance alone fails.

**Relevance to Persona**: Our graph already encodes multiple signal types (semantic, temporal, relational). AssoMem shows how to FUSE these signals adaptively rather than relying on any single one.

### 7.13 MeMo — Memorization Precedes Learning

**ArXiv**: 2502.12851 (Feb 2025)

Novel architecture where explicit token memorization in layered associative memories precedes learning. Paradigm shift from "learn then remember" to "remember then learn."

### 7.14 CAMELoT — Training-Free Associative Memory

**ArXiv**: 2402.13449 (Feb 2024)

Training-free consolidated associative memory with consolidated episodic buffers. SOTA recall without fine-tuning.

### 7.15 Additional Systems (Summary)

| System | ArXiv | Key Innovation |
|--------|-------|---------------|
| Memvid V2 | memvid.com | Portable, deterministic memory (debugging focus) |
| MemVerse | 2512.03627 | Multimodal memory for lifelong learning agents |
| Agent Drift | 2601.04170 | Quantifying behavioral degradation over time |

---

## 8. Four New Benchmarks

### 8.1 BEAM + LIGHT (ICLR 2026)

**Paper**: "Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs"
**Venue**: ICLR 2026 Poster

Multi-domain benchmark with long (100K–10M token) conversations + comprehensive memory probes. LIGHT framework improves LLM memory by 3.5%–12.69% over strongest baselines.

**Why it matters**: Tests at a scale (10M tokens) that most systems can't handle. The gap between 100K and 10M performance reveals fundamental architectural limitations.

### 8.2 MEMORYBENCH (ICLR 2026)

**ArXiv**: 2510.17281

Benchmark for memory AND continual learning. Focuses on memorization impact on incremental learning — how does adding new memories affect existing knowledge?

**Why it matters**: Addresses catastrophic forgetting, which most memory systems ignore. As memory grows, can the system still recall early memories accurately?

### 8.3 MemoryAgentBench (ICLR 2026)

**ArXiv**: 2507.05257
**GitHub**: github.com/HUST-AI-HYZ/MemoryAgentBench (223 stars)
**Authors**: Yuanzhe Hu, Yu Wang, Julian McAuley (UCSD)

Four core competencies: (1) Accurate Retrieval (AR), (2) Test-Time Learning (TTL), (3) Long-Range Understanding (LRU), (4) Conflict Resolution (CR).

**Why it matters**: First benchmark to explicitly test conflict resolution — handling contradictory information. This directly maps to Persona's Psyche reconsolidation challenge.

### 8.4 MemBench (ACL 2025)

**Paper**: "MemBench: Towards More Comprehensive Evaluation on Memory of LLM-based Agents"

Comprehensive multi-dimensional evaluation. Addresses limitations of single-metric benchmarks.

### 8.5 Benchmark Landscape Summary

| Benchmark | Scale | Focus | Persona Relevance |
|-----------|-------|-------|-------------------|
| PersonaMem | 32 users, 240K tokens | Personal facts, preferences | Our primary eval |
| LoCoMo | 10 convos, 270K tokens | Long-term conversational | Direct comparison target |
| BEAM+LIGHT | 100K-10M tokens | Scale stress-testing | Long-horizon capability |
| MEMORYBENCH | Variable | Continual learning | Forgetting/drift |
| MemoryAgentBench | Variable | 4 competencies | Conflict resolution |
| MemBench | Variable | Multi-dimensional | Comprehensive baseline |

---

## 9. Surveys & Community

### 9.1 Key Surveys

**"Memory in the Age of AI Agents" (Dec 2025)**
ArXiv: 2512.13564. Yuyang Hu et al. (9 authors). GitHub tracker: github.com/Shichun-Liu/Agent-Memory-Paper-List (1,162 stars). Comprehensive survey of memory mechanisms, operations, emerging topics. ALMA cites this as a key reference.

**"A Survey on Evolution of LLM Agent Memory Mechanisms" (Jan 2026)**
Authors: Jinghao Luo et al. DOI: 10.20944/preprints202601.0618.v2. Focus: from storage to experience evolution.

**"Rethinking Memory in LLM Based Agents" (May 2025, revised Dec 2025)**
ArXiv: 2505.00675. Focus: representations, operations, emerging topics.

**"Self-Evolving Agents" (Jul 2025)**
ArXiv: 2507.21046v4. Comprehensive survey on self-evolving AI agent systems.

### 9.2 ICLR 2026 MemAgents Workshop

**Date**: April 26-27, 2026, Rio de Janeiro
**Website**: sites.google.com/view/memagent-iclr26
**Focus**: Memory for LLM-Based Agentic Systems — explicit vs in-weights vs interaction-driven memory
**Organizers**: Zhenguang Cai, Wenyue Hua, Keshuang Li, Yunpu Ma, Ercong Nie, Hinrich Schütze, et al.

**Significance**: Dedicated workshop signals the field has matured enough for focused community attention. Papers presented here will define the 2026-2027 research agenda.

### 9.3 GitHub Tracking Repositories

| Repo | Stars | Scope |
|------|-------|-------|
| Agent-Memory-Paper-List | 1,162 | Comprehensive paper tracker |
| awesome-lifelong-llm-agent | — | TPAMI 2026 collection |
| Awesome-Memory-for-Agents | — | TsinghuaC3I curated list |

---

## 10. Founder's Research Notes

*The following summarizes the 5-system landscape analysis provided by the founder (Saxenauts), originally compiled from independent research.*

### 10.1 The Five Systems Studied

**EverMemOS**: "The most production-ready. MemCells → MemScenes hierarchy is elegant. But 6-layer architecture is over-engineered for personal AI."

**MemRL**: "Q-value retrieval is the right idea. The challenge for personal AI: what's the reward signal? Task agents have clear success/failure. Personal memory doesn't."

**AgeMem**: "Unified LTM/STM with GRPO is promising. Three-stage training addresses sparse rewards. Tool-based actions (STORE/RETRIEVE/UPDATE/SUMMARIZE/DISCARD) align with our philosophy."

**A-MEM (Zettelkasten)**: "Zettelkasten-inspired atomic notes with linking. Clean but doesn't handle temporal evolution well. Static once created."

**Mem0**: "Production layer, not research. Simple key-value with optional graph. Good developer experience but limited architecture. 69% on LoCoMo tells the story."

### 10.2 Founder's Key Insights

**Insight 1 — The Hippocampus Analogy**: "Our link model should replicate the role of the hippocampus — not just storing memories but INDEXING them, creating associations, enabling pattern completion. The hippocampus doesn't store content; it stores the INDEX to content distributed across cortex."

**Insight 2 — Self-Organizing Memory**: "The LLM can work on the persona graph with some constraints to shape its evolution towards a self-organizing memory across time. Everything is literally a game — game-theoretic constraints on evolution."

**Insight 3 — The Secondary Entity**: "Regardless of how intelligent LLMs become, we would still need a secondary entity taking the place of the user's mind space and world model."

**Insight 4 — Memetic Organisms**: "The architecture vision doc describes memory as a 'memetic organism' — memes that propagate, compete, and evolve. This is the right framing. Not a database. A living system."

### 10.3 The "Pogu Anima/Animus" Vision

The founder envisions Persona as a "digital anima/animus" — a Jungian concept of the inner complementary self. The system should:
- Hold the user's identity stable across time
- Evolve WITH the user (not lag behind)
- Enable self-reflection (showing the user their own patterns)
- Bridge conscious intentions (Notes) with unconscious patterns (Psyche)

---

## 11. Seven Macro Shifts

### Shift 1 — Memory Becomes First-Class Primitive

Memory is no longer an afterthought bolted onto chat. Every major agent framework (LangChain, CrewAI, AutoGen) now has memory as a core module. The ICLR 2026 MemAgents workshop makes this official.

**Implication for Persona**: We're building in the right space. The market is coming to us.

### Shift 2 — RAG Bifurcation

RAG is splitting into two patterns: (1) Agentic RAG — multi-step, tool-based retrieval with reasoning loops, and (2) Cache Pattern — fast, single-lookup retrieval for known patterns. Most systems will need both.

**Implication**: Persona needs Fast mode (cache pattern for simple recalls) + Agentic mode (multi-round for complex queries). EverMemOS already has this bifurcation.

### Shift 3 — Test-Time Compute Reduces but Doesn't Eliminate External Memory

Larger context windows and RLMs reduce the NEED for external memory in some cases (all data fits in context). But they don't eliminate it for: (1) cross-session state, (2) identity continuity, (3) auditability, (4) cost management (storing everything in context is expensive).

**Implication**: Persona's value proposition shifts from "the model can't remember" to "the model SHOULDN'T have to remember everything every time." Efficiency + identity + audit.

### Shift 4 — MCP Standardizes Integration

Model Context Protocol (MCP) standardizes how LLMs interact with external tools and data. Memory becomes one of many MCP-compatible services.

**Implication**: Persona should be MCP-native. Memory as a service, not a monolith.

### Shift 5 — World Models Emerge as Separate Layer

Multiple systems now implement "world models" — compressed representations of the user's entire context (topics, entities, relationships, current state). Persona's Memeplex IS a world model.

**Implication**: Validate and strengthen Memeplex. It's ahead of the curve.

### Shift 6 — Multi-Agent Demands Shared Memory

As multi-agent systems become standard, shared memory becomes critical. Agents need to read each other's state, coordinate, and avoid redundant work.

**Implication**: Design Persona's graph to support multi-agent access patterns. Not just "my memories" but "our shared understanding."

### Shift 7 — The Bitter Lesson Looms

ALMA and MemEvolve show that learned architectures outperform hand-designed ones at scale. The question is WHEN this becomes dominant, not IF.

**Implication**: Persona's hedge: keep the 4-pillar CONTRACT stable (immutable evidence, evidence/inference separation, temporal grounding, provenance). Make everything WITHIN the contract learnable/evolvable. If learned architectures win, Persona becomes the substrate they operate over.

---

## 12. Oracle Synthesis

*The following is synthesized from Oracle consultation, grounded in all research above.*

### 12.1 Positioning

**What's genuinely differentiated**: Persona is a *personal cognitive substrate* (identity + evidence + referents + commitments) rather than "a chat app with RAG."

**What's commodity**: Vector store + basic summarization + single-shot recall. Even "world model lists" are trending toward baseline features.

**What's differentiated (protect these)**:
1. Hard separation of evidence vs inference (Episodes vs Psyche/Entity)
2. Temporal grounding + provenance as invariants
3. Graph as *state* (not just storage) enabling attractor-like retrieval
4. Memetic framing: self-organizing connectivity within fixed pillars
5. Tool-native interface that can scale to RLM-style recursive compute

### 12.2 The Consolidation Answer

Adopt a TiMem/EverMemOS-like temporal hierarchy INSIDE Episodes without breaking the 4 pillars:

```
Raw Messages → MemCells (atomic chunks) → MemScenes (thematic clusters) → Episodes (canonical evidence)
```

Psyche consolidation should be **conflict-aware reconsolidation** — track competing beliefs/preferences with timestamps + supporting episode pointers. Not "latest wins" but "evidence-weighted evolution."

Entities should be **provenance-first upserts** — attributes linked to source evidence.

Notes gain **validity windows** (EverMemOS-style foresight) for temporal awareness.

### 12.3 The Retrieval Answer

Personal memory retrieval needs *temporal + relational reconstruction*, not just top-k similarity. Most questions are "when/after/before/around/why did I" and "what's true about X given conflicts."

**Recommended architecture:**
1. **Fast mode**: BM25 + vector + recency prior, fused with RRF → returns compact candidate set
2. **Agentic mode**: Multi-round query refinement + time anchoring + link expansion (expand_neighbors/follow_relationship) until confidence/coverage stop condition
3. **Time anchoring**: Resolve relative references ("yesterday," "last week") BEFORE retrieval
4. **Link expansion as default**: expand_neighbors/follow_relationship become natural moves in the agent loop, not exotic last-resort tools

This directly targets the deterministic failures: "yesterday → 0 results" is a time-anchoring failure, not a missing embedding.

### 12.4 The RL Answer

RL-trained memory management is very likely the medium-term winner for selection and updates. But the right 6-month move is to **build the policy surface and data**, not jump to full RL.

**Phase 1 (now)**: Heuristic utility scores (access count, recency, link degree) + comprehensive logging
**Phase 2 (3 months)**: Two-phase retrieval (semantic → value-aware) with tunable λ
**Phase 3 (6 months)**: Offline RL/GRPO training on 10K+ retrieval traces

Prompt engineering alone won't reliably learn to ignore distractors or manage conflicts under distribution shift.

### 12.5 The RLM Answer

Persona becomes MORE important in an RLM world, not less. RLMs increase test-time compute and recursive self-calls, amplifying the need for a persistent, queryable, provenance-rich substrate to recurse over. Persona IS that substrate.

RLM replaces the "single-pass agent loop" assumption, not the need for memory.

### 12.6 The Bitter Lesson Hedge

Keep pillars as the *contract* (immutability of evidence, evidence/inference separation, temporal grounding, provenance, versioned schemas). Make everything else learned/offline-optimized (retrieval fusion weights, link creation/strengthening, consolidation triggers, what to summarize, what to discard).

If learned architectures dominate in 3–5 years, Persona's survival path is being the **stable substrate those learned policies operate over**, plus an offline evolution loop that searches policies/graphs within constraints rather than redesigning the whole ontology.

### 12.7 The Five Highest-Leverage Changes (Next 6 Months)

**Change 1 — Remove the 1-Tool Budget and Ship Multi-Round Retrieval**
Highest-leverage single fix. Directly addresses "links invisible" and query formulation failures. Add explicit stop criteria (confidence threshold, max rounds, coverage metric).

**Change 2 — Ship Fast + Agentic Retrieval with Time Anchoring**
Fast mode: BM25 + vector + recency + RRF. Agentic mode: query expansion + link expansion + temporal chaining. Deterministic time resolution ("yesterday" → actual date) as a preprocessing step.

**Change 3 — Build Consolidation v1**
MemCells/Scenes/Episodes hierarchy within the Episode pillar. Conflict-aware Psyche reconsolidation. Entity provenance pointers. Validity windows for Notes/Foresight.

**Change 4 — Make Links a Retrieval Substrate**
LED_TO/CAUSED_BY/MENTIONED_IN become retrieval primitives. Add lightweight hub structures (scene/episode anchors) to reduce traversal noise. SYNAPSE-style spreading activation as the retrieval mechanism.

**Change 5 — Add Memory Policy Layer + Instrumentation**
Log candidate sets, chosen memories, distractors, tool outcomes, and eval labels. Build the data infrastructure for RL. Start with heuristic utility scores. Graduate to learned policies.

---

## 13. The 10-Year View

### 13.1 What 2036 Looks Like

Personal memory in 2036 will be a **continuously reconciling, multi-agent, multi-device "life substrate"** with shared state across assistants, modalities, and environments. Models will do heavy test-time compute, but they'll still need durable external state for identity continuity, auditability, and control.

### 13.2 Invariants (Things That Won't Change)

These hold regardless of how capable base models become:

1. **Immutable evidence store** — what actually happened must be preserved unchanged
2. **Evidence vs inference separation** — derived beliefs must be distinguishable from source observations
3. **Temporal grounding** — every piece of information must be anchored in time
4. **Provenance/audit trails** — every belief must trace back to its source
5. **Selective retention/forgetting** — with user-governed policies, not opaque heuristics
6. **Conflict representation** — not forced single truth; competing beliefs coexist with evidence weights
7. **World-model index layer** — small, navigable, versioned; the "table of contents" for a life

### 13.3 What WILL Change

- Retrieval becomes fully learned (RL-trained policies)
- Consolidation becomes continuous and automatic (not scheduled/manual)
- Graph structure evolves via offline optimization (ALMA/MemEvolve-style)
- Memory operations become multi-modal (text + image + audio + sensor)
- Multiple agents share and coordinate through the same memory substrate
- Privacy and sovereignty become the primary differentiator, not capability

### 13.4 Persona's Position in 2036

If Persona executes correctly, it becomes the **cognitive infrastructure layer** — not the AI that talks to you, but the persistent identity substrate that any AI can build on. The "operating system for personal memory" where models come and go but your identity graph endures.

The memetic organism vision (Section 10.4) is the right framing: a living system that evolves with the user, maintains identity continuity, and can be introspected. Not a database. Not a chatbot add-on. A digital extension of the self.

---

## Appendix A: Paper Catalog

### Memory Systems

| System | ArXiv/Venue | Date | Key Contribution |
|--------|------------|------|-----------------|
| EverMemOS | 2601.02163 | Jan 2026 | Self-organizing memory OS, 93% LoCoMo |
| SimpleMem | 2601.02553 | Jan 2026 | Semantic compression, 26.4% F1↑, 30x token reduction |
| TiMem | 2601.02845 | Jan 2026 | Temporal Memory Tree, 75.3% LoCoMo |
| HiMem | 2601.06377 | Jan 2026 | Hierarchical Episode+Note, conflict reconsolidation |
| CMA | 2601.09913 | Jan 2026 | 5 architectural requirements for agent memory |
| AgeMem v2 | 2601.01885 | Jan 2026 | Unified LTM/STM, tool-based RL |
| SYNAPSE | 2601.02744 | Jan 2026 | Spreading activation, episodic-semantic fusion |
| E-mem | 2601.21714 | Jan 2026 | Multi-agent episodic context reconstruction |
| CAST | 2602.06051 | Jan 2026 | Dramatic theory, character-and-scene memory |
| Self-Consolidation | 2602.01966 | Feb 2026 | Learnable parameters from textual experience |
| Memory-R1 | 2508.19828v5 | Jan 2026 | RL for memory management, 28% F1↑ |
| Mem-alpha | 2509.25911 | Sep 2025 | RL-driven memory construction as MDP |
| MemEvolve | 2512.18746 | Dec 2025 | Meta-evolution of memory architectures |
| AssoMem | 2510.10397 | Oct 2025 | Multi-signal associative retrieval (Meta) |
| MEM1 | ICLR 2026 | 2026 | Synthesizing memory and reasoning |
| MemGen | 2509.24704 | ICLR 2026 | Generative latent memory |
| ALMA | 2602.07755 | Feb 2026 | Meta-learning memory designs (Clune lab) |
| A-MEM | — | 2025 | Zettelkasten-inspired atomic notes |
| Mem0 | — | 2025 | Production memory-as-a-service |
| MeMo | 2502.12851 | Feb 2025 | Memorization precedes learning |
| CAMELoT | 2402.13449 | Feb 2024 | Training-free associative memory |
| MemVerse | 2512.03627 | Dec 2025 | Multimodal lifelong learning |
| Memvid V2 | memvid.com | Jan 2026 | Portable, deterministic memory |

### RLM / Compute Paradigm

| Paper | ArXiv | Date | Key Contribution |
|-------|-------|------|-----------------|
| RLM | 2512.24601 | Dec 2025 | Recursive Language Models (MIT CSAIL) |
| RLM blog | primeintellect.ai | Jan 2026 | "The paradigm of 2026" |

### Benchmarks

| Benchmark | ArXiv/Venue | Date | Focus |
|-----------|------------|------|-------|
| LoCoMo | 2402.17753 | Feb 2024 | Long-term conversational (10 convos) |
| PersonaMem | — | 2025 | Personal facts and preferences (32 users) |
| BEAM+LIGHT | ICLR 2026 | 2026 | 100K-10M token conversations |
| MEMORYBENCH | 2510.17281 | ICLR 2026 | Memory + continual learning |
| MemoryAgentBench | 2507.05257 | ICLR 2026 | 4 core competencies |
| MemBench | ACL 2025 | 2025 | Comprehensive multi-dimensional |

### Surveys

| Survey | ArXiv | Date | Scope |
|--------|-------|------|-------|
| Memory in the Age of AI Agents | 2512.13564 | Dec 2025 | Comprehensive (1,162★ tracker) |
| Evolution of LLM Agent Memory | preprints202601.0618 | Jan 2026 | From storage to experience evolution |
| Rethinking Memory in LLM Agents | 2505.00675v3 | Dec 2025 | Representations, operations, topics |
| Self-Evolving Agents | 2507.21046v4 | Jul 2025 | Self-evolution in agent systems |

### Safety & Stability

| Paper | ArXiv | Date | Focus |
|-------|-------|------|-------|
| Agent Drift | 2601.04170 | Jan 2026 | Behavioral degradation over time |
| Episodic Memory Safety | LessWrong | Feb 2026 | Novel safety risks from memory |

---

## Appendix B: Benchmark Comparison

| Dimension | PersonaMem | LoCoMo | BEAM+LIGHT | MemoryAgentBench |
|-----------|-----------|--------|------------|------------------|
| **Scale** | 32 users, 240K tok | 10 convos, 270K tok | 100K-10M tok | Variable |
| **Sessions** | 5-10 per user | Up to 35 per convo | Long continuous | Multi-turn |
| **Question Types** | MCQ (personal facts) | QA, summarization, dialogue | Memory probes | 4 competencies |
| **Temporal** | Some | Strong | Strong | Moderate |
| **Conflict** | Limited | Limited | Unknown | Explicit (CR) |
| **Multi-hop** | Some | Strong (91% EverMemOS) | Unknown | Long-range (LRU) |
| **Format** | MCQ with options | Free-form + evaluation | Probes | Mixed |
| **Best Score** | 65% (Persona) | 93% (EverMemOS) | Baselines +12.69% | — |

---

## Appendix C: Architecture Comparison Matrix

| Feature | Persona | EverMemOS | SimpleMem | TiMem | HiMem | A-MEM | Mem0 |
|---------|---------|-----------|-----------|-------|-------|-------|------|
| **Memory Types** | 4 pillars (Ep/Ps/En/No) | MemCell/Scene/Episode | Compressed units | TMT hierarchy | Episode + Note | Zettelkasten nodes | Key-value + graph |
| **Storage** | Neo4j + vectors | MongoDB + Milvus + ES + Redis | Vector + compressed | Tree structure | Hierarchical | Linked notes | Vector + optional graph |
| **Consolidation** | Manual memeplex | Auto MemScene clustering | Online semantic synthesis | TMT consolidation | Conflict-aware reconsolidation | Static | Scheduled |
| **Retrieval** | Single-pass recall | Fast (RRF) + Agentic (multi-round) | Intent-aware planning | Complexity-aware | Hybrid best-effort | Link traversal | Vector similarity |
| **Temporal** | Timestamps | Foresight + validity windows | Indexed | TMT tree | Event-surprise channels | None | Timestamps |
| **Links** | LED_TO, CAUSED_BY (unused) | Implicit via MemScenes | Semantic connections | Tree edges | Topic-aware | Zettelkasten links | Optional |
| **RL** | None | None | None | None | None | None | None |
| **World Model** | Memeplex | Implicit in MemScenes | Compressed views | TMT root | Note memory | None | None |
| **Eval** | 65% PersonaMem | 93% LoCoMo | 26.4% F1↑ LoCoMo | 75.3% LoCoMo | Outperforms baselines | — | 69% LoCoMo |

---

*End of document. Last updated: February 10, 2026.*
*Sources: arxiv.org, iclr.cc, openreview.net, GitHub, Prime Intellect blog, EverMemOS repo, founder research notes, Oracle consultation.*
