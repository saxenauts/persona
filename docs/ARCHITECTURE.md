# Persona Architecture

## Design Principles

- **LLM-first**: no manual routing or keyword gates in the query path; tool choice stays model-driven.
- **Minimal primitives**: four memory types only (Episode, Psyche, Entity, Note).
- **Chronology-first**: ingestion captures `event_time` and preserves temporal order.
- **Working memory**: a compact, human-analog context (UserCard + recent episodes + active psyche/notes).
- **Future-ready**: quantification (frequency/recency/saliency) and async consolidation ("dreaming") are planned upgrades, not required for v1.

## System Overview

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                    FastAPI Server                        │
                              │                   (server/main.py)                       │
                              └────────────────────────┬────────────────────────────────┘
                                                       │
                    ┌──────────────────────────────────┼──────────────────────────────────┐
                    │                                  │                                  │
                    ▼                                  ▼                                  ▼
         ┌──────────────────┐              ┌──────────────────┐              ┌──────────────────┐
         │   /ingest API    │              │    /chat API     │              │ /memeplex API    │
         │  (graph_api.py)  │              │  (graph_api.py)  │              │ (graph_api.py)   │
         └────────┬─────────┘              └────────┬─────────┘              └────────┬─────────┘
                  │                                 │                                  │
                  ▼                                 ▼                                  ▼
         ┌──────────────────┐              ┌──────────────────┐              ┌──────────────────┐
         │ PersonaAdapter   │              │  PersonaService  │              │ MemoryStore      │
         │ (ingest entry)   │              │  (query entry)   │              │ (memeplex ops)   │
         └────────┬─────────┘              └────────┬─────────┘              └──────────────────┘
                  │                                 │
                  │                                 │
      ┌───────────┼───────────┐        ┌───────────┴───────────┐
      │           │           │        │                       │
      ▼           ▼           ▼        ▼                       ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────┐
│Ingestion │ │MemoryStore│ │Integration│ │  AgentRunner │ │  UserCard +  │
│ Service  │ │ .create() │ │  Agent   │ │  (run loop)  │ │   Memeplex   │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘ └──────────────┘
     │            │            │              │
     │            │            │              │
     ▼            ▼            ▼              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           Tool Handlers                                 │
│  recall() | browse() | record() | update_memory() | expand_neighbors() │
│                    (persona/tools/memory.py)                           │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                            MemoryStore                                  │
│           (persona/core/memory_store.py - unified CRUD)                │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌────────────────┐            ┌────────────────┐
     │   GraphDB      │            │  VectorStore   │
     │   (Neo4j)      │            │  (Neo4j HNSW)  │
     └────────────────┘            └────────────────┘
```

---

## Data Flow: Ingestion Path

```
Raw Content (conversation, note, etc.)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. PersonaAdapter.ingest()                                  │
│    - Entry point for all data ingestion                     │
│    - Optional: store raw transcript as Episode              │
│    - Sessions are product-defined (idle, context, close)    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. IngestionService.ingest()                                │
│    - LLM extracts 4-pillar memories (Episode, Psyche,       │
│      Entity, Note) with structured output                   │
│    - Generates embeddings for each memory                   │
│    - Episode is narrative; if sequence is explicit, it may  │
│      include a compact "Ordered mentions" line (optional)   │
│    - Returns IngestionResult with memories + links          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. MemoryStore.create() + .create_link()                    │
│    - Persists memories to Neo4j graph                       │
│    - Stores embeddings in Neo4j vector index                │
│    - Creates temporal NEXT/PREVIOUS episode chains          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Integration Agent (async, background)                    │
│    - Finds unintegrated memories                            │
│    - Creates semantic links: MENTIONS, LED_TO, CAUSED_BY    │
│    - Entity resolution: merge duplicates via SAME_AS        │
│    - Marks memories as integrated                           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Consolidation (triggered by integration)                 │
│    - Refreshes UserCard (identity prose)                    │
│    - Refreshes Memeplex (world model index)                 │
│    - Stores as JSON blobs in Neo4j                          │
│    - Future: async consolidation ("dreaming")               │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Query Path

```
User Query ("What did I do last week?")
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. PersonaService.run_agent()                               │
│    - Loads UserCard + Memeplex for system prompt            │
│    - Builds ToolContext with user_id, store, timezone       │
│    - Injects {world_model}, {user_context}, {today_date}    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. AgentRunner.run() (tool-calling loop)                    │
│    - LLM decides which tools to call                        │
│    - Tools: recall, browse, expand_neighbors, follow_rel    │
│    - Loop until LLM produces final answer                   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Tool Execution (recall, browse, etc.)                    │
│    - recall(): vector similarity search + filters           │
│    - browse(): time-ordered listing                         │
│    - expand_neighbors(): graph traversal                    │
│    - Returns MemoryHit[] with snippets, timestamps          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Final Answer                                             │
│    - LLM synthesizes answer from tool results               │
│    - Optional: structured output via output_schema          │
└─────────────────────────────────────────────────────────────┘
```

---

## Working Memory Composition

Persona injects a compact, human-analog working memory into the system prompt:

- `<user>`: UserCard identity prose (stable anchor)
- `<recent_context>`: recent Episodes (chronological narrative)
- `<active_context>`: active Psyche + Notes (preferences + commitments)

The Memeplex is injected separately as `{world_model}` (table-of-contents for the user's world).

---

## 4-Pillar Memory Model

| Pillar | Class | Cognitive Function | Update Semantics | Storage |
|--------|-------|-------------------|------------------|---------|
| **Episode** | `EpisodeMemory` | What happened | Append-only | Neo4j node `:Episode` |
| **Psyche** | `PsycheMemory` | Who they are | Consolidate/evolve | Neo4j node `:Psyche` |
| **Entity** | `EntityMemory` | What/who exists | Upsert with attributes | Neo4j node `:Entity` |
| **Note** | `NoteMemory` | What to do | State machine (active→done) | Neo4j node `:Note` |

### Key Distinction

**Entity vs Note (CRITICAL):**
- **Entity** = Things that EXIST (nouns with attributes): "Sarah (birthday: June 5)"
- **Note** = Things to DO (intentions with status): "Call Sarah tomorrow"
- Facts about entities are **Entity attributes**, not Notes

---

## Key Components

### Adapters Layer
**PersonaAdapter** (`persona/adapters/persona_adapter.py`)
- Single entry point for ALL ingestion
- Handles: conversations, Apple Notes, Twitter, Instagram
- Orchestrates: extraction → persistence → integration → consolidation

### Services Layer

**IngestionService** (`persona/services/ingestion_service.py`)
- LLM-based extraction of 4-pillar memories
- Structured output parsing (Episode, Psyche, Entity, Note)
- Embedding generation for vector search

**IntegrationAgent** (`persona/services/integration_agent.py`)
- Background agent that runs after ingestion
- Creates semantic links: MENTIONS, LED_TO, CAUSED_BY, NEXT/PREVIOUS
- Entity resolution: detects and merges duplicates

**ConsolidationService** (`persona/services/consolidation_service.py`)
- Generates UserCard (identity prose summary)
- Refreshes Memeplex (world model index)
- Triggered after integration completes

**PersonaService** (`persona/services/persona_service.py`)
- Main query entry point
- Loads UserCard + Memeplex into prompt
- Runs agent loop with memory tools

### Tools Layer

**Static Registry** (`persona/tools/runner.py`)
- Global `REGISTRY` maps tool names → handlers
- Per-request context injection via `ToolContext`
- Bounded execution with semaphore + timeouts

**Memory Tools** (`persona/tools/memory.py`):

| Tool | Purpose | Returns |
|------|---------|---------|
| `recall(query, filters)` | Semantic vector search | MemoryHit[] |
| `browse(date_range, types)` | Time-ordered listing | MemoryHit[] |
| `get_memory(id)` | Full memory with metadata | Memory |
| `expand_neighbors(id)` | Graph traversal | MemoryHit[] |
| `follow_relationship(id, type)` | Trace relationship chains | MemoryHit[] |
| `record(text)` | Ingest new information | IngestionResult |
| `update_memory(id, updates)` | Modify existing memory | Memory |

### Core Layer

**MemoryStore** (`persona/core/memory_store.py`)
- Unified CRUD for all memory types
- Vector similarity search via Neo4j HNSW
- Link management (create, query)

**GraphOps** (`persona/core/graph_ops.py`)
- Low-level Neo4j operations
- Connection pooling, transaction management

---

## Context Engineering

**UserCard** (`persona/models/memory.py`)
- Compact identity anchor (3-8 sentences)
- Placed first in context (primacy effect)
- Generated by consolidation from recent memories

**Memeplex** (`persona/models/memory.py`)
- World model index: topics, people, projects, places, concepts
- Provides LLM "table of contents" for user's memory
- Includes recency signals: last_week_topics, last_month_topics

**System Prompt Structure** (`persona/llm/prompts.py`):
```
<clock> today_date, user_timezone
<world_model> Memeplex (topics, people, projects)
<user_context> UserCard identity prose
<tools> Memory tool descriptions
<retrieval_policy> When to use which tool
<answering_rules> Evidence requirements
```

---

## Relationship Types

| Type | Semantics | Created By |
|------|-----------|------------|
| `MENTIONS` | Episode/Note references Entity | Integration agent |
| `LED_TO` | Causal forward: A caused B | Integration agent |
| `CAUSED_BY` | Causal backward: B from A | Integration agent |
| `NEXT` / `PREVIOUS` | Temporal sequence | Ingestion (episodes) |
| `RELATES_TO` | Thematic association | Integration agent |
| `CONTRADICTS` | Information conflicts | Integration agent |
| `SAME_AS` | Entity deduplication | Integration agent |

---

## Design Principles

### LLM-First (NON-NEGOTIABLE)

**No manual routers. No keyword matching. No heuristic gating.**

All decisions are made by LLMs through prompt engineering:
- Tool selection: LLM decides which tool based on context
- Write vs defer: LLM decides when to `record()` based on prompt
- Retrieval strategy: LLM chooses vector vs graph based on query

**Anti-patterns:**
```python
# WRONG - keyword routing
if "remind" in message:
    enable_record_tool()

# WRONG - intent classification
if has_immediate_write_intent(message):
    call_record()

# RIGHT - expose all tools, let LLM decide
tools = [recall, browse, record, update_memory, ...]
llm.chat(messages, tools=tools)
```

---

## Architectural Drift Analysis (Validated 2026-01-21)

### 1. Retriever Class - TEST-ONLY USAGE ✅
**File**: `persona/core/retrieval.py`
**Status**: ONLY used in tests (`tests/integration/test_retrieval.py`, `tests/core/test_date_retrieval.py`)
**Finding**: Main query path uses `PersonaService` → `AgentRunner` → Tools directly. Retriever is dead code for production.
**Recommendation**: Either wire Retriever into a CLI/batch interface OR park/remove it. Currently adds confusion.

### 2. GraphPatch Duality - TWO SEPARATE SCHEMAS ⚠️
**Files**:
- `persona/tools/integration.py`: `GraphPatch` (tool-side, 4 fields: items, link(), unlink(), mark_integrated())
- `persona/models/graph_patch.py`: `GraphPatch` (model-side, richer: run_id, user_id, links, merges, updates, etc.)

**Finding**: These are **different** schemas serving different purposes:
- `tools/integration.py` GraphPatch: Simple declarative patch for LLM to emit
- `models/graph_patch.py` GraphPatch: Full audit model with IntegrationJob lifecycle

**Risk**: Import confusion (`from persona.tools.integration import GraphPatch` vs `from persona.models.graph_patch import GraphPatch`)
**Recommendation**: Rename tool-side to `GraphPatchPayload` or namespace clearly. Keep both - they serve different layers.

### 3. Memeplex Protocol Layer - PROTOCOLS NOT IMPLEMENTED ✅
**File**: `persona/core/memeplex.py`
**Status**: Contains PROTOCOL DEFINITIONS only (EntityRegistry, TemporalIndex, TopicCluster)
**Finding**: Actual Memeplex is implemented in:
- Model: `persona/models/memory.py` (simple Memeplex class)
- Storage: `persona/core/memory_store.py` (save_memeplex, get_memeplex)
- Refresh: `persona/services/consolidation_service.py` (refresh_memeplex)

The `core/memeplex.py` protocols are FUTURE interface definitions for v2. Not a bug.
**Recommendation**: Add prominent docstring (already done) or rename to `memeplex_protocols.py`

---

## File Index

| Path | Purpose |
|------|---------|
| `server/main.py` | FastAPI app entry, lifespan |
| `server/routers/graph_api.py` | API routes: /ingest, /chat, /memeplex |
| `persona/adapters/persona_adapter.py` | Unified ingestion interface |
| `persona/services/persona_service.py` | Query entry point, agent runner |
| `persona/services/ingestion_service.py` | LLM extraction of memories |
| `persona/services/integration_agent.py` | Background linking agent |
| `persona/services/consolidation_service.py` | UserCard + Memeplex refresh |
| `persona/tools/memory.py` | Tool handlers: recall, browse, record |
| `persona/tools/runner.py` | Agent loop with tool execution |
| `persona/tools/schemas.py` | Tool JSON schemas for LLM |
| `persona/core/memory_store.py` | Unified CRUD for memories |
| `persona/core/graph_ops.py` | Neo4j operations |
| `persona/core/retrieval.py` | Time-windowed retrieval (unused?) |
| `persona/models/memory.py` | Memory types, UserCard, Memeplex |
| `persona/llm/prompts.py` | System prompts |
