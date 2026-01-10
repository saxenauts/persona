# Codebase Overview

This document provides a high-level orientation for contributors.

> **For the full architecture vision and philosophy, see [ARCHITECTURE.md](architecture/ARCHITECTURE.md)**

## Project Structure

```
persona/           # Core library
├── adapters/      # High-level orchestrators
├── core/          # Database operations and retrieval
├── llm/           # LLM providers, prompts, embeddings
├── models/        # Data models
├── services/      # Business logic (ingestion, persona)
└── tools/         # Agent tools (recall, record, expand, follow)

server/            # FastAPI application
tests/             # Test suite
docs/              # Documentation
```

## Core Concepts

### 4-Pillar Memory Model

| Pillar | Purpose | Example |
|--------|---------|---------|
| **Episode** | What happened | "Had coffee with Sam to discuss his startup" |
| **Psyche** | Who they are | "Prefers remote work", "Values efficiency" |
| **Entity** | What/who exists | People, places, orgs, projects, concepts |
| **Note** | What to do | Goals, tasks, reminders (intention-triggered only) |

All memories are stored in Neo4j with embeddings for vector similarity search.

**Entity vs Note (CRITICAL):**
- Entity = Things that EXIST (nouns): "Sarah", "Paris", "Project Alpha"
- Note = Things to DO (intentions): "call Sarah", "book trip to Paris"
- Facts about entities (e.g., "Sarah's birthday is June 5") are **Entity attributes**, not Notes

### Key Components

1. **PersonaAdapter** (`adapters/persona_adapter.py`)
   - Unified entry point for ingestion
   - Orchestrates: extraction → linking → persistence

2. **Retriever** (`core/retrieval.py`)
   - Time-windowed fetch + link expansion
   - Returns prose-formatted context for LLM consumption

3. **Tools Layer** (`tools/memory.py`)
   - `recall(query)`: Search memories with structured filters
   - `record(text)`: Ingest new memories with type classification
   - `expand_neighbors(memory_id)`: Graph expansion from a memory node
   - `follow_relationship(source_id, relation_type)`: Trace relationship chains

4. **MemoryStore** (`core/memory_store.py`)
   - CRUD operations for typed memories
   - Handles temporal linking between episodes
   - Batch operations: `get_memories_by_ids()`, `get_nodes_by_ids()`

5. **ContextFormatter** (`core/context.py`)
   - Transforms memories into prose context
   - Renders links inline for narrative continuity
   - Sections: `<user>`, `<recent_context>`, `<active_context>`

6. **Memeplex** (`models/memory.py`)
   - Per-user "world model index" - what the LLM knows about this user
   - Topics (universal), people, projects, places, concepts
   - Time windows: last_week_topics, last_month_topics
   - Stored as JSON blob, injected into system prompt as `{world_model}`

### Services

Business logic layer between API and core:

1. **PersonaService** (`services/persona_service.py`) - **PRIMARY ENTRY POINT**
   - Unified orchestrator for memory-augmented dialogue
   - `run_agent()`: Agent loop with recall/record/expand/follow tools
   - `ask()`: Structured JSON extraction via agent loop
   - Accepts `graph_ops` via constructor (no duplicate connections)

2. **MemoryIngestionService** (`services/ingestion_service.py`)
   - Extracts memories from raw text using LLM
   - Creates Episode, Psyche, Entity, and Note memories
   - Handles temporal linking between episodes

3. **ConsolidationService** (`services/consolidation_service.py`)
   - Synthesizes UserCard identity prose + Memeplex refresh
   - Cached in graph with TTL (lazy generation on first query)

## Data Flow

### Ingestion
```
Raw Text → PersonaAdapter → MemoryIngestionService → MemoryStore → Neo4j
                              ↓
                         LLM extracts:
                         - Episode (what happened)
                         - Psyche (traits/preferences)
                         - Entity (people/places/things)
                         - Note (tasks/goals/reminders)
                         
                         Provenance tracked:
                         - session_id (source conversation)
                         - extraction_model (which LLM)
```

### Retrieval (Agent Loop)
```
User Message → /chat endpoint → PersonaService.run_agent()
                                       ↓
                              Agent decides which tools to call:
                              - recall(query) → vector search + time filter
                              - expand_neighbors(id) → graph traversal
                              - follow_relationship(id, type) → chain tracing
                              - record(text) → save new memories
                                       ↓
                              Format context → LLM generates response
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/users/{user_id}/chat` | POST | **Primary** - Personal AI conversation |
| `/users/{user_id}/persona/ask` | POST | Structured JSON extraction |
| `/users/{user_id}/ingest` | POST | Ingest single content |
| `/users/{user_id}/ingest/batch` | POST | Batch ingest |
| `/users/{user_id}/sessions/{session_id}/close` | POST | Close session + integrate |
| `/users/{user_id}/memeplex` | GET | Read user's world model index |
| `/users/{user_id}/memeplex/refresh` | POST | Force refresh memeplex from memories |
| `/users/{user_id}/memories` | GET | List memories (debug) |
| `/users/{user_id}/memories/stats` | GET | Memory statistics (debug) |
| `/users/{user_id}` | POST | Create user |
| `/users/{user_id}` | DELETE | Delete user |

## Dependency Injection

The application uses FastAPI's dependency injection:

```python
@router.post("/users/{user_id}/chat")
async def chat(
    user_id: str,
    request: ChatRequest,
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    service = PersonaService(graph_ops)
    result = await service.run_agent(user_id=user_id, query=request.messages[-1].content)
    ...
```

A single `GraphOps` instance is created at startup and shared across requests.

## Running the Project

```bash
# Docker (recommended)
docker compose up -d

# Access
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
# Neo4j: http://localhost:7474
```

## Design Philosophy: LLM-First

**No manual routers. No keyword matching. No heuristic gating.**

All decisions are made by LLMs through prompt engineering:
- Tool selection: LLM decides which tool to call based on context
- Write vs defer: LLM decides when to `record()` based on prompt guidance
- Retrieval strategy: LLM chooses vector vs graph based on query semantics

**Anti-patterns (NEVER do these):**
- `if "remind" in message: enable_record_tool()` - NO keyword routing
- `has_immediate_write_intent(message)` - NO intent classifiers
- Manual routing logic before LLM calls - NO preprocessing gates

## Context Engineering

### UserCard

Compact identity anchor placed at the start of context (primacy position):

```python
UserCard(
    user_id="user_123",
    timezone="America/Los_Angeles",
    identity_prose="Alex is a software engineer and parent who values work-life balance...",
)
```

### Prose Format

Context rendered as natural language with semantic sections:

```
<user>
Alex is a software engineer and parent...
</user>

<recent_context>
December 27: Had a great meeting with team (led to project kickoff)
December 20: Started the new project
</recent_context>

<active_context>
Trait: Values efficiency. Preference: Morning meetings.
Current tasks: Launch MVP. Review documentation.
</active_context>
```

## Memeplex: World Model Index

The Memeplex provides the LLM with a "table of contents" for the user's world. Without it, the agent only sees what's retrieved - with it, the agent knows what *exists* and can proactively explore.

**Schema** (`persona/models/memory.py`):
```python
Memeplex(
    user_id: str,
    topics: List[str],           # "fitness", "AI research", "cooking"
    people: List[str],           # "Sarah (wife)", "Max (colleague)"
    projects: List[str],         # "Persona", "Home Renovation"
    places: List[str],           # "SF (home)", "Tokyo (2024 trip)"
    concepts: List[str],         # "stoicism", "minimalism"
    last_week_topics: List[str], # What's been active recently
    last_month_topics: List[str],
    recent_focus: str,           # "Building Memeplex for v1 release"
    memory_stats: MemoryStats,
)
```

**Design Principle**: Topics are universal (everyone has them). Entities (people/projects/places) are optional.

**Injected into system prompt** via `{world_model}` slot in `PERSONAL_AI_SYSTEM_PROMPT`.

**Refresh flow**: Ingestion → Integration → Consolidation → `refresh_memeplex()` → stored in Neo4j.

## Further Reading

- [API Reference](API.md)
- [Memory Model Deep Dive](MEMORY_MODEL.md)
- [Development Guide](DEVELOPMENT.md)
