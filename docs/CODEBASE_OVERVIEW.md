# Codebase Overview

This document provides a high-level orientation for contributors.

## Architecture

```
persona/           # Core library
├── adapters/      # High-level orchestrators
├── core/          # Database operations and retrieval
├── llm/           # LLM providers and functions
├── models/        # Data models
└── services/      # Business logic

server/            # FastAPI application
tests/             # Test suite
docs/              # Documentation
```

## Core Concepts

### Memory Model

Persona stores user data as three typed memory classes:

| Type | Purpose | Example |
|------|---------|---------|
| **Episode** | What happened | "Had coffee with Sam to discuss his startup" |
| **Psyche** | Who they are | "Prefers remote work" |
| **Note** | Structured info | Goals, tasks, facts, contacts, reminders |

All memories are stored in Neo4j with embeddings for vector similarity search.

**Note Memory Types** (`note_type` field):
- `goal` - Objectives and targets
- `task` - Action items
- `fact` - Stored facts about the user
- `contact` - People in the user's network
- `reminder` - Time-based reminders
- `list` - Collections (favorites, preferences)

### Key Components

1. **PersonaAdapter** (`adapters/persona_adapter.py`)
   - Unified entry point for ingestion
   - Orchestrates: extraction → linking → persistence

2. **Retriever** (`core/retrieval.py`)
   - Combines vector similarity with graph traversal
   - LLM-enhanced query expansion for temporal/entity parsing
   - Returns formatted context for LLM consumption

3. **QueryExpansion** (`core/query_expansion.py`)
   - Parses natural language queries into structured hints
   - Extracts date ranges ("last week" → date filter)
   - Identifies entities and relationship threads
   - Falls back to rule-based parsing when LLM fails

4. **MemoryStore** (`core/memory_store.py`)
   - CRUD operations for typed memories
   - Handles temporal linking between episodes

5. **ContextFormatter** (`core/context.py`)
   - Transforms memories into XML context
   - Groups by type for LLM readability
   - Supports multiple context views (profile, timeline, tasks, graph)
   - Research-based ordering: UserCard first (primacy), Episodes last (recency)

## Data Flow

### Ingestion
```
Raw Text → PersonaAdapter → MemoryIngestionService → MemoryStore → Neo4j
                              ↓
                         LLM extracts:
                         - Episode (what happened)
                         - Psyche (traits/preferences)
                         - Note (goals/tasks/facts)
                         
                         Provenance tracked:
                         - session_id (source conversation)
                         - extraction_model (which LLM)
```

### Retrieval
```
Query → QueryExpansion → Retriever → Vector Search + Graph Traversal → ContextFormatter → LLM
             ↓                                                               ↓
        Extracts:                                                    <memory_context>
        - date_range                                                   <episodes>...</episodes>
        - entities                                                     <psyche>...</psyche>
        - semantic_query                                               <notes>...</notes>
                                                                     </memory_context>
```

## Dependency Injection

The application uses FastAPI's dependency injection:

```python
@router.post("/users/{user_id}/rag/query")
async def rag_query(
    user_id: str,
    query: RAGQuery,
    graph_ops: GraphOps = Depends(get_graph_ops)
):
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

## Session & Episode Management

An **Episode** in Persona represents a distinct block of interaction or event. The definition of an episode is flexible and depends on the **UI and integration context** of the agent using Persona.

Factors influencing episode boundaries:
- **Time Blocks**: Dividing a user's day into morning/afternoon/evening sessions.
- **Auth Sessions**: Ingesting data per login session.
- **UI Interaction**: Triggering a new episode on "New Chat", "Reload", or "Clear Context".
- **Token Limits**: Chunking long histories to manage LLM context windows.
- **Multiple Sources**: Handling parallel streams from different platforms (Slack, Email, Chat).

The system is designed to handle both **Single-Session** (one focused chat) and **Multi-Session** (long-term historical) retrieval patterns.

## Customization & Extensibility

Persona is built to be highly customizable via the **PersonaAdapter**.

- **Custom Schemas**: Define your own extraction logic to focus on specific domains (e.g., medical, financial, technical).
- **Extraction Rules**: Modify how text is parsed into Episode, Psyche, and Note types.
- **Cross-Linking**: Configure how memories are linked across different sessions or sources.
- **Memory Priority**: Customize weighted retrieval for specific use cases.

The `PersonaAdapter` acts as the primary interface for these customizations, allowing you to tailor the "memetic digital organism" to your specific application needs.

## Retrieval Layer

The retrieval layer (`core/retrieval.py` + `core/query_expansion.py`) implements intelligent context fetching.

### Query Expansion

Before vector search, queries are expanded using LLM-enhanced parsing:

```python
from persona.core.query_expansion import expand_query

expansion = await expand_query("What did I eat last week?", user_timezone="America/Los_Angeles")
# Returns:
# QueryExpansion(
#     original_query="What did I eat last week?",
#     date_range=DateRange(start=date(2025, 12, 19), end=date(2025, 12, 26)),
#     entities=[],
#     semantic_query="What did I eat"
# )
```

**Supported temporal patterns** (rule-based fallback):
- "yesterday" → single day
- "last week" / "past week" → 7-day window
- "last month" / "past month" → 30-day window
- "today" → current day

### Retriever Pipeline

```python
from persona.core.retrieval import Retriever

retriever = Retriever(user_id, store, graph_ops)
context = await retriever.get_context(
    query="What happened last week?",
    top_k=5,              # Vector search results
    hop_depth=1,          # Graph traversal depth
    include_static=True,  # Include active notes + psyche
    user_timezone="UTC",  # For temporal parsing
)
```

**Pipeline stages**:
1. **Query Expansion** - Parse temporal refs, extract entities
2. **Static Context** - Always include active notes + core psyche
3. **Vector Search** - Semantic similarity with optional date filtering
4. **Graph Crawl** - Follow relationships from seed nodes
5. **Format** - XML context for LLM consumption

### Provenance Tracking

Every extracted memory includes source tracking:

| Field | Purpose |
|-------|---------|
| `session_id` | Which conversation it came from |
| `extraction_model` | LLM used (e.g., "gpt-4o-mini") |
| `extraction_confidence` | Future: confidence score |

## Context Engineering

Research-backed context formatting based on "Lost in the Middle" (Stanford) and StructRAG findings.

### UserCard

A compact identity anchor placed at the start of context (primacy position):

```python
from persona.models.memory import UserCard

card = UserCard(
    user_id="user_123",
    name="Alex",
    roles=["software engineer", "parent"],
    core_values=["work-life balance", "continuous learning"],
    current_focus=["career transition", "health goals"],
)
```

### Context Views

Query-adaptive structure selection via `ContextView` enum:

| View | Use Case | Ordering |
|------|----------|----------|
| `PROFILE` | Identity questions ("who am I?") | Psyche → Notes → Episodes |
| `TIMELINE` | Temporal queries ("last week") | Chronological episodes |
| `TASKS` | Action queries ("my tasks") | Active notes first |
| `GRAPH_NEIGHBORHOOD` | Entity queries | Entity-linked memories |

```python
from persona.core.context import ContextView

# View is auto-routed based on query
context = await retriever.get_context(
    query="What happened last week?",  # → TIMELINE view
    user_timezone="America/Los_Angeles"
)
```

### Memory Importance

All memories have an `importance` field (0.0-1.0) used for ordering within context:

```python
from persona.models.memory import Episode

episode = Episode(
    content="Got promoted to senior engineer",
    importance=0.9,  # High importance, appears earlier in context
    ...
)
```

### Link Scoring

Graph traversal scores linked memories based on:
- Base importance field
- Entity matches from QueryExpansion (+0.2 per match)
- Recency bonus for episodes (<7 days: +0.3, <30 days: +0.1)

## Further Reading

- [API Reference](API.md)
- [LLM Clients Implementation](LLM_CLIENTS_IMPLEMENTATION.md)
- [Development Guide](DEVELOPMENT.md)
