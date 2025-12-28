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
   - Time-windowed fetch with vector similarity
   - Returns formatted context for LLM consumption

3. **Tools Layer** (`tools/memory.py`)
   - `recall(query)`: Parses temporal refs, fetches context
   - `record(text)`: Ingests new memories with type classification

4. **MemoryStore** (`core/memory_store.py`)
   - CRUD operations for typed memories
   - Handles temporal linking between episodes

5. **ContextFormatter** (`core/context.py`)
   - Transforms memories into prose context
   - Renders links inline for narrative continuity
   - Sections: `<user>`, `<recent_context>`, `<active_context>`

### Services

Business logic layer between API and core:

1. **MemoryIngestionService** (`services/ingestion_service.py`)
   - Extracts memories from raw text using LLM
   - Creates Episode, Psyche, and Note memories
   - Handles temporal linking between episodes

2. **UserCardService** (`services/user_service.py`)
   - Synthesizes UserCard from Psyche memories + active Notes
   - Uses LLM to generate name, roles, values, current focus
   - Falls back to rule-based extraction on LLM failure
   - Cached per session (lazy generation on first query)

3. **RAGService** (`services/rag_service.py`)
   - Thin wrapper around RAGInterface for API use
   - Handles async context management

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
Query → recall() tool → Retriever → Vector Search + Time Filter → format_working_memory_prose() → LLM
              ↓                                                               ↓
         Parses:                                                    <user>identity</user>
         - temporal refs                                            <recent_context>episodes</recent_context>
         - time windows                                             <active_context>psyche + notes</active_context>
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

The retrieval layer (`core/retrieval.py` + `tools/memory.py`) implements intelligent context fetching.

### Temporal Parsing

The `recall()` tool parses temporal references using rule-based patterns:

```python
from persona.tools.memory import recall

context = await recall("What did I eat last week?", user_id="user_123", timezone="America/Los_Angeles")
# Parses "last week" → 7-day time window, fetches memories, formats as prose
```

**Supported temporal patterns**:
- "yesterday" → single day
- "last week" / "past week" → 7-day window
- "last month" / "past month" → 30-day window
- "today" → current day

### Retriever Pipeline

```python
from persona.core.retrieval import Retriever

retriever = Retriever(user_id, store, graph_ops)
context = await retriever.get_working_memory(
    query="What happened last week?",
    config=RetrievalConfig(
        time_window_days=7,
        max_episodes=10,
    ),
    user_card=user_card,
)
```

**Pipeline stages**:
1. **Static Context** - Always include active notes + core psyche
2. **Vector Search** - Semantic similarity with time filtering
3. **Format** - Prose context with link annotations

### Provenance Tracking

Every extracted memory includes source tracking:

| Field | Purpose |
|-------|---------|
| `session_id` | Which conversation it came from |
| `extraction_model` | LLM used (e.g., "gpt-4o-mini") |
| `extraction_confidence` | Future: confidence score |

## Context Engineering

Prose-based context formatting for natural LLM consumption.

### UserCard

A compact identity anchor placed at the start of context (primacy position):

```python
from persona.models.memory import UserCard

card = UserCard(
    user_id="user_123",
    timezone="America/Los_Angeles",
    identity_prose="Alex is a software engineer and parent who values work-life balance and continuous learning. Currently focused on career transition and health goals.",
)
```

### Prose Format

Context is rendered as natural language with semantic sections:

```python
from persona.core.context import format_working_memory_prose

context = format_working_memory_prose(
    user_card=card,
    episodes=episodes,
    psyche=psyche_memories,
    active_notes=notes,
    links=memory_links,
)

# Output:
# <user>
# Alex is a software engineer and parent...
# </user>
#
# <recent_context>
# December 27: Had a great meeting with team (led to project kickoff)
# December 20: Started the new project
# </recent_context>
#
# <active_context>
# Trait: Values efficiency. Preference: Morning meetings.
# Current tasks: Launch MVP. Review documentation.
# </active_context>
```

### Link Prose

Memory links are rendered inline for narrative continuity:
- "led to X" - causal relationships
- "caused by Y" - reverse causation
- "related to Z" - associations

## Further Reading

- [API Reference](API.md)
- [LLM Clients Implementation](LLM_CLIENTS_IMPLEMENTATION.md)
- [Development Guide](DEVELOPMENT.md)
