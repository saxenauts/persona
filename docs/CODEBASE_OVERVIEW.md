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
| **Goal** | What they want | "Finish Q4 roadmap by Friday" |

All memories are stored in Neo4j with embeddings for vector similarity search.

### Key Components

1. **PersonaAdapter** (`adapters/persona_adapter.py`)
   - Unified entry point for ingestion
   - Orchestrates: extraction → linking → persistence

2. **Retriever** (`core/retrieval.py`)
   - Combines vector similarity with graph traversal
   - Returns formatted context for LLM consumption

3. **MemoryStore** (`core/memory_store.py`)
   - CRUD operations for typed memories
   - Handles temporal linking between episodes

4. **ContextFormatter** (`core/context.py`)
   - Transforms memories into XML context
   - Groups by type for LLM readability

## Data Flow

### Ingestion
```
Raw Text → PersonaAdapter → MemoryIngestionService → MemoryStore → Neo4j
                              ↓
                         LLM extracts:
                         - Episode (what happened)
                         - Psyche (traits/preferences)
                         - Goals (tasks/todos)
```

### Retrieval
```
Query → Retriever → Vector Search + Graph Traversal → ContextFormatter → LLM
                                                            ↓
                                                    <memory_context>
                                                      <episodes>...</episodes>
                                                      <psyche>...</psyche>
                                                      <goals>...</goals>
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
- **Extraction Rules**: Modify how text is parsed into Episode, Psyche, and Goal types.
- **Cross-Linking**: Configure how memories are linked across different sessions or sources.
- **Memory Priority**: Customize weighted retrieval for specific use cases.

The `PersonaAdapter` acts as the primary interface for these customizations, allowing you to tailor the "memetic digital organism" to your specific application needs.

## Further Reading

- [API Reference](API.md)
- [LLM Clients Implementation](LLM_CLIENTS_IMPLEMENTATION.md)
- [Development Guide](DEVELOPMENT.md)
