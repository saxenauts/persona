# Repository Guidelines

## Project Structure

```
persona/           # Core library
├── adapters/      # High-level orchestrators (PersonaAdapter)
├── core/          # Database operations, retrieval, context formatting
├── llm/           # LLM clients, embeddings, prompts
├── models/        # Memory types (memory.py) and API schemas (schema.py)
└── services/      # Business logic (ingestion, RAG, ask)

server/            # FastAPI application
├── main.py        # App entry point with lifespan management
├── routers/       # API route definitions
├── dependencies.py # Dependency injection
└── config.py      # Environment configuration

tests/             # Test suite
├── unit/          # Unit tests (no external deps)
└── integration/   # Integration tests (requires Neo4j)
```

## Build & Development

```bash
# Install dependencies
poetry install

# Run API locally
poetry run uvicorn server.main:app --reload

# Run with Docker (recommended)
docker compose up -d

# Run tests
docker compose run --rm test        # Docker (preferred)
poetry run pytest tests/unit -v    # Local unit tests only
```

## Coding Conventions

- **Python 3.12**, PEP 8, 4-space indentation
- **Type hints** on all function signatures
- **Naming**: `snake_case` (modules/functions), `PascalCase` (classes), `UPPER_SNAKE` (constants)
- **Imports**: stdlib → third-party → local; no wildcards

## Testing Guidelines

- **Unit tests**: `tests/unit/test_*.py` - mock external dependencies
- **Integration tests**: `tests/integration/test_*.py` - require Neo4j
- Run integration tests via Docker: `docker compose run --rm test`

## Key Architecture Patterns

1. **Unified Memory Model**: All data stored as `Episode`, `Psyche`, or `Note` types
2. **PersonaAdapter**: Single entry point for ingestion (extracts, links, persists)
3. **Retriever**: Vector similarity + graph traversal + query expansion for context retrieval
4. **QueryExpansion**: LLM-enhanced parsing of temporal refs and entities
5. **Dependency Injection**: `GraphOps` injected via FastAPI's `Depends()`

## Environment Configuration

Required in `.env`:
```env
LLM_SERVICE=openai/gpt-4o-mini
EMBEDDING_SERVICE=openai/text-embedding-3-small
OPENAI_API_KEY=sk-...
URI_NEO4J=bolt://neo4j:7687
USER_NEO4J=neo4j
PASSWORD_NEO4J=...
```

## Observability Requirements

This repo prioritizes **optimization work** - every layer must be transparent and measurable.

**Core Principles:**
1. **Always log metrics** - Rate monitors, timing, token counts must emit data even when throttling is disabled
2. **Never silent failures** - Errors bubble up with context; no empty catches
3. **Atomic state changes** - Checkpoints, configs use temp+rename pattern
4. **Resource cleanup** - Failed operations must release connections, reset adapters

**Eval Framework Observability:**
- `CallRateMonitor`: Logs RPM/TPM every 30s (must record calls regardless of `GRAPHITI_RPS` setting)
- `DeepLogger`: Per-question JSONL with ingestion/retrieval/generation timings
- Stage logs: `evals/results/graphiti_stage_logs/{user_id}.jsonl` for debugging adapter internals

**When adding new components:**
- Include timing instrumentation (`time.time()` around operations)
- Log to structured format (JSONL preferred)
- Expose metrics via env var or stats dict
