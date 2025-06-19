# Codebase Overview

This document provides a high-level orientation for new contributors. It explains the major packages in the repository and how they fit together to form the Persona service.

## Top Level Layout

```
persona/           # Core library code
server/            # FastAPI service
tests/             # pytest suite
docs/              # Documentation
```

### persona/

The `persona` package holds all domain logic. Important subpackages include:

- **core/** – graph handling and retrieval logic. Notable modules are:
  - `graph_ops.py`: async wrapper around Neo4j for CRUD operations and vector search.
  - `constructor.py`: orchestrates graph updates from raw text using LLM calls.
  - `rag_interface.py`: implements retrieval augmented generation using graph context.
- **llm/** – wrappers for calling OpenAI APIs. See [LLM_CODEBASE.md](LLM_CODEBASE.md) for details.
- **models/** – Pydantic models for data exchange between layers.
- **services/** – high level business logic used by the API routes (user creation, ingestion, RAG queries, etc.).

### server/

Contains the FastAPI application. Key files are:

- `main.py`: entry point that creates the app instance with lifespan management.
- `routers/graph_api.py`: defines the `/api/v1` endpoints which call into the service layer.
- `dependencies.py`: FastAPI dependency injection for GraphOps and other shared resources.
- `config.py`: loads environment variables such as database credentials and OpenAI keys.

### docs/

Additional documentation including API reference and development guides.

### tests/

Automated tests executed via `pytest`. The `tests/docker-compose.yml` file spins up Neo4j for the suite.

## Architecture Patterns

### GraphOps Dependency Injection

The application uses a **single global GraphOps instance** managed through FastAPI's dependency injection system:

1. **Initialization**: A single `GraphOps` instance is created during FastAPI startup in the lifespan context manager
2. **Storage**: The instance is stored in `app.state.graph_ops` for global access
3. **Injection**: Services receive the GraphOps instance via FastAPI's `Depends(get_graph_ops)` dependency
4. **Benefits**: Eliminates connection overhead, ensures consistent resource lifecycle, and simplifies testing

This pattern ensures that all database operations share the same Neo4j connection pool and avoids the overhead of creating new connections for each request.

## Data Flow

1. **Ingestion** – Text is submitted to the `/api/v1/users/{user_id}/ingest` endpoint. `IngestService` invokes the `GraphConstructor` which calls `persona.llm.llm_graph` to extract nodes and relationships. These are persisted via `GraphOps`.
2. **Retrieval** – A question is sent to `/api/v1/users/{user_id}/rag/query`. `RAGService` uses `GraphOps` to retrieve relevant nodes, gathers context with `GraphContextRetriever`, and sends it to the LLM for an answer.
3. **Custom Data** – The `/api/v1/users/{user_id}/custom-data` endpoint allows clients to post structured nodes or relationships directly.

## Running the Project

Development is easiest through Docker Compose:

```bash
docker compose up
```

This launches Neo4j and the FastAPI server with hot reload enabled. The API documentation is then available at `http://localhost:8000/docs`.

## Summary

Persona builds a personal knowledge graph for each user, stores it in Neo4j, and uses OpenAI models for extraction and question answering. The `persona` package implements the domain logic while the `server` package exposes a REST API. Newcomers should start by exploring the service layer and the LLM modules described in [LLM_CODEBASE.md](LLM_CODEBASE.md).


