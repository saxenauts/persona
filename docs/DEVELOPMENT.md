# Development Guide

## Prerequisites

- **Docker & Docker Compose**: Required for running Neo4j and the application
- **OpenAI API Key**: Required for LLM operations
- **Python 3.12+**: For local development

## Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/saxenauts/persona.git
   cd persona
   ```

2. **Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start Services**
   ```bash
   docker compose up -d
   ```

## Project Structure

```
persona/
├── adapters/         # High-level orchestrators (PersonaAdapter)
├── core/             # Database operations, retrieval, context
├── services/         # Business logic (ingestion, persona)
├── models/           # Memory types and API schemas
└── llm/              # LLM clients and functions

server/
├── main.py           # App entry point
├── routers/          # API route definitions
├── dependencies.py   # Dependency injection
└── config.py         # Environment configuration

tests/
├── unit/             # Unit tests (no external deps)
└── integration/      # Integration tests (requires Neo4j)
```

## Key Components

1. **PersonaAdapter** (`adapters/persona_adapter.py`)
   - Unified entry point for ingestion
   - Orchestrates extraction → linking → persistence

2. **PersonaService** (`services/persona_service.py`)
   - Primary entry point for memory-augmented dialogue
   - `run_agent()`: Agent loop with recall/record tools
   - `ask()`: Structured JSON extraction

3. **Retriever** (`core/retrieval.py`)
   - Time-windowed fetch + graph traversal (via stored links)
   - Returns prose-formatted context for LLM

4. **MemoryStore** (`core/memory_store.py`)
   - CRUD operations for typed memories
   - Handles temporal linking

## Development Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Run Tests**
   ```bash
   # Docker (recommended)
   docker compose run --rm test
   
   # Local unit tests
   poetry run pytest tests/unit -v
   ```

3. **Code Style**
   ```bash
   black .
   mypy .
   flake8
   ```

## Evaluation Workflow (PersonaMem)

Run evals from the `memory-evals` repo using the venv binary:
```bash
cd /Users/saxenauts/Documents/InnerNets AI Inc/persona/memory-evals
PYTHONUNBUFFERED=1 EVAL_CLOSE_SESSIONS=false EVAL_REFRESH_MEMEPLEX=false .venv/bin/mem-eval run \
  --benchmark personamem --adapter persona --samples 100 --seed 42 --output results
```

Monitor progress with the log and deep logs:
```bash
tail -f logs/full/<run>.log
ls -lt results/run_*/deep_logs.jsonl | head -3
```

Summaries and comparisons:
```bash
.venv/bin/mem-eval analyze <run_id> --summary
.venv/bin/mem-eval compare <run_a> <run_b>
.venv/bin/mem-eval aggregate <run_id> --output-json results/<run_id>_agg.json
```

## Testing

```bash
# Run all tests
docker compose run --rm test

# Unit tests only
poetry run pytest tests/unit -v

# Specific test file
poetry run pytest tests/unit/test_memory.py -v
```

## Common Development Tasks

### Adding a New Endpoint
1. Add route in `server/routers/graph_api.py`
2. Implement service logic in `persona/services/`
3. Add tests in `tests/unit/`
4. Update API docs

### Adding a New Memory Type
1. Extend `persona/models/memory.py`
2. Update `MemoryStore` in `persona/core/memory_store.py`
3. Update `ContextFormatter` in `persona/core/context.py`
4. Add tests

## Troubleshooting

1. **Neo4j Connection**
   - Check Neo4j is running: `docker ps`
   - Verify credentials in `.env`
   - Check logs: `docker logs persona-neo4j`

2. **API Issues**
   - Check FastAPI logs: `docker logs persona-app`
   - Use debug mode: `docker compose logs app`

3. **Testing Problems**
   - Ensure services are running: `docker compose up -d`
   - Check API key is valid

## API Usage Examples

```bash
# Create a user
curl -X POST "http://localhost:8000/api/v1/users/my_user"

# Ingest data
curl -X POST "http://localhost:8000/api/v1/users/my_user/ingest" \
  -H "Content-Type: application/json" \
  -d '{"content": "I love Python programming"}'

# Chat with agent loop
curl -X POST "http://localhost:8000/api/v1/users/my_user/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What do I like?"}]}'

# Get structured output
curl -X POST "http://localhost:8000/api/v1/users/my_user/persona/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my preferences?", "output_schema": {"preferences": []}}'
```
