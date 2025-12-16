# Repository Guidelines

## Project Structure & Module Organization
- `persona/`: Core library
  - `core/`: Graph + Neo4j ops (constructor, graph_ops, neo4j_database)
  - `llm/`: Model clients, embeddings, prompts, providers
  - `services/`: App-facing services (ingest, ask, rag, custom data)
  - `models/`, `utils/`: Schemas and helpers
- `server/`: FastAPI app (`main.py`, routers, config, logging)
- `tests/`: `unit/` and `integration/` (container-dependent)
- `docs/`, `examples/`, `neo4j-data/`, `docker-compose.yml`, `pyproject.toml`

## Build, Test, and Development Commands
- Install (local): `poetry install`
- Run API (local): `poetry run uvicorn server.main:app --reload`
- Start stack (Docker, recommended): `docker compose up -d`
- Run tests (Docker, preferred): `docker compose run --rm test`
- Run tests (local): ensure Neo4j + `.env` set, then `poetry run pytest -q`
- Example: `docker compose run --rm -e DOCKER_ENV=1 app python examples/conversation.py`

## Coding Style & Naming Conventions
- Python 3.12, PEP 8, 4-space indentation, type hints encouraged
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`
- Imports: stdlib → third-party → local; avoid unused and wildcard imports
- Docstrings: concise, triple-quoted; document inputs/outputs and side effects

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`, `httpx` for API tests
- Structure: `tests/unit/test_*.py`, `tests/integration/test_*.py`
- Integration tests require Neo4j and LLM keys; prefer Docker profile: `docker compose run --rm test`
- Aim for meaningful coverage on new/changed code; add tests with clear arrange/act/assert sections

## Commit & Pull Request Guidelines
- Commits: imperative and scoped; Conventional style preferred (e.g., `feat:`, `fix:`, `refactor(ci):`)
- PRs: clear description, linked issues, behavior notes, screenshots/logs when helpful; list breaking changes explicitly
- Requirements: passing tests, updated docs/examples if behavior changes; small, focused PRs > large mixed ones

## Security & Configuration Tips
- Never commit secrets; use `.env` (see `.env.example`)
- Required ML config: `LLM_SERVICE`, `EMBEDDING_SERVICE` and provider API keys
- Tests and examples may incur LLM usage; monitor costs and rate limits
- Neo4j data persists in `neo4j-data/`; clear locally if you need a clean slate

