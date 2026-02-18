# Persona v0.3 Quick Start

Last updated: 2026-02-17 (America/Los_Angeles)

This guide is the minimal onboarding path for v0.3 release validation: install, run, ingest one fact, and query it back.

## 1) Prerequisites

- Docker with Docker Compose
- Python 3.12+
- An LLM API key (OpenAI or Azure Foundry)

## 2) Setup

```bash
git clone https://github.com/saxenauts/persona.git
cd persona
cp .env.example .env
```

Edit `.env` and set at least:

- `OPENAI_API_KEY` (or Azure settings)
- `PASSWORD_NEO4J`

## 3) Start Services

```bash
docker compose up -d
```

Validate runtime is up:

```bash
curl -sf http://localhost:8000/health
```

Expected output:

```json
{"status":"ok"}
```

## 4) Minimal Ingest and Query

Create a user:

```bash
curl -s -X POST "http://localhost:8000/api/v1/users/demo_user"
```

Ingest one memory:

```bash
curl -s -X POST "http://localhost:8000/api/v1/users/demo_user/ingest" \
  -H "Content-Type: application/json" \
  -d '{"content":"I prefer espresso in the morning."}'
```

Query with chat endpoint:

```bash
curl -s -X POST "http://localhost:8000/api/v1/users/demo_user/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What do I prefer in the morning?"}]}'
```

## 5) Troubleshooting

- If Neo4j/auth fails, verify `.env` values for `USER_NEO4J`, `PASSWORD_NEO4J`, and `URI_NEO4J`.
- If app is not reachable, run `docker compose logs app`.
- If ingest/query fails with model errors, verify your API key and provider model settings in `.env`.

## Reproducibility Commands

Run the same checks used in release hardening:

```bash
poetry run pytest tests/unit -v
docker compose run --rm test
```
