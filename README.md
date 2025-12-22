# Persona - Intelligent User Memory

![Persona Banner](docs/assets/banner.svg)

## Overview

Persona is a language-dependent digital identity system that evolves with a user's digital footprint.

It creates and maintains a dynamic knowledge graph for each user, providing contextually rich information to any programming interface—particularly LLM-based systems—enabling next-generation personalization and user-centric experiences.

At its core, Persona aims to create a **memetic digital organism** that can evolve and grow with the user, representing the user's mindspace digitally.

## Vision 

The digital future is getting more personalized, and our apps and web are getting data rich. Interfaces should naturally evolve and allow more personalized, dynamic experiences. Personalization at its fundamental is about **understanding the user**.

Traditionally we've represented users as fixed, static relational data points in tables or vector embeddings. But our minds and identities are associative and dynamic in nature.

We use language to connect different parts of our lives. With LLMs it's possible to map this complexity and to make sense of it. So our user representation should not be left behind. 

Read the motivation and design decisions in depth [here](https://saxenauts.io/blog/persona-graph).

## Memory Model

Persona organizes user data into three fundamental memory types:

| Type | Purpose | Examples |
|------|---------|----------|
| **Episode** | What happened | Events, conversations, experiences |
| **Psyche** | Who they are | Traits, preferences, values, beliefs |
| **Goal** | What they want | Tasks, projects, reminders |

All memories are connected through a knowledge graph with **temporal linking** for narrative continuity.

## Features

- **Dynamic Knowledge Graph Construction**: Automatically builds and updates a user's knowledge graph from interaction data
- **Typed Memory System**: Three memory classes with semantic meaning (Episode, Psyche, Goal)
- **Temporal Chaining**: Automatic linking of episodes to track narrative progression
- **Contextual Query Processing**: RAG with vector similarity + graph traversal
- **Structured Insights**: Ask questions and get JSON-structured answers

## Quick Start

### Prerequisites

- **Docker & Docker Compose**: Required for Neo4j and the application
- **OpenAI API Key**: Required for LLM operations

### Installation

```bash
git clone https://github.com/saxenauts/persona.git
cd persona
```

Create a `.env` file:

```env
URI_NEO4J=bolt://neo4j:7687
USER_NEO4J=neo4j
PASSWORD_NEO4J=your_secure_password
NEO4J_AUTH=neo4j/your_secure_password

LLM_SERVICE=openai/gpt-4o-mini
EMBEDDING_SERVICE=openai/text-embedding-3-small
OPENAI_API_KEY=your_openai_api_key
```

Start services:

```bash
docker compose up -d
```

Access the API at `http://localhost:8000/docs`.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/users/{user_id}` | Create a new user |
| `DELETE /api/v1/users/{user_id}` | Delete a user |
| `POST /api/v1/users/{user_id}/ingest` | Ingest text and extract memories |
| `POST /api/v1/users/{user_id}/rag/query` | Query with full context retrieval |
| `POST /api/v1/users/{user_id}/ask` | Get structured JSON insights |

See full API documentation at `http://localhost:8000/docs`.

## Architecture

```
persona/
├── adapters/          # Unified ingestion orchestrator
├── core/              # Database ops, retrieval, context
├── models/            # Memory types (Episode, Psyche, Goal)
├── llm/               # Multi-provider LLM clients
└── services/          # Business logic

server/                # FastAPI application
tests/                 # Test suite
```

**Key Components:**
- **PersonaAdapter**: Unified entry point for all data ingestion
- **Retriever**: Vector similarity + graph traversal for context
- **ContextFormatter**: Memory → XML context for LLM consumption

## Running Tests

```bash
# Docker (recommended)
docker compose run --rm test

# Local
poetry install
poetry run pytest tests/unit -v
```

## Roadmap

- [ ] Intelligent forgetting
- [ ] Associativity weight
- [ ] Memory relevance metric
- [ ] Agentic memory update and linking
- [ ] Agentic retrieval
- [ ] Agentic manual edits
- [ ] Real-time personalized context
- [ ] Persona adapter cross-linking

## License

MIT License
