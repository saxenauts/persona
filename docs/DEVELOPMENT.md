# Development Guide

## Prerequisites

- **Docker & Docker Compose**: Required for running Neo4j and the application
- **OpenAI API Key**: Required for LLM operations (will consume API calls during testing)
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
   # Edit .env with your settings:
   # - OPENAI_API_KEY (required)
   # - NEO4J credentials
   ```

3. **Start Services**
   ```bash
   docker compose up -d
   ```

## Project Structure

```
persona/
├── server/           # FastAPI server and API routes
├── persona/          # Core application logic
│   ├── core/         # Database operations and graph management
│   ├── services/     # Business logic services
│   ├── models/       # Pydantic models and schemas
│   └── llm/          # LLM integration and embeddings
├── tests/            # Test suite
├── examples/         # Example scripts
├── docs/             # Documentation
└── docker-compose.yml # Docker services configuration
```

## Key Components

1. **Graph Database Manager** (`persona/core/neo4j_database.py`)
   - Handles all Neo4j operations
   - Manages node and relationship creation
   - Implements vector similarity search

2. **Services** (`persona/services/`)
   - `user_service.py`: User management
   - `ingest_service.py`: Data ingestion
   - `rag_service.py`: RAG query processing
   - `ask_service.py`: Structured insights
   - `custom_data_service.py`: Custom data handling

3. **API Layer** (`server/routers/`)
   - RESTful endpoints following `/users/{user_id}/...` pattern
   - Request validation with Pydantic models
   - Response formatting

## Development Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Run Tests**
   ```bash
   # Run all tests in Docker (recommended)
   docker compose run --rm test
   
   # Or run specific test files
   docker compose run --rm test pytest tests/test_api.py -v
   ```

3. **Code Style**
   ```bash
   # Format code
   black .
   
   # Check types
   mypy .
   
   # Lint code
   flake8
   ```

4. **Submit PR**
   - Write clear PR description
   - Include test coverage
   - Link related issues

## Testing

**Note**: Tests require Docker and will consume OpenAI API calls.

1. **Run All Tests**
   ```bash
   docker compose run --rm test
   ```

2. **Run Specific Test Categories**
   ```bash
   # Unit tests
   docker compose run --rm test pytest tests/unit/
   
   # Integration tests
   docker compose run --rm test pytest tests/integration/
   
   # API tests
   docker compose run --rm test pytest tests/test_api.py
   ```

3. **Test with Verbose Output**
   ```bash
   docker compose run --rm test pytest -v
   ```

## Common Development Tasks

1. **Adding a New Endpoint**
   - Add route in `server/routers/`
   - Implement service logic in `persona/services/`
   - Add tests in `tests/`
   - Update API docs in `docs/API.md`

2. **Modifying Graph Schema**
   - Update models in `persona/models/schema.py`
   - Update database operations in `persona/core/`
   - Update affected services
   - Add tests

3. **Adding Custom Data Type**
   - Extend `CustomDataService` in `persona/services/`
   - Add validation in `persona/models/`
   - Update graph schema
   - Add example usage

## Troubleshooting

1. **Neo4j Connection**
   - Check Neo4j is running: `docker ps`
   - Verify credentials in `.env`
   - Check logs: `docker logs persona-neo4j`

2. **API Issues**
   - Check FastAPI logs: `docker logs persona-app`
   - Verify request format matches API docs
   - Use debug mode: `docker compose logs app`

3. **Testing Problems**
   - Ensure all services are running: `docker compose up -d`
   - Check OpenAI API key is valid
   - Verify test database connection

4. **OpenAI API Costs**
   - Tests will consume API calls
   - Monitor usage in OpenAI dashboard
   - Consider using test API keys for development

## Best Practices

1. **Code Quality**
   - Write docstrings
   - Add type hints
   - Keep functions focused
   - Use meaningful names

2. **Testing**
   - Write tests first (TDD)
   - Mock external services when possible
   - Use fixtures
   - Test edge cases

3. **Git Workflow**
   - Small, focused commits
   - Clear commit messages
   - Regular rebasing
   - Clean PR history

## API Usage Examples

### Create a User
```bash
curl -X POST "http://localhost:8000/api/v1/users/my_user"
```

### Ingest Data
```bash
curl -X POST "http://localhost:8000/api/v1/users/my_user/ingest" \
  -H "Content-Type: application/json" \
  -d '{"content": "I love Python programming"}'
```

### Query the Graph
```bash
curl -X POST "http://localhost:8000/api/v1/users/my_user/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What do I like?"}'
```

### View Data in Neo4j
```cypher
MATCH (n:NodeName {UserId: 'my_user'}) RETURN n
``` 