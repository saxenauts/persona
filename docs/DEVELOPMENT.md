# Development Guide

## Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/persona-graph.git
   cd persona-graph
   ```

2. **Install Dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Project Structure

```
persona/
├── api/              # FastAPI routes and endpoints
├── services/         # Business logic and core services
├── database/         # Database managers and models
├── utils/            # Helper functions and utilities
├── tests/            # Test suite
└── examples/         # Example scripts and notebooks
```

## Key Components

1. **Graph Database Manager** (`database/neo4j_database.py`)
   - Handles all Neo4j operations
   - Manages node and relationship creation
   - Implements query operations

2. **Services** (`services/`)
   - `conversation_service.py`: Processes conversations
   - `custom_data_service.py`: Handles custom data ingestion
   - `query_service.py`: Implements RAG queries

3. **API Layer** (`api/`)
   - RESTful endpoints
   - Request validation
   - Response formatting

## Development Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Run Tests**
   ```bash
   pytest tests/
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

1. **Unit Tests**
   ```bash
   pytest tests/unit/
   ```

2. **Integration Tests**
   ```bash
   pytest tests/integration/
   ```

3. **End-to-End Tests**
   ```bash
   pytest tests/e2e/
   ```

## Common Development Tasks

1. **Adding a New Endpoint**
   - Add route in `api/routes/`
   - Implement service logic
   - Add tests
   - Update API docs

2. **Modifying Graph Schema**
   - Update models in `database/models.py`
   - Add migration if needed
   - Update affected services
   - Add tests

3. **Adding Custom Data Type**
   - Extend `CustomDataService`
   - Add validation
   - Update graph schema
   - Add example usage

## Troubleshooting

1. **Neo4j Connection**
   - Check Neo4j is running: `docker ps`
   - Verify credentials in `.env`
   - Check logs: `docker logs neo4j`

2. **API Issues**
   - Check FastAPI logs
   - Verify request format
   - Use debug mode: `uvicorn main:app --reload --log-level debug`

3. **Testing Problems**
   - Use `pytest -v` for verbose output
   - Check test database connection
   - Verify test data setup

## Best Practices

1. **Code Quality**
   - Write docstrings
   - Add type hints
   - Keep functions focused
   - Use meaningful names

2. **Testing**
   - Write tests first (TDD)
   - Mock external services
   - Use fixtures
   - Test edge cases

3. **Git Workflow**
   - Small, focused commits
   - Clear commit messages
   - Regular rebasing
   - Clean PR history 