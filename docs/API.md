# API Reference

Base URL: `http://localhost:8000/api/v1`

## Version
```http
GET /version
```
Returns the current API version.

## User Management

### Create User
```http
POST /users/{user_id}

Response: 201 Created
{
    "message": "User {user_id} created successfully"
}
```

### Delete User
```http
DELETE /users/{user_id}

Response: 200 OK
{
    "message": "User {user_id} deleted successfully"
}
```

## Ingestion

### Ingest Content
```http
POST /users/{user_id}/ingest
Content-Type: application/json

{
    "content": "Had a great meeting with Sarah about the Q4 roadmap...",
    "source_type": "conversation"
}

Response: 201 Created
{
    "message": "Content ingested successfully",
    "memories_created": 3
}
```

### Batch Ingest
```http
POST /users/{user_id}/ingest/batch
Content-Type: application/json

{
    "items": [
        {"content": "First entry...", "source_type": "notes"},
        {"content": "Second entry...", "source_type": "conversation"}
    ]
}
```

## Query Operations

### RAG Query
```http
POST /users/{user_id}/rag/query
Content-Type: application/json

{
    "query": "What projects am I working on?"
}

Response: 200 OK
{
    "query": "What projects am I working on?",
    "response": "Based on your memories, you're working on..."
}
```

### Ask (Structured Insights)
```http
POST /users/{user_id}/ask
Content-Type: application/json

{
    "query": "What are my preferences?",
    "output_schema": {
        "preferences": ["example"],
        "summary": "string"
    }
}

Response: 200 OK
{
    "result": {
        "preferences": ["remote work", "morning meetings"],
        "summary": "User prefers flexible work arrangements"
    }
}
```

## Error Responses

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input |
| 404 | User not found |
| 500 | Internal server error |
| 502 | External service (LLM) error |
| 503 | Database connection error |
