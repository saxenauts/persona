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
    "message": "User {user_id} created successfully",
    "status": "created"
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
    "message": "Data ingested successfully",
    "memories_created": 3,
    "memories_created_by_type": {"episode": 1, "psyche": 1, "note": 1},
    "links_created": 2,
    "timings_ms": {"extract": 150.0, "embed": 50.0, "persist": 30.0, "total": 230.0}
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

Response: 201 Created
{
    "message": "Successfully ingested batch of 2 items",
    "memories_created": 5,
    "memories_created_by_type": {"episode": 2, "psyche": 2, "note": 1},
    "links_created": 3,
    "timings_ms": {"extract": 300.0, "embed": 100.0, "persist": 60.0, "total": 460.0}
}
```

## Query Operations

### Persona Query (Agent Loop)
Agentic query with tool-calling loop. The agent autonomously decides when to search memories, read context, or write new information.

```http
POST /users/{user_id}/persona/query
Content-Type: application/json

{
    "query": "What do you remember about my fitness goals?",
    "include_stats": true,
    "user_timezone": "America/Los_Angeles",
    "max_turns": 10,
    "timeout": 30.0
}

Response: 200 OK
{
    "answer": "Based on your memories, you set a goal to...",
    "status": "success",
    "stats": {
        "tool_calls_made": 3,
        "turns": 2,
        "usage": {"prompt_tokens": 500, "completion_tokens": 200},
        "total_ms": 1234.5
    }
}
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| query | string | required | The user's question |
| include_stats | bool | false | Include execution statistics |
| user_timezone | string | "UTC" | Timezone for temporal queries |
| session_id | string | null | Session ID for conversation continuity |
| max_turns | int | null | Max tool-calling turns (null = unlimited) |
| timeout | float | null | Max seconds before returning (null = unlimited) |

**Status values:**
- `success` - Agent finished naturally
- `max_turns` - Hit turn limit
- `timeout` - Hit time limit
- `error` - LLM call failed

### Persona Ask (Structured Output)
Direct retrieval + structured JSON extraction. No agent loop, no tools.

```http
POST /users/{user_id}/persona/ask
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

**Parameters:**

| Field | Type | Description |
|-------|------|-------------|
| query | string | The extraction query |
| output_schema | object | Expected output structure with example values |

## Error Responses

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input |
| 404 | User not found |
| 500 | Internal server error |
| 502 | External service (LLM) error |
| 503 | Database connection error |
