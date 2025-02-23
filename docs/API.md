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
POST /user/create
Content-Type: application/json

{
    "user_id": "string"
}

Response: 201 Created
{
    "message": "User {user_id} created successfully"
}
```

### Delete User
```http
POST /user/delete
Content-Type: application/json

{
    "user_id": "string"
}

Response: 200 OK
{
    "message": "User {user_id} deleted successfully"
}
```

## Data Operations

### Ingest Data
```http
POST /ingest
Content-Type: application/json

{
    "user_id": "string",
    "content": "string"  // Conversation or text content
}

Response: 201 Created
{
    "message": "Data ingested successfully"
}
```

### Add Custom Data
```http
POST /custom-data
Content-Type: application/json

{
    "user_id": "string",
    "nodes": [
        {
            "name": "string",  // Unique identifier for this node
            "properties": {
                "key1": "value1",
                "key2": "value2"
            },
            "target": "string",  // Optional target node
            "relation": "string" // Optional relationship type
        }
    ]
}

Response: 200 OK
{
    "status": "success",
    "message": "Added {n} nodes and {m} relationships",
    "nodes": ["node1", "node2"]
}
```

## Query Operations

### RAG Query
```http
POST /rag/query
Content-Type: application/json

{
    "user_id": "string",
    "query": "string"
}

Response: 200 OK
{
    "answer": "string"
}
```

### Vector-Only RAG Query
```http
POST /rag-query-vector
Content-Type: application/json

{
    "user_id": "string",
    "query": "string"
}

Response: 200 OK
{
    "query": "string",
    "response": "string"
}
```

### Ask Insights
```http
POST /ask
Content-Type: application/json

{
    "user_id": "string",
    "query": "string",
    "output_schema": {
        // Expected output structure with example values
    }
}

Response: 200 OK
{
    "result": {
        // Response matching output_schema
    }
}
```

## Error Handling

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
    "detail": "Error message explaining what went wrong"
}
```

### 404 Not Found
```json
{
    "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
    "detail": "Internal server error"
}
```

### 503 Service Unavailable
```json
{
    "detail": "Database connection error. Please ensure Neo4j is running and accessible."
}
```

## Database Connection

The API requires a running Neo4j instance. If Neo4j is not accessible, most endpoints will return a 503 error. Ensure Neo4j is running and properly configured in your environment variables before using the API.
