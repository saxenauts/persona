# API Documentation

Persona Graph exposes a set of RESTful APIs to interact with user knowledge graphs. This document provides detailed information about each endpoint, including request formats, parameters, and responses.

## Base URL

All API endpoints are prefixed with `/api/v1`. For example, to access the user creation endpoint, use `http://localhost:8000/api/v1/users`.

## Endpoints

### 1. Create a New User

- **URL:** `/api/v1/users`
- **Method:** `POST`
- **Description:** Creates a new user in the system.

#### Request

- **Headers:**
  - `Content-Type: application/json`

- **Body:**

  ```json
  {
    "user_id": "alice123"
  }
  ```

#### Response

- **Status Code:** `201 Created`
- **Body:**

  ```json
  {
    "message": "User alice123 created successfully"
  }
  ```

### 2. Delete a User

- **URL:** `/api/v1/users/{user_id}`
- **Method:** `DELETE`
- **Description:** Deletes an existing user and all associated data.

#### Request

- **Parameters:**
  - `user_id` (string): Unique identifier of the user to delete.

#### Response

- **Status Code:** `200 OK`
- **Body:**

  ```json
  {
    "message": "User alice123 deleted successfully"
  }
  ```

### 3. Ingest User Data

- **URL:** `/api/v1/ingest/{user_id}`
- **Method:** `POST`
- **Description:** Ingests unstructured data into the user's knowledge graph.

#### Request

- **Parameters:**
  - `user_id` (string): Unique identifier of the user.

- **Headers:**
  - `Content-Type: application/json`

- **Body:**

  ```json
  {
    "content": "Alice is a software engineer who loves hiking and photography."
  }
  ```

#### Response

- **Status Code:** `200 OK`
- **Body:**

  ```json
  {
    "message": "Data ingested successfully"
  }
  ```

### 4. Perform a RAG Query

- **URL:** `/api/v1/rag/{user_id}/query`
- **Method:** `POST`
- **Description:** Performs a Retrieval-Augmented Generation (RAG) query based on the user's knowledge graph.

#### Request

- **Parameters:**
  - `user_id` (string): Unique identifier of the user.

- **Headers:**
  - `Content-Type: application/json`

- **Body:**

  ```json
  {
    "query": "What are Alice's hobbies?"
  }
  ```

#### Response

- **Status Code:** `200 OK`
- **Body:**

  ```json
  {
    "answer": "Alice enjoys hiking and photography."
  }
  ```

### 5. Test Constructor Flow

- **URL:** `/api/v1/test-constructor-flow`
- **Method:** `POST`
- **Description:** Tests the graph construction flow by processing predefined data.

#### Request

- **Headers:**
  - `Content-Type: application/json`

#### Response

- **Status Code:** `200 OK`
- **Body:**

  ```json
  {
    "status": "Graph updated successfully",
    "context": "..."
  }
  ```

### 6. RAG Query (Alternative Endpoint)

- **URL:** `/api/v1/rag-query`
- **Method:** `POST`
- **Description:** Performs a RAG query based on the user's knowledge graph.

#### Request

- **Headers:**
  - `Content-Type: application/json`

- **Body:**

  ```json
  {
    "query": "What is Python?",
    "user_id": "alice123"
  }
  ```

#### Response

- **Status Code:** `200 OK`
- **Body:**

  ```json
  {
    "query": "What is Python?",
    "response": "Python is a high-level programming language..."
  }
  ```

### 7. RAG Query Vector (Alternative Endpoint)

- **URL:** `/api/v1/rag-query-vector`
- **Method:** `POST`
- **Description:** Performs a vector-based RAG query for enhanced similarity search.

#### Request

- **Headers:**
  - `Content-Type: application/json`

- **Body:**

  ```json
  {
    "query": "Describe Python programming.",
    "user_id": "alice123"
  }
  ```

#### Response

- **Status Code:** `200 OK`
- **Body:**

  ```json
  {
    "query": "Describe Python programming.",
    "response": "Python is a versatile programming language used for..."
  }
  ```

## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of an API request.

- **400 Bad Request:** The request was invalid or cannot be otherwise served.
- **404 Not Found:** The requested resource could not be found.
- **500 Internal Server Error:** An unexpected error occurred on the server.

#### Example Error Response

```json
{
"detail": "User alice123 does not exist."
}
```
