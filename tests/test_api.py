import pytest
from fastapi.testclient import TestClient

def test_version(test_client):
    response = test_client.get("/api/v1/version")
    assert response.status_code == 200
    assert "version" in response.json()

def test_neo4j_connection(test_client):
    # This will implicitly test Neo4j connection through the app lifespan
    response = test_client.get("/api/v1/version")
    assert response.status_code == 200

def test_user_lifecycle(test_client):
    # Create user
    create_response = test_client.post(
        "/api/v1/users",
        json={"user_id": "test_user"}
    )
    assert create_response.status_code == 201
    
    # Test data ingestion
    ingest_response = test_client.post(
        "/api/v1/ingest/test_user",
        json={"content": "Python is a great programming language for AI and web development."}
    )
    assert ingest_response.status_code == 200
    
    # Test RAG query
    rag_response = test_client.post(
        "/api/v1/rag/test_user/query",
        json={"query": "What is Python good for?"}
    )
    assert rag_response.status_code == 200
    assert "answer" in rag_response.json()
    
    # Test vector-only RAG query
    vector_response = test_client.post(
        "/api/v1/rag-query-vector",
        params={"query": "Python programming", "user_id": "test_user"}
    )
    assert vector_response.status_code == 200
    assert "response" in vector_response.json()
    
    # Delete user
    delete_response = test_client.delete("/api/v1/users/test_user")
    assert delete_response.status_code == 200

@pytest.mark.asyncio
async def test_constructor_flow(test_client):
    # First create a test user
    create_response = test_client.post(
        "/api/v1/users",
        json={"user_id": "test_user"}
    )
    assert create_response.status_code == 201
    
    # Then test the constructor flow
    response = test_client.post("/api/v1/test-constructor-flow")
    assert response.status_code == 200
    assert "context" in response.json()
    assert "status" in response.json()
    
    # Clean up
    delete_response = test_client.delete("/api/v1/users/test_user")
    assert delete_response.status_code == 200