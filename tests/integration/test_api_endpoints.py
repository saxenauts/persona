import pytest
from fastapi.testclient import TestClient
from server.main import app
from persona.models.schema import (
    CustomGraphUpdate,
    CustomNodeData,
    CustomRelationshipData
)

def test_version(test_client):
    """Test version endpoint"""
    response = test_client.get("/api/v1/version")
    assert response.status_code == 200
    assert "version" in response.json()

def test_user_lifecycle(test_client):
    """Test user creation and deletion"""
    user_id = "test_user_api"
    
    # Create user
    response = test_client.post(f"/api/v1/users/{user_id}")
    assert response.status_code == 201
    assert "created successfully" in response.json()["message"]
    
    # Delete user
    response = test_client.delete(f"/api/v1/users/{user_id}")
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]

def test_ingest_data(test_client):
    """Test data ingestion endpoint"""
    # Ensure user exists
    test_client.post("/api/v1/users/test_user_api")
    response = test_client.post(
        "/api/v1/users/test_user_api/ingest",
        json={
            "title": "Space Exploration",
            "content": "This is a test content about space exploration."
        }
    )
    if response.status_code != 201:
        print("DEBUG response:", response.json())
    assert response.status_code == 201
    assert "ingested successfully" in response.json()["message"]

def test_rag_query(test_client):
    """Test RAG query endpoint"""
    response = test_client.post(
        "/api/v1/users/test_user_api/rag/query",
        json={"query": "What is this text about?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()

def test_rag_query_vector(test_client):
    """Test vector-only RAG query endpoint"""
    response = test_client.post(
        "/api/v1/users/test_user_api/rag/query-vector",
        json={"query": "What is this text about?"}
    )
    assert response.status_code == 200
    assert "query" in response.json()
    assert "response" in response.json()

def test_ask_insights(test_client):
    """Test ask insights endpoint"""
    test_request = {
        "query": "What are the main topics?",
        "output_schema": {
            "topics": ["topic1", "topic2"],
            "summary": "test summary"
        }
    }
    response = test_client.post("/api/v1/users/test_user_api/ask", json=test_request)
    assert response.status_code == 200
    assert response.json() is not None

def test_custom_data(test_client):
    """Test custom data endpoint"""
    test_data = {
        "nodes": [
            {
                "name": "Test Node",
                "properties": {"test": "value"},
                "perspective": "test"
            }
        ],
        "relationships": [
            {
                "source": "Test Node",
                "target": "Another Node",
                "relation_type": "TEST_RELATION",
                "data": {"confidence": 0.9}
            }
        ]
    }
    response = test_client.post("/api/v1/users/test_user_api/custom-data", json=test_data)
    assert response.status_code == 200
    assert "status" in response.json() 