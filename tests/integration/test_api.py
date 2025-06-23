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

def test_user_creation_and_deletion(test_client, api_test_user):
    """Test user creation and deletion"""
    user_id = api_test_user
    
    # Create user
    response = test_client.post(f"/api/v1/users/{user_id}")
    # User might already exist from previous runs if not cleaned properly, so accept 200 or 201
    assert response.status_code in [200, 201]
    assert f"User {user_id}" in response.json()["message"]
    
    # Delete user
    response = test_client.delete(f"/api/v1/users/{user_id}")
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]

def test_duplicate_user_creation(test_client, api_test_user):
    """Test that creating a user that already exists returns a 200."""
    user_id = api_test_user
    # First create a user
    response = test_client.post(f"/api/v1/users/{user_id}")
    assert response.status_code in [200, 201]

    # Try to create the same user again
    response = test_client.post(f"/api/v1/users/{user_id}")
    # Should return 200 for existing user
    assert response.status_code == 200
    assert "already exists" in response.json()["message"]

def test_delete_nonexistent_user(test_client):
    """Test that deleting a non-existent user returns a 404."""
    response = test_client.delete("/api/v1/users/nonexistent_user_for_sure")
    assert response.status_code == 404

def test_ingest_data(test_client, api_test_user):
    """Test data ingestion endpoint"""
    user_id = api_test_user
    # Ensure user exists
    test_client.post(f"/api/v1/users/{user_id}")
    response = test_client.post(
        f"/api/v1/users/{user_id}/ingest",
        json={
            "title": "Space Exploration",
            "content": "This is a test content about space exploration.",
            "metadata": {}
        }
    )
    if response.status_code != 201:
        print("DEBUG response:", response.json())
    assert response.status_code == 201
    assert "ingested successfully" in response.json()["message"]

def test_rag_query(test_client, api_test_user):
    """Test RAG query endpoint"""
    user_id = api_test_user
    test_client.post(f"/api/v1/users/{user_id}")
    response = test_client.post(
        f"/api/v1/users/{user_id}/rag/query",
        json={"query": "What is this text about?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()

def test_rag_query_vector(test_client, api_test_user):
    """Test vector-only RAG query endpoint"""
    user_id = api_test_user
    test_client.post(f"/api/v1/users/{user_id}")
    response = test_client.post(
        f"/api/v1/users/{user_id}/rag/query-vector",
        json={"query": "What is this text about?"}
    )
    assert response.status_code == 200
    assert "query" in response.json()
    assert "response" in response.json()

def test_ask_insights(test_client, api_test_user):
    """Test ask insights endpoint"""
    user_id = api_test_user
    test_client.post(f"/api/v1/users/{user_id}")
    test_request = {
        "query": "What are the main topics?",
        "output_schema": {
            "topics": ["topic1", "topic2"],
            "summary": "test summary"
        }
    }
    response = test_client.post(f"/api/v1/users/{user_id}/ask", json=test_request)
    assert response.status_code == 200
    assert response.json() is not None

def test_custom_data(test_client, api_test_user):
    """Test custom data endpoint"""
    user_id = api_test_user
    test_client.post(f"/api/v1/users/{user_id}")
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
    response = test_client.post(f"/api/v1/users/{user_id}/custom-data", json=test_data)
    assert response.status_code == 200
    assert "status" in response.json() 