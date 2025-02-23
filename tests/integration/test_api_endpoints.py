import pytest
from fastapi.testclient import TestClient
from server.main import app
from persona.models.schema import (
    CustomGraphUpdate,
    CustomNodeData,
    CustomRelationshipData
)

client = TestClient(app)

def test_version():
    """Test version endpoint"""
    response = client.get("/api/v1/version")
    assert response.status_code == 200
    assert "version" in response.json()

def test_user_lifecycle():
    """Test user creation and deletion"""
    user_id = "test_user_api"
    
    # Create user
    response = client.post(
        "/api/v1/user/create",
        json={"user_id": user_id}
    )
    assert response.status_code == 201
    assert "created successfully" in response.json()["message"]
    
    # Delete user
    response = client.post(
        "/api/v1/user/delete",
        json={"user_id": user_id}
    )
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]

def test_ingest_data():
    """Test data ingestion endpoint"""
    response = client.post(
        "/api/v1/ingest",
        json={
            "user_id": "test_user_api",
            "content": "This is a test content about space exploration."
        }
    )
    assert response.status_code == 201
    assert "ingested successfully" in response.json()["message"]

def test_rag_query():
    """Test RAG query endpoint"""
    response = client.post(
        "/api/v1/rag/query",
        json={
            "user_id": "test_user_api",
            "query": "What is this text about?"
        }
    )
    assert response.status_code == 200
    assert "answer" in response.json()

def test_rag_query_vector():
    """Test vector-only RAG query endpoint"""
    response = client.post(
        "/api/v1/rag-query-vector",
        json={
            "user_id": "test_user_api",
            "query": "What is this text about?"
        }
    )
    assert response.status_code == 200
    assert "response" in response.json()

def test_ask_insights():
    """Test ask insights endpoint"""
    test_request = {
        "user_id": "test_user_api",
        "query": "What are the main topics?",
        "output_schema": {
            "topics": ["topic1", "topic2"],
            "summary": "test summary"
        }
    }
    response = client.post("/api/v1/ask", json=test_request)
    assert response.status_code == 200
    assert response.json() is not None

def test_custom_data():
    """Test custom data endpoint"""
    test_update = CustomGraphUpdate(
        user_id="test_user_api",
        nodes=[
            CustomNodeData(
                name="test_preference",
                properties={
                    "type": "test",
                    "value": "example"
                }
            )
        ],
        relationships=[
            CustomRelationshipData(
                source="test_preference",
                target="test_user_api",
                relation_type="PREFERENCE_OF"
            )
        ]
    )
    response = client.post("/api/v1/custom-data", json=test_update.model_dump())
    assert response.status_code == 200
    assert response.json()["status"] == "success" 