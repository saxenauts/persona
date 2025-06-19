import pytest
# from fastapi.testclient import TestClient  # Not needed here
# from server.main import app  # Not needed here

# Remove local test_client fixture

# All tests will use the test_client fixture from conftest.py

def test_version(test_client):
    response = test_client.get("/api/v1/version")
    assert response.status_code == 200
    assert response.json() == {"version": "1.0.0"}

def test_create_user(test_client):
    response = test_client.post("/api/v1/users/test_user")
    # User might already exist, so accept both 200 and 201
    assert response.status_code in [200, 201]
    assert "User test_user" in response.json()["message"]

def test_duplicate_user_creation(test_client):
    # First create a user
    test_client.post("/api/v1/users/test_user_dup")
    # Try to create the same user again
    response = test_client.post("/api/v1/users/test_user_dup")
    # Should return 200 for existing user
    assert response.status_code == 200
    assert "already exists" in response.json()["message"]

def test_ingest_data(test_client):
    # Ensure user exists
    test_client.post("/api/v1/users/test_user")
    response = test_client.post(
        "/api/v1/users/test_user/ingest",
        json={
            "title": "Python Programming",
            "content": "Python is a great programming language for AI and web development.",
            "metadata": {}
        }
    )
    if response.status_code != 201:
        print("DEBUG response:", response.json())
    assert response.status_code == 201
    assert "ingested successfully" in response.json()["message"]

def test_rag_query(test_client):
    response = test_client.post(
        "/api/v1/users/test_user/rag/query",
        json={"query": "What is Python good for?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()

def test_rag_query_vector(test_client):
    response = test_client.post(
        "/api/v1/users/test_user/rag/query-vector",
        json={"query": "Python programming"}
    )
    assert response.status_code == 200
    assert "query" in response.json()
    assert "response" in response.json()

def test_delete_user(test_client):
    # First create a user to delete
    test_client.post("/api/v1/users/test_user")
    response = test_client.delete("/api/v1/users/test_user")
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]

def test_delete_nonexistent_user(test_client):
    response = test_client.delete("/api/v1/users/nonexistent_user")
    # Should return 404 for non-existent user
    assert response.status_code == 404