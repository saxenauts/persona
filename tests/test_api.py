import pytest
from fastapi.testclient import TestClient
from app_server.main import app  # Adjust the import based on your project structure

client = TestClient(app)

@pytest.fixture(scope="module")
def test_client():
    return client

def test_version(test_client):
    response = test_client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": "1.0.0"}

def test_create_user(test_client):
    response = test_client.post(
        "/api/v1/users",
        json={"user_id": "test_user"}
    )
    assert response.status_code == 201
    assert response.json() == {"user_id": "test_user"}

def test_duplicate_user_creation(test_client):
    response = test_client.post(
        "/api/v1/users",
        json={"user_id": "test_user"}
    )
    assert response.status_code == 400  # Assuming it returns 400 for duplicates
    assert response.json()["detail"] == "User already exists."

def test_ingest_data(test_client):
    response = test_client.post(
        "/api/v1/ingest/test_user",
        json={"content": "Python is a great programming language for AI and web development."}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "Data ingested successfully."

def test_rag_query(test_client):
    response = test_client.post(
        "/api/v1/rag/test_user/query",
        json={"query": "What is Python good for?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
    assert isinstance(response.json()["answer"], str)

def test_rag_query_vector(test_client):
    response = test_client.post(
        "/api/v1/rag-query-vector",
        params={"query": "Python programming", "user_id": "test_user"}
    )
    assert response.status_code == 200
    assert "response" in response.json()
    assert isinstance(response.json()["response"], str)

def test_delete_user(test_client):
    response = test_client.delete("/api/v1/users/test_user")
    assert response.status_code == 200
    assert response.json()["status"] == "User deleted successfully."

def test_delete_nonexistent_user(test_client):
    response = test_client.delete("/api/v1/users/nonexistent_user")
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found."

def test_test_learn_flow(test_client):
    response = test_client.post("/api/v1/test-learn-flow")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "Success"
    assert "schema_id" in json_response
    assert "stored_schema" in json_response

def test_test_custom_data_flow(test_client):
    response = test_client.post("/api/v1/test-custom-data-flow")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "Success"
    assert "update_result" in json_response
    assert "stored_data" in json_response