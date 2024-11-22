import pytest
from fastapi.testclient import TestClient
from app_server.main import app

client = TestClient(app)

@pytest.fixture(scope="module")
def test_client():
    return client

@pytest.mark.asyncio
async def test_test_constructor_flow(test_client):
    response = test_client.post("/api/v1/test-constructor-flow")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "Graph updated successfully"
    assert "context" in json_response

@pytest.mark.asyncio
async def test_test_learn_flow(test_client):
    response = test_client.post("/api/v1/test-learn-flow")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "Success"
    assert "schema_id" in json_response
    assert "stored_schema" in json_response

@pytest.mark.asyncio
async def test_test_custom_data_flow(test_client):
    response = test_client.post("/api/v1/test-custom-data-flow")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "Success"
    assert "update_result" in json_response
    assert "stored_data" in json_response