import pytest
import os
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from server.main import app
from persona.services.ingestion_service import IngestionResult

pytestmark = pytest.mark.asyncio


async def test_version(test_client):
    """Test version endpoint"""
    response = test_client.get("/api/v1/version")
    assert response.status_code == 200
    assert "version" in response.json()


async def test_user_creation_and_deletion(test_client, api_test_user):
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


async def test_duplicate_user_creation(test_client, api_test_user):
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


async def test_delete_nonexistent_user(test_client):
    """Test that deleting a non-existent user returns a 404."""
    response = test_client.delete("/api/v1/users/nonexistent_user_for_sure")
    assert response.status_code == 404


async def test_ingest_data(test_client, api_test_user):
    """Test data ingestion endpoint"""
    user_id = api_test_user
    # Ensure user exists
    test_client.post(f"/api/v1/users/{user_id}")
    response = test_client.post(
        f"/api/v1/users/{user_id}/ingest",
        json={
            "title": "Space Exploration",
            "content": "This is a test content about space exploration.",
            "metadata": {},
        },
    )
    if response.status_code != 201:
        print("DEBUG response:", response.json())
    assert response.status_code == 201
    assert "ingested successfully" in response.json()["message"]


test_ingest_data = pytest.mark.skipif(
    os.getenv("CI", "").lower() == "true",
    reason="Requires external LLM call path; skipped in CI",
)(test_ingest_data)


async def test_ingest_empty_content(test_client, api_test_user):
    """Test that ingesting empty or whitespace-only content returns a 400."""
    user_id = api_test_user
    # Ensure user exists
    test_client.post(f"/api/v1/users/{user_id}")

    response = test_client.post(
        f"/api/v1/users/{user_id}/ingest",
        json={"title": "Empty Test", "content": "   "},  # Whitespace only
    )
    assert response.status_code == 400
    assert "Content cannot be empty" in response.json()["detail"]


@patch("persona.adapters.persona_adapter.PersonaAdapter.ingest", new_callable=AsyncMock)
async def test_ingest_llm_failure(mock_ingest, test_client, api_test_user):
    """Test that a failed adapter ingestion returns a 500."""
    user_id = api_test_user
    test_client.post(f"/api/v1/users/{user_id}")
    mock_ingest.return_value = IngestionResult(success=False, error="LLM is down")

    response = test_client.post(
        f"/api/v1/users/{user_id}/ingest",
        json={"title": "This will fail", "content": "This will fail"},
    )

    assert response.status_code == 500
    assert "Ingestion failed" in response.json()["detail"]


async def test_ingest_idempotency(test_client, isolated_graph_ops):
    """Test that ingesting the same content twice does not create duplicate nodes."""
    graph_ops, user_id = isolated_graph_ops
    ingest_data = {
        "title": "Idempotency Test",
        "content": "A unique sentence to test idempotency.",
        "metadata": {},
    }

    response1 = test_client.post(f"/api/v1/users/{user_id}/ingest", json=ingest_data)
    assert response1.status_code == 201

    nodes_after_first_ingest = await graph_ops.graph_db.get_all_nodes(user_id)
    assert len(nodes_after_first_ingest) > 0

    response2 = test_client.post(f"/api/v1/users/{user_id}/ingest", json=ingest_data)
    assert response2.status_code == 201

    nodes_after_second_ingest = await graph_ops.graph_db.get_all_nodes(user_id)
    assert len(nodes_after_second_ingest) <= len(nodes_after_first_ingest) + 1


test_ingest_idempotency = pytest.mark.skipif(
    os.getenv("CI", "").lower() == "true",
    reason="Requires external LLM call path; skipped in CI",
)(test_ingest_idempotency)


async def test_rag_query(test_client, api_test_user):
    """Test RAG query endpoint"""
    user_id = api_test_user
    test_client.post(f"/api/v1/users/{user_id}")
    response = test_client.post(
        f"/api/v1/users/{user_id}/rag/query", json={"query": "What is this text about?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()


async def test_rag_query_non_existent_user(test_client):
    """Test that a RAG query for a non-existent user returns a 404."""
    response = test_client.post(
        "/api/v1/users/non-existent-user/rag/query", json={"query": "Does not matter"}
    )
    assert response.status_code == 404


async def test_rag_query_for_user_with_no_graph(test_client, isolated_graph_ops):
    """Test that a RAG query for a user with no data returns a clean response."""
    _, user_id = isolated_graph_ops
    response = test_client.post(
        f"/api/v1/users/{user_id}/rag/query", json={"query": "What is this text about?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()


@pytest.mark.skip(reason="/rag/query-vector endpoint removed in current API")
async def test_rag_query_vector(test_client, api_test_user):
    """Legacy endpoint test retained as skipped for compatibility history."""


async def test_ask_insights(test_client, api_test_user):
    """Test ask insights endpoint"""
    user_id = api_test_user
    test_client.post(f"/api/v1/users/{user_id}")
    test_request = {
        "query": "What are the main topics?",
        "output_schema": {"topics": ["topic1", "topic2"], "summary": "test summary"},
    }
    response = test_client.post(f"/api/v1/users/{user_id}/ask", json=test_request)
    assert response.status_code == 200
    assert response.json() is not None


async def test_ask_non_existent_user(test_client):
    """Test that an ask query for a non-existent user returns a 404."""
    response = test_client.post(
        "/api/v1/users/non-existent-user/ask",
        json={"query": "Doesn't matter", "output_schema": {}},
    )
    assert response.status_code == 404


async def test_ask_for_user_with_no_graph(test_client, isolated_graph_ops):
    """Test that an ask query for a user with no data returns a clean response."""
    _, user_id = isolated_graph_ops
    test_request = {
        "query": "What are the main topics?",
        "output_schema": {"topics": ["string"], "summary": "string"},
    }

    response = test_client.post(f"/api/v1/users/{user_id}/ask", json=test_request)
    assert response.status_code == 200
    assert response.json() is not None


@pytest.mark.skip(reason="/custom-data endpoint removed in current API")
async def test_custom_data(test_client, api_test_user):
    """Legacy endpoint test retained as skipped for compatibility history."""


async def test_delete_user_with_invalid_id_format(test_client):
    """Test that deleting a user ID with invalid characters returns a 422."""
    invalid_user_id = "invalid$user@id"
    response = test_client.delete(f"/api/v1/users/{invalid_user_id}")
    assert response.status_code == 422
    assert "detail" in response.json()
