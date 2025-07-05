import os
import pytest
import asyncio
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from server.main import app
from persona.core.neo4j_database import Neo4jConnectionManager
from persona.core.graph_ops import GraphOps
from server.config import config
from persona.models.schema import Node

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_client():
    # Use TestClient context manager to trigger FastAPI lifespan events
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="session")
async def neo4j_manager():
    manager = Neo4jConnectionManager()
    await manager.initialize()
    
    yield manager
    
    await manager.close()

@pytest.fixture(scope="function")
async def test_user(neo4j_manager):
    user_id = f"test-user-{uuid.uuid4()}"
    # Clean up just in case of leftovers from a failed run
    await neo4j_manager.run_query("MATCH (u:User {id: $user_id}) DETACH DELETE u", {"user_id": user_id})
    
    yield user_id
    
    # Cleanup after test
    await neo4j_manager.run_query("MATCH (u:User {id: $user_id}) DETACH DELETE u", {"user_id": user_id})

@pytest.fixture(scope="function")
def api_test_user():
    """Sync fixture for API tests that provides a user ID without Neo4j cleanup."""
    return f"test-user-{uuid.uuid4()}"

# LLM Client mocks - these should ALWAYS be active to prevent real API calls
@pytest.fixture(autouse=True)
def mock_llm_clients(monkeypatch):
    """Mock LLM client objects and functions - always active to prevent real API calls from any context."""
    
    def deterministic_embedding(texts):
        """Generate deterministic, realistic embeddings based on text content"""
        embeddings = []
        for text in texts:
            # Create a deterministic but realistic embedding based on text hash
            import hashlib
            text_hash = hashlib.md5(text.encode()).digest()
            
            # Convert hash to floats between -1 and 1
            embedding = []
            for i in range(1536):
                byte_idx = i % len(text_hash)
                # Normalize byte value to [-1, 1] range
                normalized = (text_hash[byte_idx] / 255.0) * 2 - 1
                embedding.append(float(normalized))
            
            embeddings.append(embedding)
        
        return embeddings
    
    # Mock the generate_embeddings function used throughout the system
    monkeypatch.setattr(
        "persona.core.graph_ops.generate_embeddings",
        deterministic_embedding
    )
    
    # Mock the new LLM client system
    from persona.llm.providers.base import ChatResponse
    
    # Create mock chat client
    mock_chat_client = AsyncMock()
    mock_chat_client.chat = AsyncMock(return_value=ChatResponse(
        content="This is a mocked LLM response from the new client system.",
        model="mock-model"
    ))
    
    # Create mock embedding client
    mock_embedding_client = AsyncMock()
    mock_embedding_client.embeddings = AsyncMock(side_effect=lambda texts: deterministic_embedding(texts))
    
    # Mock the client factory functions
    monkeypatch.setattr("persona.llm.client_factory.get_chat_client", lambda: mock_chat_client)
    monkeypatch.setattr("persona.llm.client_factory.get_embedding_client", lambda: mock_embedding_client)

@pytest.fixture(autouse=True)
def mock_llm_graph_calls(monkeypatch):
    """Mock LLM calls for graph construction and querying - always active to prevent real API calls."""
    # Return a dummy node to allow ingestion pipeline to proceed
    from persona.llm.llm_graph import Node
    dummy_node = Node(name="Dummy Node", type="Test")
    
    async def mock_get_nodes(*args, **kwargs):
        return [dummy_node]
    
    async def mock_get_relationships(*args, **kwargs):
        return ([], {})
    
    async def mock_generate_response_with_context(*args, **kwargs):
        return "This is a mocked RAG response based on the provided context."
    
    async def mock_generate_structured_insights(ask_request, context):
        # Return a response that matches the expected schema structure
        return {k: f"mocked_{k}" if isinstance(v, str) else ["mocked_item"] if isinstance(v, list) else {"mocked_key": "mocked_value"} 
                for k, v in ask_request.output_schema.items()}
    
    monkeypatch.setattr("persona.llm.llm_graph.get_nodes", mock_get_nodes)
    monkeypatch.setattr("persona.llm.llm_graph.get_relationships", mock_get_relationships)
    monkeypatch.setattr("persona.llm.llm_graph.generate_response_with_context", mock_generate_response_with_context)
    monkeypatch.setattr("persona.llm.llm_graph.generate_structured_insights", mock_generate_structured_insights)

# Neo4j mocks - only use these in unit tests where we want to isolate components
@pytest.fixture
def mock_neo4j_vector_calls(monkeypatch):
    """Mock Neo4j vector index calls - use this fixture explicitly in unit tests only."""
    async def mock_add_embedding(*args, **kwargs):
        return
    
    async def mock_query_similarity(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "persona.core.neo4j_database.Neo4jConnectionManager.add_embedding_to_vector_index",
        mock_add_embedding
    )
    monkeypatch.setattr(
        "persona.core.neo4j_database.Neo4jConnectionManager.query_text_similarity",
        mock_query_similarity
    )

@pytest.fixture
def mock_neo4j_manager(monkeypatch):
    """Mock the entire Neo4j manager - use this fixture explicitly in unit tests."""
    mock_manager = MagicMock()
    mock_manager.user_exists = MagicMock(return_value=True)
    mock_manager.create_nodes = MagicMock()
    mock_manager.create_relationships = MagicMock()
    mock_manager.get_node_data = MagicMock(return_value={"name": "test", "properties": {}})
    mock_manager.get_all_nodes = MagicMock(return_value=[])
    mock_manager.get_all_relationships = MagicMock(return_value=[])
    mock_manager.add_embedding_to_vector_index = MagicMock()
    mock_manager.query_text_similarity = MagicMock(return_value=[])
    
    return mock_manager

@pytest.fixture
async def integration_graph_ops(neo4j_manager):
    """
    A GraphOps for integration tests that uses real Neo4j operations including vector index.
    This allows us to test the full pipeline including embeddings and similarity search.
    """
    # Create a GraphOps instance with the real Neo4j manager
    graph_ops = GraphOps(neo4j_manager)
    await graph_ops.initialize()
    
    # Ensure vector index exists for testing
    await graph_ops.neo4j_manager.ensure_vector_index()
    
    yield graph_ops
    
    # Clean up
    await graph_ops.close()

@pytest.fixture(scope="function")
async def isolated_graph_ops(neo4j_manager):
    """
    Provides a GraphOps instance and a unique user_id for isolated integration tests.
    Creates the user before the test and guarantees cleanup of the user and all 
    their associated data after the test.
    
    Yields:
        tuple[GraphOps, str]: A tuple containing the GraphOps instance and the user_id.
    """
    user_id = f"test-user-{uuid.uuid4()}"
    
    # Correctly resolve the async generator fixture
    async for manager in neo4j_manager:
        graph_ops = GraphOps(manager)
        await graph_ops.create_user(user_id)
        
        try:
            yield graph_ops, user_id
        finally:
            # This will delete the user node and all associated data
            await graph_ops.delete_user(user_id)