import os
import pytest
import asyncio
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from server.main import app
from persona.core import GraphOps
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.core.backends.neo4j_vector import Neo4jVectorStore
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
async def graph_db():
    """Session-scoped GraphDatabase for tests."""
    db = Neo4jGraphDatabase()
    await db.initialize()
    yield db
    await db.close()

@pytest.fixture(scope="session")
async def vector_store(graph_db):
    """Session-scoped VectorStore that shares the graph connection."""
    store = Neo4jVectorStore(graph_driver=graph_db.driver)
    await store.initialize()
    yield store
    await store.close()

@pytest.fixture(scope="function")
async def test_user(graph_db):
    user_id = f"test-user-{uuid.uuid4()}"
    yield user_id
    # Cleanup after test
    await graph_db.delete_user(user_id)

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
def mock_vector_store(monkeypatch):
    """Mock VectorStore calls - use this fixture explicitly in unit tests only."""
    async def mock_add_embedding(*args, **kwargs):
        return
    
    async def mock_search_similar(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "persona.core.backends.neo4j_vector.Neo4jVectorStore.add_embedding",
        mock_add_embedding
    )
    monkeypatch.setattr(
        "persona.core.backends.neo4j_vector.Neo4jVectorStore.search_similar",
        mock_search_similar
    )

@pytest.fixture
def mock_graph_db(monkeypatch):
    """Mock the entire GraphDatabase - use this fixture explicitly in unit tests."""
    mock_db = MagicMock()
    mock_db.user_exists = AsyncMock(return_value=True)
    mock_db.create_nodes = AsyncMock()
    mock_db.create_relationships = AsyncMock()
    mock_db.get_node = AsyncMock(return_value={"name": "test", "properties": {}})
    mock_db.get_all_nodes = AsyncMock(return_value=[])
    mock_db.get_all_relationships = AsyncMock(return_value=[])
    return mock_db

@pytest.fixture
async def integration_graph_ops(graph_db, vector_store):
    """
    A GraphOps for integration tests that uses real Neo4j operations including vector index.
    This allows us to test the full pipeline including embeddings and similarity search.
    """
    graph_ops = GraphOps(graph_db=graph_db, vector_store=vector_store)
    yield graph_ops

@pytest.fixture(scope="function")
async def isolated_graph_ops(graph_db, vector_store):
    """
    Provides a GraphOps instance and a unique user_id for isolated integration tests.
    Creates the user before the test and guarantees cleanup of the user and all 
    their associated data after the test.
    
    Yields:
        tuple[GraphOps, str]: A tuple containing the GraphOps instance and the user_id.
    """
    user_id = f"test-user-{uuid.uuid4()}"
    
    graph_ops = GraphOps(graph_db=graph_db, vector_store=vector_store)
    await graph_ops.create_user(user_id)
    
    try:
        yield graph_ops, user_id
    finally:
        # This will delete the user node and all associated data
        await graph_ops.delete_user(user_id)