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
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    
    manager = Neo4jConnectionManager(
        uri=neo4j_uri,
        user=config.NEO4J.USER,
        password=config.NEO4J.PASSWORD
    )
    
    # Test connection and wait for Neo4j
    try:
        await manager.wait_for_neo4j()
    except Exception as e:
        pytest.fail(f"Neo4j connection failed: {str(e)}")
    
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

# OpenAI mocks - these should ALWAYS be active to prevent real API calls
@pytest.fixture(autouse=True)
def mock_openai_clients(monkeypatch):
    """Mock OpenAI client objects and functions - always active to prevent real API calls from any context."""
    
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
    
    # Mock sync OpenAI client for embeddings
    mock_sync_client = MagicMock()
    mock_embeddings = MagicMock()
    
    def mock_create_embeddings(**kwargs):
        texts = kwargs.get('input', [])
        if isinstance(texts, str):
            texts = [texts]
        embeddings = deterministic_embedding(texts)
        return MagicMock(
            data=[MagicMock(embedding=emb) for emb in embeddings]
        )
    
    mock_embeddings.create = mock_create_embeddings
    mock_sync_client.embeddings = mock_embeddings
    
    # Mock async OpenAI client for LLM calls
    mock_async_client = AsyncMock()
    mock_async_completions = AsyncMock()
    mock_async_completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="This is a mocked LLM response from client."))]
    ))
    mock_async_client.chat = MagicMock(completions=mock_async_completions)
    
    # Patch the client instantiations
    monkeypatch.setattr("persona.llm.embeddings.openai_client", mock_sync_client)
    monkeypatch.setattr("persona.llm.llm_graph.openai_client", mock_async_client)

@pytest.fixture(autouse=True)
def mock_llm_graph_calls(monkeypatch):
    """Mock LLM calls for graph construction and querying - always active to prevent real API calls."""
    # Return a dummy node to allow ingestion pipeline to proceed
    dummy_node = Node(name="Dummy Node")
    monkeypatch.setattr("persona.llm.llm_graph.get_nodes", lambda *args, **kwargs: [dummy_node])
    monkeypatch.setattr("persona.llm.llm_graph.get_relationships", lambda *args, **kwargs: [])
    
    # Mock the generate_response_with_context function used by RAG
    monkeypatch.setattr(
        "persona.llm.llm_graph.generate_response_with_context",
        lambda *args, **kwargs: "This is a mocked RAG response based on the provided context."
    )
    
    # Mock the generate_structured_insights function used by Ask service
    def mock_structured_insights(ask_request, context):
        # Return a response that matches the expected schema structure
        return {k: f"mocked_{k}" if isinstance(v, str) else ["mocked_item"] if isinstance(v, list) else {"mocked_key": "mocked_value"} 
                for k, v in ask_request.output_schema.items()}
    
    monkeypatch.setattr(
        "persona.llm.llm_graph.generate_structured_insights", 
        mock_structured_insights
    )

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