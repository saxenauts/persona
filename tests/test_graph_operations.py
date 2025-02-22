import pytest
from persona.core.graph_ops import GraphOps
from persona.models.schema import NodeModel, RelationshipModel
from unittest.mock import AsyncMock

@pytest.fixture
def mock_neo4j_manager():
    # Mock the Neo4j manager
    manager = AsyncMock()
    return manager

@pytest.fixture
def graph_ops(mock_neo4j_manager):
    return GraphOps(mock_neo4j_manager)

@pytest.mark.asyncio
async def test_add_node_with_embedding(graph_ops):
    await graph_ops.add_node_with_embedding(
        "Python",
        "test_user",
        {"description": "Programming language"}
    )
    graph_ops.neo4j_manager.create_node.assert_awaited_with(
        name="Python",
        user_id="test_user",
        properties={"description": "Programming language"},
        embedding=None
    )

@pytest.mark.asyncio
async def test_get_node(graph_ops):
    mock_node = {"name": "Python", "description": "Programming language"}
    graph_ops.neo4j_manager.get_node.return_value = mock_node

    node = await graph_ops.get_node("Python", "test_user")
    assert node == mock_node
    graph_ops.neo4j_manager.get_node.assert_awaited_once()

@pytest.mark.asyncio
async def test_add_relationship(graph_ops):
    await graph_ops.add_relationship(
        "Python",
        "FastAPI",
        "HAS_FRAMEWORK",
        "test_user"
    )
    graph_ops.neo4j_manager.create_relationship.assert_awaited_with(
        source="Python",
        target="FastAPI",
        relation="HAS_FRAMEWORK",
        user_id="test_user"
    )

@pytest.mark.asyncio
async def test_get_relationship(graph_ops):
    mock_relationship = {"source": "Python", "target": "FastAPI", "relation": "HAS_FRAMEWORK"}
    graph_ops.neo4j_manager.get_relationship.return_value = mock_relationship

    rel = await graph_ops.get_relationship("Python", "FastAPI", "HAS_FRAMEWORK", "test_user")
    assert rel == mock_relationship
    graph_ops.neo4j_manager.get_relationship.assert_awaited_once()