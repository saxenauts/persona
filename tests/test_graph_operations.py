import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from persona.core.graph_ops import GraphOps
from persona.models.schema import NodeModel, RelationshipModel

@pytest.fixture
async def graph_ops():
    mock = AsyncMock()
    # Mock user_exists to return True by default
    mock.user_exists = AsyncMock(return_value=True)
    mock.add_nodes = AsyncMock()
    mock.add_nodes_batch_embeddings = AsyncMock()
    mock.add_relationships = AsyncMock()
    mock.get_node_data = AsyncMock()
    mock.text_similarity_search = AsyncMock()
    mock.neo4j_manager = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock()
    mock.initialize = AsyncMock()
    return mock

@pytest.mark.asyncio
async def test_add_nodes():
    nodes = [
        NodeModel(
            name="Python",
            properties={"description": "Programming language"}
        )
    ]
    
    with patch('persona.core.neo4j_database.Neo4jConnectionManager') as mock_manager:
        mock_manager.return_value = AsyncMock()
        mock_manager.return_value.user_exists = AsyncMock(return_value=True)
        mock_manager.return_value.create_nodes = AsyncMock()
        mock_manager.return_value.add_embedding_to_vector_index = AsyncMock()
        
        async with GraphOps(mock_manager.return_value) as graph_ops:
            await graph_ops.add_nodes(nodes, "test_user")
            # user_exists is called multiple times due to the implementation
            assert mock_manager.return_value.user_exists.call_count >= 1
            assert all(call.args[0] == "test_user" for call in mock_manager.return_value.user_exists.call_args_list)
            mock_manager.return_value.create_nodes.assert_called()

@pytest.mark.asyncio
async def test_add_relationships():
    relationships = [
        RelationshipModel(
            source="Python",
            target="FastAPI",
            relation="HAS_FRAMEWORK"
        )
    ]
    
    with patch('persona.core.neo4j_database.Neo4jConnectionManager') as mock_manager:
        mock_manager.return_value = AsyncMock()
        mock_manager.return_value.user_exists = AsyncMock(return_value=True)
        mock_manager.return_value.create_relationships = AsyncMock()
        
        async with GraphOps(mock_manager.return_value) as graph_ops:
            await graph_ops.add_relationships(relationships, "test_user")
            mock_manager.return_value.user_exists.assert_called_once_with("test_user")
            mock_manager.return_value.create_relationships.assert_called_once_with([rel.model_dump() for rel in relationships], "test_user")

@pytest.mark.asyncio
async def test_get_node_data():
    mock_node = {"name": "Python", "properties": {"description": "Programming language"}}
    
    with patch('persona.core.neo4j_database.Neo4jConnectionManager') as mock_manager:
        mock_manager.return_value = AsyncMock()
        mock_manager.return_value.user_exists = AsyncMock(return_value=True)
        mock_manager.return_value.get_node_data = AsyncMock(return_value=mock_node)
        
        async with GraphOps(mock_manager.return_value) as graph_ops:
            result = await graph_ops.get_node_data("Python", "test_user")
            mock_manager.return_value.user_exists.assert_called_once_with("test_user")
            mock_manager.return_value.get_node_data.assert_called_once_with("Python", "test_user")
            assert isinstance(result, NodeModel)
            assert result.name == "Python"

@pytest.mark.asyncio
async def test_text_similarity_search():
    mock_results = [{"nodeId": 1, "nodeName": "test", "score": 0.9}]
    
    with patch('persona.core.neo4j_database.Neo4jConnectionManager') as mock_manager, \
         patch('persona.llm.embeddings.generate_embeddings') as mock_embeddings:
        mock_manager.return_value = AsyncMock()
        mock_manager.return_value.user_exists = AsyncMock(return_value=True)
        mock_manager.return_value.query_text_similarity = AsyncMock(return_value=mock_results)
        mock_embeddings.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding vector
        
        async with GraphOps(mock_manager.return_value) as graph_ops:
            result = await graph_ops.text_similarity_search("test query", "test_user")
            mock_manager.return_value.user_exists.assert_called_once_with("test_user")
            mock_manager.return_value.query_text_similarity.assert_called_once()
            assert "results" in result
            assert result["results"] == [{"nodeId": 1, "nodeName": "test", "score": 0.9}]