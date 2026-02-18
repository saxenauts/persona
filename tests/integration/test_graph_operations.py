import pytest
import uuid
from unittest.mock import AsyncMock, patch, MagicMock
from persona.core import GraphOps
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.core.backends.neo4j_vector import Neo4jVectorStore
from persona.models.schema import NodeModel, RelationshipModel, Node, Relationship


@pytest.fixture
async def mock_graph_ops():
    mock = AsyncMock()
    mock.user_exists = AsyncMock(return_value=True)
    mock.add_nodes = AsyncMock()
    mock.add_nodes_batch_embeddings = AsyncMock()
    mock.add_relationships = AsyncMock()
    mock.get_node_data = AsyncMock()
    mock.text_similarity_search = AsyncMock()
    mock.graph_db = AsyncMock()
    mock.vector_store = AsyncMock()
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
    
    mock_graph_db = AsyncMock()
    mock_graph_db.user_exists = AsyncMock(return_value=True)
    mock_graph_db.create_nodes = AsyncMock()
    mock_graph_db.initialize = AsyncMock()
    mock_graph_db.close = AsyncMock()
    
    mock_vector = AsyncMock()
    mock_vector.add_embedding = AsyncMock()
    mock_vector.initialize = AsyncMock()
    mock_vector.close = AsyncMock()
    
    with patch('persona.core.graph_ops.generate_embeddings_async', return_value=[[0.1] * 1536]):
        graph_ops = GraphOps(graph_db=mock_graph_db, vector_store=mock_vector)
        await graph_ops.add_nodes(nodes, "test_user")
        
        assert mock_graph_db.user_exists.call_count >= 1
        mock_graph_db.create_nodes.assert_called()


@pytest.mark.asyncio
async def test_add_relationships():
    relationships = [
        RelationshipModel(
            source="Python",
            target="FastAPI",
            relation="HAS_FRAMEWORK"
        )
    ]
    
    mock_graph_db = AsyncMock()
    mock_graph_db.user_exists = AsyncMock(return_value=True)
    mock_graph_db.create_relationships = AsyncMock()
    mock_graph_db.initialize = AsyncMock()
    mock_graph_db.close = AsyncMock()
    
    mock_vector = AsyncMock()
    mock_vector.initialize = AsyncMock()
    mock_vector.close = AsyncMock()
    
    graph_ops = GraphOps(graph_db=mock_graph_db, vector_store=mock_vector)
    await graph_ops.add_relationships(relationships, "test_user")
    
    mock_graph_db.user_exists.assert_called_once_with("test_user")
    mock_graph_db.create_relationships.assert_called_once()


@pytest.mark.asyncio
async def test_get_node_data():
    mock_node = {"name": "Python", "properties": {"description": "Programming language"}}
    
    mock_graph_db = AsyncMock()
    mock_graph_db.user_exists = AsyncMock(return_value=True)
    mock_graph_db.get_node = AsyncMock(return_value=mock_node)
    mock_graph_db.initialize = AsyncMock()
    mock_graph_db.close = AsyncMock()
    
    mock_vector = AsyncMock()
    mock_vector.initialize = AsyncMock()
    mock_vector.close = AsyncMock()
    
    graph_ops = GraphOps(graph_db=mock_graph_db, vector_store=mock_vector)
    result = await graph_ops.get_node_data("Python", "test_user")
    
    mock_graph_db.user_exists.assert_called_once_with("test_user")
    mock_graph_db.get_node.assert_called_once_with("Python", "test_user")
    assert isinstance(result, NodeModel)
    assert result.name == "Python"


@pytest.mark.asyncio
async def test_text_similarity_search():
    mock_results = [{"node_name": "test", "score": 0.9}]
    
    mock_graph_db = AsyncMock()
    mock_graph_db.user_exists = AsyncMock(return_value=True)
    mock_graph_db.initialize = AsyncMock()
    mock_graph_db.close = AsyncMock()
    
    mock_vector = AsyncMock()
    mock_vector.search_similar = AsyncMock(return_value=mock_results)
    mock_vector.initialize = AsyncMock()
    mock_vector.close = AsyncMock()
    
    with patch('persona.core.graph_ops.generate_embeddings_async', return_value=[[0.1, 0.2, 0.3]]):
        graph_ops = GraphOps(graph_db=mock_graph_db, vector_store=mock_vector)
        result = await graph_ops.text_similarity_search("test query", "test_user")
        
        mock_graph_db.user_exists.assert_called_once_with("test_user")
        mock_vector.search_similar.assert_called_once()
        assert "results" in result
        assert result["results"] == [{"nodeName": "test", "score": 0.9}]


@pytest.mark.asyncio
async def test_vector_index_end_to_end():
    """Test the complete vector index pipeline using GraphOps with real backends."""
    
    graph_db = Neo4jGraphDatabase()
    vector_store = Neo4jVectorStore()
    
    try:
        await graph_db.initialize()
        await vector_store.initialize()
        
        graph_ops = GraphOps(graph_db=graph_db, vector_store=vector_store)
        
        user_id = f"test-vector-user-{uuid.uuid4()}"
        await graph_ops.create_user(user_id)
        
        try:
            # Add test nodes 
            test_nodes = [
                NodeModel(name="Machine Learning Algorithm", type="Goal"),
                NodeModel(name="Deep Learning Network", type="Goal"),
                NodeModel(name="Cooking Recipe", type="Preference"),
                NodeModel(name="Travel Destination", type="Preference")
            ]
            
            await graph_ops.add_nodes(test_nodes, user_id)
            
            # Verify nodes were created
            for node in test_nodes:
                node_data = await graph_ops.get_node_data(node.name, user_id)
                assert node_data.name == node.name
                assert node_data.type == node.type
            
            # Test similarity search
            test_queries = [
                "Machine Learning Algorithm",
                "Cooking Recipe", 
            ]
            
            for query in test_queries:
                results = await graph_ops.text_similarity_search(query, user_id)
                assert results["query"] == query
                assert isinstance(results["results"], list)
                
                for result in results["results"]:
                    assert "nodeName" in result  
                    assert "score" in result
            
            # Test user isolation
            other_user_id = f"test-vector-other-user-{uuid.uuid4()}"
            await graph_ops.create_user(other_user_id)
            
            try:
                other_nodes = [NodeModel(name="Other Algorithm", type="Goal")]
                await graph_ops.add_nodes(other_nodes, other_user_id)
                
                # Search from original user should not return other user's nodes
                results = await graph_ops.text_similarity_search("Other Algorithm", user_id)
                if results["results"]:
                    for result in results["results"]:
                        assert result["nodeName"] != "Other Algorithm"
            finally:
                await graph_ops.delete_user(other_user_id)
                
        finally:
            await graph_ops.delete_user(user_id)
    
    finally:
        await graph_db.close()
        await vector_store.close()


@pytest.mark.asyncio
async def test_vector_index_operations_only_unit_test(mock_vector_store):
    """Unit test for vector operations with mocked backend."""
    pass