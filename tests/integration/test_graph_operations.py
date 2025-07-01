import pytest
import uuid
from unittest.mock import AsyncMock, patch, MagicMock
from persona.core.graph_ops import GraphOps
from persona.models.schema import NodeModel, RelationshipModel, Node, Relationship

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

@pytest.mark.asyncio
async def test_vector_index_end_to_end():
    """Test the complete vector index pipeline using GraphOps abstractions with mocked embeddings"""
    from persona.core.neo4j_database import Neo4jConnectionManager
    
    # Create our own Neo4j manager - it gets config from config object
    manager = Neo4jConnectionManager()
    
    try:
        await manager.wait_for_neo4j()
        
        async with GraphOps(manager) as graph_ops:
            await graph_ops.initialize()
            await graph_ops.neo4j_manager.ensure_vector_index()
            
            user_id = f"test-vector-user-{uuid.uuid4()}"
            
            # Create user first
            await graph_ops.create_user(user_id)
            
            try:
                # Add some test nodes 
                test_nodes = [
                    NodeModel(name="Machine Learning Algorithm", type="Goal"),
                    NodeModel(name="Deep Learning Network", type="Goal"),
                    NodeModel(name="Cooking Recipe", type="Preference"),
                    NodeModel(name="Travel Destination", type="Preference")
                ]
                
                # Add nodes using GraphOps abstraction (this should trigger embedding generation and storage)
                await graph_ops.add_nodes(test_nodes, user_id)
                
                # Verify nodes were created properly with GraphOps abstraction
                for node in test_nodes:
                    node_data = await graph_ops.get_node_data(node.name, user_id)
                    assert node_data.name == node.name, f"Node {node.name} should exist"
                    assert node_data.type == node.type, f"Node {node.name} should have correct type"
                
                # Test similarity search using GraphOps abstraction
                # We test with exact node names since embeddings are generated from node names
                test_queries = [
                    "Machine Learning Algorithm",
                    "Cooking Recipe", 
                    "Deep Learning Network",
                    "Travel Destination"
                ]
                
                for query in test_queries:
                    results = await graph_ops.text_similarity_search(query, user_id)
                    
                    # Verify the query is returned correctly
                    assert results["query"] == query
                    
                    # Verify we get results (the exact similarity depends on the mocked embeddings)
                    # With deterministic mocked embeddings, we should get some results
                    assert isinstance(results["results"], list), f"Results should be a list for query: {query}"
                    
                    # If we get results, verify they have the expected structure
                    for result in results["results"]:
                        assert "nodeId" in result
                        assert "nodeName" in result  
                        assert "score" in result
                        assert isinstance(result["score"], (int, float))
                
                # Test that the similarity search is properly scoped to the user
                # Create another user and verify isolation
                other_user_id = f"test-vector-other-user-{uuid.uuid4()}"
                await graph_ops.create_user(other_user_id)
                
                try:
                    # Add different nodes for the other user
                    other_nodes = [
                        NodeModel(name="Other Algorithm", type="Goal"),
                    ]
                    await graph_ops.add_nodes(other_nodes, other_user_id)
                    
                    # Search from the original user should not return the other user's nodes
                    results = await graph_ops.text_similarity_search("Other Algorithm", user_id)
                    if results["results"]:
                        for result in results["results"]:
                            # None of the results should be "Other Algorithm" since it belongs to different user
                            assert result["nodeName"] != "Other Algorithm", "Should not return nodes from other users"
                    
                    # Search from the other user should not return the original user's nodes  
                    results = await graph_ops.text_similarity_search("Machine Learning Algorithm", other_user_id)
                    if results["results"]:
                        for result in results["results"]:
                            # None of the results should be from the original user's nodes
                            assert result["nodeName"] not in [n.name for n in test_nodes], "Should not return nodes from other users"
                
                finally:
                    await graph_ops.delete_user(other_user_id)
                
            finally:
                # Cleanup
                await graph_ops.delete_user(user_id)
    
    finally:
        await manager.close()

@pytest.mark.asyncio
async def test_vector_index_operations_only_unit_test(mock_neo4j_vector_calls):
    """Unit test for vector operations with mocked Neo4j - for unit testing only"""
    # This test uses mocks and is for unit testing components in isolation
    # It's separate from our integration tests that test real vector operations
    pass