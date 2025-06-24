import pytest
from persona.core.graph_ops import GraphOps
from persona.services.ask_service import AskService
from persona.services.custom_data_service import CustomDataService
from persona.models.schema import (
    CustomGraphUpdate,
    CustomNodeData,
    CustomRelationshipData,
    AskRequest
)

@pytest.mark.asyncio
async def test_constructor_flow(mock_neo4j_vector_calls):
    """Test the complete flow of constructing the graph with real Neo4j operations"""
    # Use mock for vector operations but real Neo4j for graph operations
    async with GraphOps() as graph_ops:
        user_id = "test-constructor-user"
        
        # Create user first
        await graph_ops.create_user(user_id)
        
        # Add test nodes directly via Neo4j (real operation)
        nodes = [
            {
                "name": "Python",
                "properties": {"description": "Programming language"}
            }
        ]
        await graph_ops.neo4j_manager.create_nodes(nodes, user_id)
        
        # Verify nodes were added (real Neo4j query)
        node = await graph_ops.neo4j_manager.get_node_data("Python", user_id)
        assert node is not None
        assert node["properties"]["description"] == "Programming language"
        
        # Cleanup
        await graph_ops.delete_user(user_id)

@pytest.mark.asyncio
async def test_ask_flow(mock_neo4j_vector_calls):
    """Test the complete flow of asking insights from the graph"""
    async with GraphOps() as graph_ops:
        user_id = "test-ask-user"
        
        # Create user first
        await graph_ops.create_user(user_id)
        
        test_request = AskRequest(
            query="What programming languages are there?",
            output_schema={
                "languages": ["Python"],
                "summary": "Found Python programming language"
            }
        )
        
        # This demonstrates our hybrid approach:
        # - OpenAI calls are mocked (no real API calls)
        # - Neo4j graph operations are real
        # - Vector operations are mocked (to avoid embedding issues)
        response = await AskService.ask_insights(user_id, test_request, graph_ops)
        assert response is not None
        
        # Cleanup
        await graph_ops.delete_user(user_id)

@pytest.mark.asyncio
async def test_custom_data_flow(mock_neo4j_vector_calls):
    """Test the flow of adding and retrieving custom structured data"""
    async with GraphOps() as graph_ops:
        user_id = "test-custom-data-user"
        
        # Create user first
        await graph_ops.create_user(user_id)
        
        custom_service = CustomDataService(graph_ops)
        
        test_update = CustomGraphUpdate(
            nodes=[
                CustomNodeData(
                    name="current_gaming_preference",
                    properties={
                        "favorite_genre": "RPG",
                        "current_game": "Baldur's Gate 3",
                        "hours_played": "120",
                        "last_played": "2024-03-15T14:30:00Z"
                    }
                )
            ],
            relationships=[
                CustomRelationshipData(
                    source="current_gaming_preference",
                    target=user_id,
                    relation_type="PREFERENCE_OF"
                )
            ]
        )
        
        response = await custom_service.update_custom_data(user_id, test_update)
        assert response["status"] == "success"
        
        # Verify the node was actually created in Neo4j (real query)
        node_data = await graph_ops.neo4j_manager.get_node_data("current_gaming_preference", user_id)
        assert node_data is not None
        assert node_data["properties"]["favorite_genre"] == "RPG"
        
        # Cleanup
        await graph_ops.delete_user(user_id) 