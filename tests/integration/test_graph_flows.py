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
async def test_constructor_flow():
    """Test the complete flow of constructing the graph"""
    async with GraphOps() as graph_ops:
        # Create user first
        await graph_ops.create_user("test_user")
        
        # Add test nodes
        nodes = [
            {
                "name": "Python",
                "properties": {"description": "Programming language"}
            }
        ]
        await graph_ops.neo4j_manager.create_nodes(nodes, "test_user")
        
        # Verify nodes were added
        node = await graph_ops.neo4j_manager.get_node_data("Python", "test_user")
        assert node is not None
        assert node["properties"]["description"] == "Programming language"

@pytest.mark.asyncio
async def test_ask_flow():
    """Test the complete flow of asking insights from the graph"""
    async with GraphOps() as graph_ops:
        test_request = AskRequest(
            query="What programming languages are there?",
            output_schema={
                "languages": ["Python"],
                "summary": "Found Python programming language"
            }
        )
        
        response = await AskService.ask_insights("test_user", test_request, graph_ops)
        assert response is not None

@pytest.mark.asyncio
async def test_custom_data_flow():
    """Test the flow of adding and retrieving custom structured data"""
    async with GraphOps() as graph_ops:
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
                    target="test_user",
                    relation_type="PREFERENCE_OF"
                )
            ]
        )
        
        response = await custom_service.update_custom_data("test_user", test_update)
        assert response["status"] == "success" 