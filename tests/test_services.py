import pytest
from unittest.mock import AsyncMock
from persona.services.learn_service import LearnService
from persona.services.custom_data_service import CustomDataService
from persona.core.graph_ops import GraphOps
from persona.models.schema import LearnRequest, GraphSchema, CustomGraphUpdate, CustomNodeData, CustomRelationshipData, NodesAndRelationshipsResponse, NodeModel, RelationshipModel

@pytest.fixture
def mock_graph_ops():
    return AsyncMock(spec=GraphOps)

@pytest.fixture
def learn_service(mock_graph_ops):
    return LearnService(mock_graph_ops)

@pytest.fixture
def custom_data_service(mock_graph_ops):
    return CustomDataService(mock_graph_ops)

@pytest.mark.asyncio
async def test_learn_user_success(learn_service, mock_graph_ops):
    test_schema = GraphSchema(
        name="Food Preferences",
        description="Learn about user's food preferences and eating patterns",
        attributes=[
            'FAVORITE_CUISINE',
            'DIETARY_RESTRICTION',
            'MEAL_TIMING',
            'FLAVOR_PREFERENCE'
        ],
        relationships=[
            'PAIRS_WELL_WITH',
            'AVOIDS_WITH',
            'PREFERS_BEFORE',
            'PREFERS_AFTER'
        ]
    )
    learn_request = LearnRequest(
        user_id="test_user",
        graph_schema=test_schema,
        description="Understanding user's food preferences and eating patterns"
    )
    mock_graph_ops.store_schema.return_value = "schema123"

    response = await learn_service.learn_user(learn_request)

    assert response.status == "Success"
    assert response.schema_id == "schema123"
    mock_graph_ops.store_schema.assert_awaited_once_with(test_schema)

@pytest.mark.asyncio
async def test_learn_user_failure(learn_service, mock_graph_ops):
    test_schema = GraphSchema(
        name="Food Preferences",
        description="Learn about user's food preferences and eating patterns",
        attributes=[
            'FAVORITE_CUISINE',
            'DIETARY_RESTRICTION',
            'MEAL_TIMING',
            'FLAVOR_PREFERENCE'
        ],
        relationships=[
            'PAIRS_WELL_WITH',
            'AVOIDS_WITH',
            'PREFERS_BEFORE',
            'PREFERS_AFTER'
        ]
    )
    learn_request = LearnRequest(
        user_id="test_user",
        graph_schema=test_schema,
        description="Understanding user's food preferences and eating patterns"
    )
    mock_graph_ops.store_schema.side_effect = Exception("Database error")

    response = await learn_service.learn_user(learn_request)

    assert response["status"] == "error"
    assert response["message"] == "Database error"
    mock_graph_ops.store_schema.assert_awaited_once_with(test_schema)

@pytest.mark.asyncio
async def test_update_custom_data_success(custom_data_service, mock_graph_ops):
    test_update = CustomGraphUpdate(
        nodes=[
            CustomNodeData(
                name="current_gaming_preference",
                properties={
                    "favorite_genre": "RPG",
                    "current_game": "Baldur's Gate 3",
                    "hours_played": 120,
                    "last_played": "2024-03-15T14:30:00Z"
                }
            )
        ],
        relationships=[
            CustomRelationshipData(
                source="current_gaming_preference",
                target="test_user",
                relation_type="PREFERENCE_OF",
                data={}
            )
        ]
    )
    mock_graph_ops.update_graph.return_value = None

    response = await custom_data_service.update_custom_data("test_user", test_update)

    assert response["status"] == "success"
    assert "Updated 1 nodes and 1 relationships" in response["message"]

    expected_nodes = [
        {
            "name": "current_gaming_preference",
            "perspective": None,
            "properties": {
                "favorite_genre": "RPG",
                "current_game": "Baldur's Gate 3",
                "hours_played": 120,
                "last_played": "2024-03-15T14:30:00Z"
            },
            "embedding": None
        }
    ]
    expected_relationships = [
        RelationshipModel(
            source="current_gaming_preference",
            target="test_user",
            relation="PREFERENCE_OF"
        )
    ]
    mock_graph_ops.update_graph.assert_awaited_once_with(
        NodesAndRelationshipsResponse(
            nodes= [NodeModel(**node) for node in expected_nodes],
            relationships=expected_relationships
        ),
        "test_user"
    )

@pytest.mark.asyncio
async def test_update_custom_data_failure(custom_data_service, mock_graph_ops):
    test_update = CustomGraphUpdate(
        nodes=[
            CustomNodeData(
                name="current_gaming_preference",
                properties={
                    "favorite_genre": "RPG",
                    "current_game": "Baldur's Gate 3",
                    "hours_played": 120,
                    "last_played": "2024-03-15T14:30:00Z"
                }
            )
        ],
        relationships=[
            CustomRelationshipData(
                source="current_gaming_preference",
                target="test_user",
                relation_type="PREFERENCE_OF",
                data={}
            )
        ]
    )
    mock_graph_ops.update_graph.side_effect = Exception("Update failed")

    response = await custom_data_service.update_custom_data("test_user", test_update)

    assert response["status"] == "error"
    assert response["message"] == "Update failed"
    mock_graph_ops.update_graph.assert_awaited_once_with(
        NodesAndRelationshipsResponse(
            nodes=[
                NodeModel(
                    name="current_gaming_preference",
                    perspective=None,
                    properties={
                        "favorite_genre": "RPG",
                        "current_game": "Baldur's Gate 3",
                        "hours_played": 120,
                        "last_played": "2024-03-15T14:30:00Z"
                    },
                    embedding=None
                )
            ],
            relationships=[
                RelationshipModel(
                    source="current_gaming_preference",
                    target="test_user",
                    relation="PREFERENCE_OF"
                )
            ]
        ),
        "test_user"
    )