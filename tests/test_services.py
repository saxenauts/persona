import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from persona.services.custom_data_service import CustomDataService
from persona.services.user_service import UserService
from persona.services.ingest_service import IngestService
from persona.services.rag_service import RAGService
from persona.services.ask_service import AskService
from persona.core.graph_ops import GraphOps
from persona.models.schema import (
    CustomGraphUpdate,
    CustomNodeData,
    CustomRelationshipData,
    NodeModel,
    AskRequest
)

@pytest.fixture
async def mock_graph_ops():
    mock = AsyncMock(spec=GraphOps)
    mock.__aenter__.return_value = mock
    mock.__aexit__.return_value = None
    return mock

@pytest.fixture
def custom_data_service(mock_graph_ops):
    return CustomDataService(mock_graph_ops)

@pytest.fixture
def user_service(mock_graph_ops):
    return UserService(mock_graph_ops)

@pytest.fixture
def ingest_service(mock_graph_ops):
    return IngestService(mock_graph_ops)

@pytest.fixture
def rag_service(mock_graph_ops):
    return RAGService(mock_graph_ops)

@pytest.fixture
def ask_service(mock_graph_ops):
    return AskService(mock_graph_ops)


@pytest.mark.asyncio
async def test_create_user_success():
    with patch('persona.core.graph_ops.GraphOps.__aenter__') as mock_graph_ops:
        mock_graph_ops.return_value.create_user = AsyncMock()
        
        result = await UserService.create_user("test_user")
        assert result["message"] == "User test_user created successfully"
        mock_graph_ops.return_value.create_user.assert_called_once_with("test_user")

@pytest.mark.asyncio
async def test_delete_user_success():
    with patch('persona.core.graph_ops.GraphOps.__aenter__') as mock_graph_ops:
        mock_graph_ops.return_value.delete_user = AsyncMock()
        
        result = await UserService.delete_user("test_user")
        assert result["message"] == "User test_user deleted successfully"
        mock_graph_ops.return_value.delete_user.assert_called_once_with("test_user")

@pytest.mark.asyncio
async def test_ingest_data_success():
    # Create a mock for GraphOps
    mock_graph_ops = AsyncMock()
    mock_graph_ops.__aenter__ = AsyncMock(return_value=mock_graph_ops)
    mock_graph_ops.__aexit__ = AsyncMock()

    # Create a mock for the constructor
    mock_constructor = AsyncMock()
    mock_constructor.ingest_unstructured_data_to_graph = AsyncMock()
    mock_constructor.graph_ops = mock_graph_ops
    mock_constructor.__aenter__ = AsyncMock(return_value=mock_constructor)
    mock_constructor.__aexit__ = AsyncMock()

    # Mock the GraphConstructor class
    with patch('persona.services.ingest_service.GraphConstructor', return_value=mock_constructor):
        # Call the service method
        result = await IngestService.ingest_data("test_user", "Test content")
        
        # Verify the constructor was used correctly
        mock_constructor.ingest_unstructured_data_to_graph.assert_called_once()
        call_args = mock_constructor.ingest_unstructured_data_to_graph.call_args[0][0]
        assert call_args.title == "Ingested Data"
        assert call_args.content == "Test content"
        assert result["message"] == "Data ingested successfully"

@pytest.mark.asyncio
async def test_rag_query_success():
    with patch('persona.core.rag_interface.RAGInterface.__aenter__') as mock_rag:
        mock_rag.return_value.query = AsyncMock(return_value="Test response")
        result = await RAGService.query("test_user", "Test query")
        assert isinstance(result, str)
        assert result == "Test response"

@pytest.mark.asyncio
async def test_ask_insights_success():
    test_request = AskRequest(
        query="What are the preferences?",
        output_schema={
            "preferences": ["test"],
            "summary": "test summary"
        }
    )

    with patch('persona.core.rag_interface.RAGInterface.__aenter__') as mock_rag:
        mock_rag.return_value.get_context = AsyncMock(return_value="test context")
        mock_rag.return_value.query = AsyncMock(return_value="test response")

        response = await AskService.ask_insights("test_user", test_request)

    assert response is not None
    assert hasattr(response, 'result')