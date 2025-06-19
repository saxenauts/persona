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
    AskRequest,
    UnstructuredData
)

@pytest.fixture
def mock_graph_ops():
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
async def test_create_user_success(mock_graph_ops):
    mock_graph_ops.create_user = AsyncMock()
    
    result = await UserService.create_user("test_user", mock_graph_ops)
    assert result["message"] == "User test_user created successfully"
    mock_graph_ops.create_user.assert_called_once_with("test_user")

@pytest.mark.asyncio
async def test_delete_user_success(mock_graph_ops):
    mock_graph_ops.delete_user = AsyncMock()
    
    result = await UserService.delete_user("test_user", mock_graph_ops)
    assert result["message"] == "User test_user deleted successfully"
    mock_graph_ops.delete_user.assert_called_once_with("test_user")

@pytest.mark.asyncio
async def test_ingest_data_success(mock_graph_ops):
    # Create a mock for the constructor
    mock_constructor = AsyncMock()
    mock_constructor.ingest_unstructured_data_to_graph = AsyncMock()
    mock_constructor.graph_ops = mock_graph_ops
    mock_constructor.__aenter__ = AsyncMock(return_value=mock_constructor)
    mock_constructor.__aexit__ = AsyncMock()

    # Mock the GraphConstructor class
    with patch('persona.services.ingest_service.GraphConstructor', return_value=mock_constructor):
        # Call the service method
        content = UnstructuredData(title="Test", content="Test content")
        result = await IngestService.ingest_data("test_user", content, mock_graph_ops)
        
        # Verify the constructor was used correctly
        mock_constructor.ingest_unstructured_data_to_graph.assert_called_once()
        call_args = mock_constructor.ingest_unstructured_data_to_graph.call_args[0][0]
        assert call_args.title == "Test"
        assert call_args.content == "Test content"
        assert result["message"] == "Data ingested successfully"

@pytest.mark.asyncio
async def test_rag_query_success(mock_graph_ops):
    # Create a mock for RAGInterface
    mock_rag = AsyncMock()
    mock_rag.query = AsyncMock(return_value="Test response")
    mock_rag.graph_ops = mock_graph_ops
    
    with patch('persona.services.rag_service.RAGInterface', return_value=mock_rag):
        result = await RAGService.query("test_user", "Test query", mock_graph_ops)
        assert isinstance(result, str)
        assert result == "Test response"

@pytest.mark.asyncio
async def test_ask_insights_success(mock_graph_ops):
    test_request = AskRequest(
        query="What are the preferences?",
        output_schema={
            "preferences": ["test"],
            "summary": "test summary"
        }
    )

    # Create a mock for RAGInterface
    mock_rag = AsyncMock()
    mock_rag.get_context = AsyncMock(return_value="test context")
    mock_rag.graph_ops = mock_graph_ops
    
    with patch('persona.services.ask_service.RAGInterface', return_value=mock_rag):
        with patch('persona.services.ask_service.generate_structured_insights', return_value={"test": "data"}):
            response = await AskService.ask_insights("test_user", test_request, mock_graph_ops)

    assert response is not None
    assert hasattr(response, 'result')