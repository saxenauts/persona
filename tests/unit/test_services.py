import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from persona.services.user_service import UserService
from persona.services.rag_service import RAGService
from persona.services.ask_service import AskService
from persona.core.graph_ops import GraphOps
from persona.models.schema import AskRequest

@pytest.fixture
def mock_graph_ops():
    mock = AsyncMock(spec=GraphOps)
    mock.__aenter__.return_value = mock
    mock.__aexit__.return_value = None
    return mock


@pytest.mark.asyncio
async def test_create_user_success(mock_graph_ops):
    mock_graph_ops.create_user = AsyncMock()
    mock_graph_ops.user_exists = AsyncMock(return_value=False)  # User doesn't exist

    result = await UserService.create_user("test_user", mock_graph_ops)
    assert result["message"] == "User test_user created successfully"
    assert result["status"] == "created"

@pytest.mark.asyncio
async def test_delete_user_success(mock_graph_ops):
    mock_graph_ops.delete_user = AsyncMock()
    
    result = await UserService.delete_user("test_user", mock_graph_ops)
    assert result["message"] == "User test_user deleted successfully"
    mock_graph_ops.delete_user.assert_called_once_with("test_user")

@pytest.mark.asyncio
async def test_rag_query_success(mock_graph_ops):
    # Create a mock for RAGInterface
    mock_rag = AsyncMock()
    mock_rag.query = AsyncMock(return_value="Test response")
    
    with patch('persona.services.rag_service.RAGInterface') as MockRAGInterface:
        MockRAGInterface.return_value.__aenter__.return_value = mock_rag
        
        result = await RAGService.query("test_user", "Test query")
        assert isinstance(result, str)
        assert result == "Test response"
        # RAGInterface should be initialized with user_id
        MockRAGInterface.assert_called_with("test_user")

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
    
    with patch('persona.services.ask_service.RAGInterface') as MockRAGInterface:
        MockRAGInterface.return_value.__aenter__.return_value = mock_rag
        
        with patch('persona.services.ask_service.generate_structured_insights', return_value={"test": "data"}):
            response = await AskService.ask_insights("test_user", test_request)

    assert response is not None
    assert hasattr(response, 'result')