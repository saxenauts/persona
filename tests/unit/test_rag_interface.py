import pytest
from unittest.mock import AsyncMock, patch
from persona.core.rag_interface import RAGInterface

@pytest.fixture
async def mock_graph_db():
    mock = AsyncMock()
    mock.get_node = AsyncMock(return_value={"properties": {"content": "test content"}})
    return mock

@pytest.fixture
async def mock_vector_store():
    mock = AsyncMock()
    mock.search_similar = AsyncMock(return_value=[{"node_name": "test", "score": 0.9}])
    return mock


@pytest.mark.asyncio
async def test_rag_query():
    with patch('persona.core.rag_interface.RAGInterface.__aenter__') as mock_rag_enter, \
         patch('persona.core.graph_ops.GraphOps.__aenter__') as mock_graph_ops, \
         patch('persona.llm.llm_graph.generate_response_with_context') as mock_generate:
        
        mock_rag = AsyncMock()
        mock_rag_enter.return_value = mock_rag
        mock_rag.query = AsyncMock(return_value="test response")
        
        async with RAGInterface("test_user") as rag:
            result = await rag.query("test query")
            assert result == "test response"
            mock_rag.query.assert_called_once_with("test query")

@pytest.mark.asyncio
async def test_get_context():
    """Test that get_context properly retrieves and formats context."""
    with patch('persona.core.rag_interface.RAGInterface.__aenter__') as mock_rag_enter:
        mock_rag = AsyncMock()
        mock_rag_enter.return_value = mock_rag
        mock_rag.get_context = AsyncMock(return_value="<memory_context>test context</memory_context>")
        
        async with RAGInterface("test_user") as rag:
            result = await rag.get_context("test query")
            assert isinstance(result, str)
            assert "<memory_context>" in result
            mock_rag.get_context.assert_called_once_with("test query")