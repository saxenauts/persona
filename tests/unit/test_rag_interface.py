import pytest
from unittest.mock import AsyncMock, patch
from persona.core.rag_interface import RAGInterface
from persona.models.schema import NodeModel

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

@pytest.fixture
async def mock_rag_interface(mock_graph_db, mock_vector_store):
    with patch('persona.core.rag_interface.RAGInterface.__aenter__', return_value=AsyncMock()) as mock_enter:
        mock_enter.return_value.graph_db = mock_graph_db
        mock_enter.return_value.vector_store = mock_vector_store
        mock_enter.return_value.query = AsyncMock(return_value="test response")
        mock_enter.return_value.get_context = AsyncMock(return_value="test context")
        yield mock_enter.return_value

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
    with patch('persona.core.graph_ops.GraphOps.__aenter__') as mock_graph_ops:
        # Setup mock graph ops
        mock_graph_ops.return_value.text_similarity_search.return_value = {
            'results': [{'nodeId': 1, 'nodeName': 'test', 'score': 0.9}]
        }
        mock_graph_ops.return_value.get_node_data.return_value = NodeModel(
            name="test",
            properties={"content": "test content"}
        )
        mock_graph_ops.return_value.get_node_relationships.return_value = []
        
        async with RAGInterface("test_user") as rag:
            result = await rag.get_context("test query")
            assert isinstance(result, str)
            mock_graph_ops.return_value.text_similarity_search.assert_called_once()