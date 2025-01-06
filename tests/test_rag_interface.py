import pytest
from luna9.core.rag_interface import RAGInterface

@pytest.mark.asyncio
async def test_rag_query(neo4j_manager):
    rag = RAGInterface("test_user", neo4j_manager)
    
    # Add some test data
    await rag.graph_ops.add_node_with_embedding("Python", "test_user", {"description": "A programming language"})
    await rag.graph_ops.add_node_with_embedding("FastAPI", "test_user", {"description": "A web framework for Python"})
    
    query = "What is Python?"
    result = await rag.query(query)
    
    assert "Python" in result
    assert "programming language" in result.lower()

@pytest.mark.asyncio
async def test_get_context(neo4j_manager):
    rag = RAGInterface("test_user", neo4j_manager)
    
    query = "Tell me about Python and FastAPI"
    context = await rag.get_context(query)
    
    assert len(context) > 0
    assert any("Python" in item for item in context)
    assert any("FastAPI" in item for item in context)