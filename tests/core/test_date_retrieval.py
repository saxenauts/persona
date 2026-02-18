import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from persona.core.retrieval import Retriever
from persona.models.memory import Memory
from persona.core.memory_store import MemoryStore
from persona.core.graph_ops import GraphOps
from unittest.mock import AsyncMock, Mock

@pytest.mark.asyncio
async def test_date_extraction_and_filtering():
    """Verify that logical dates are extracted and passed as filters."""
    
    # Mock dependencies
    store = Mock(spec=MemoryStore)
    graph_ops = Mock(spec=GraphOps)
    graph_ops.text_similarity_search = AsyncMock(return_value={"results": []})
    
    retriever = Retriever(user_id="test_user", store=store, graph_ops=graph_ops)
    
    # Case 1: "Last week" relative to 2023-05-29
    query_1 = "(date: 2023-05-29) What did I do last week?"
    await retriever.get_context(query_1)
    
    # Assert date_range was extracted correctly (May 22 to May 29)
    call_args = graph_ops.text_similarity_search.call_args
    assert call_args is not None
    output_kwargs = call_args.kwargs
    
    assert "date_range" in output_kwargs
    start, end = output_kwargs["date_range"]
    assert start == datetime(2023, 5, 22).date()
    assert end == datetime(2023, 5, 29).date()
    
    print(f"✅ Case 1: 'last week' -> {start} to {end}")

    # Case 2: "Yesterday" relative to 2023-05-29
    query_2 = "(date: 2023-05-29) What happened yesterday?"
    await retriever.get_context(query_2)
    
    call_args = graph_ops.text_similarity_search.call_args
    output_kwargs = call_args.kwargs
    start, end = output_kwargs["date_range"]
    assert start == datetime(2023, 5, 28).date()
    assert end == datetime(2023, 5, 28).date()
    
    print(f"✅ Case 2: 'yesterday' -> {start} to {end}")
    
    # Case 3: No temporal keyword -> No filter
    query_3 = "(date: 2023-05-29) Who is my best friend?"
    await retriever.get_context(query_3)
    
    call_args = graph_ops.text_similarity_search.call_args
    output_kwargs = call_args.kwargs
    assert output_kwargs.get("date_range") is None
    
    print(f"✅ Case 3: No keyword -> No filter")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_date_extraction_and_filtering())
