"""
Regression test for H2 browse() historical date filtering bug.

Bug: browse() fetched limit*2 memories per type, then filtered by date in Python.
For old date ranges (e.g., "June 2023"), the fetch only got recent memories,
so Python filter found nothing. Result: empty or partial results for historical queries.

Fix: Push date filtering into Cypher WHERE clause in:
1. persona/core/backends/neo4j_graph.py: get_nodes_by_type() - add date_start/date_end params
2. persona/core/memory_store.py: get_by_type() - pass date params to graph query
3. persona/tools/memory.py: browse_handler() - remove Python filter, pass dates to get_by_type

Test verifies:
1. browse(date_start="2023-06-01", date_end="2023-06-30") returns ALL June 2023 memories
2. Not limited by recent fetch (limit*2) - proves filtering happens in Cypher, not Python
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from persona.models.memory import EpisodeMemory
from persona.core.memory_store import MemoryStore
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.core.backends.neo4j_vector import Neo4jVectorStore
from persona.tools.memory import browse_handler
from persona.tools.context import ToolContext


@pytest.mark.asyncio
async def test_browse_historical_returns_all_memories():
    """
    Verify that browse() with historical date range returns ALL matching memories,
    not just recent limit*2.

    This is the primary bug case - with limit=10, old code would fetch 20 recent
    memories, then filter by date (finding 0). New code filters in Cypher, finding all 50.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    vector_store = Neo4jVectorStore(graph_driver=graph_db.driver)
    await vector_store.initialize()
    memory_store = MemoryStore(graph_db, vector_store)

    user_id = f"test-user-{uuid4()}"
    await graph_db.create_user(user_id)

    june_2023_start = datetime(2023, 6, 1, 0, 0, 0)

    memories = []
    for i in range(50):
        event_time = june_2023_start + timedelta(hours=i)
        memory = EpisodeMemory(
            user_id=user_id,
            title=f"June 2023 Event {i + 1}",
            content=f"This is event number {i + 1} in June 2023",
            event_time=event_time,
        )
        memories.append(memory)

    await memory_store.create_many(memories, [], user_id)

    ctx = ToolContext(
        user_id=user_id,
        graph_ops=None,
        store=memory_store,
        adapter=None,
        user_timezone="UTC",
        user_card=None,
    )

    result = await browse_handler(
        ctx=ctx,
        date_start="2023-06-01",
        date_end="2023-06-30",
        memory_types=["episode"],
        limit=50,
        order="desc",
    )

    await graph_db.delete_user(user_id)
    await graph_db.close()
    await vector_store.close()

    assert result.count == 50, f"Expected 50 memories, got {result.count}"
    assert len(result.items) == 50, f"Expected 50 items, got {len(result.items)}"

    for item in result.items:
        assert "June 2023 Event" in item.title
        event_time = datetime.fromisoformat(item.event_time)
        assert event_time.year == 2023
        assert event_time.month == 6


@pytest.mark.asyncio
async def test_browse_historical_with_small_limit():
    """
    Verify that browse() with small limit still finds historical memories.

    Old bug: limit=10 would fetch 20 recent memories, filter by June 2023 -> 0 results.
    New behavior: limit=10 filters in Cypher, returns 10 oldest June 2023 memories.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    vector_store = Neo4jVectorStore(graph_driver=graph_db.driver)
    await vector_store.initialize()
    memory_store = MemoryStore(graph_db, vector_store)

    user_id = f"test-user-{uuid4()}"
    await graph_db.create_user(user_id)

    june_2023_start = datetime(2023, 6, 1, 0, 0, 0)

    memories = []
    for i in range(50):
        event_time = june_2023_start + timedelta(hours=i)
        memory = EpisodeMemory(
            user_id=user_id,
            title=f"June 2023 Event {i + 1}",
            content=f"This is event number {i + 1} in June 2023",
            event_time=event_time,
        )
        memories.append(memory)

    await memory_store.create_many(memories, [], user_id)

    ctx = ToolContext(
        user_id=user_id,
        graph_ops=None,
        store=memory_store,
        adapter=None,
        user_timezone="UTC",
        user_card=None,
    )

    result = await browse_handler(
        ctx=ctx,
        date_start="2023-06-01",
        date_end="2023-06-30",
        memory_types=["episode"],
        limit=10,
        order="desc",
    )

    await graph_db.delete_user(user_id)
    await graph_db.close()
    await vector_store.close()

    assert result.count == 10, f"Expected 10 memories, got {result.count}"
    assert len(result.items) == 10, f"Expected 10 items, got {len(result.items)}"

    for item in result.items:
        assert "June 2023 Event" in item.title


@pytest.mark.asyncio
async def test_browse_historical_ascending_order():
    """
    Verify that browse() with order="asc" returns oldest memories first.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    vector_store = Neo4jVectorStore(graph_driver=graph_db.driver)
    await vector_store.initialize()
    memory_store = MemoryStore(graph_db, vector_store)

    user_id = f"test-user-{uuid4()}"
    await graph_db.create_user(user_id)

    june_2023_start = datetime(2023, 6, 1, 0, 0, 0)

    memories = []
    for i in range(50):
        event_time = june_2023_start + timedelta(hours=i)
        memory = EpisodeMemory(
            user_id=user_id,
            title=f"June 2023 Event {i + 1}",
            content=f"This is event number {i + 1} in June 2023",
            event_time=event_time,
        )
        memories.append(memory)

    await memory_store.create_many(memories, [], user_id)

    ctx = ToolContext(
        user_id=user_id,
        graph_ops=None,
        store=memory_store,
        adapter=None,
        user_timezone="UTC",
        user_card=None,
    )

    result = await browse_handler(
        ctx=ctx,
        date_start="2023-06-01",
        date_end="2023-06-30",
        memory_types=["episode"],
        limit=5,
        order="asc",
    )

    await graph_db.delete_user(user_id)
    await graph_db.close()
    await vector_store.close()

    assert result.count == 5

    event_times = [datetime.fromisoformat(item.event_time) for item in result.items]
    assert event_times == sorted(event_times), "Events should be in ascending order"

    for item in result.items:
        assert "June 2023 Event" in item.title
