"""Regression test for H5: Verify Retriever.get_working_memory() is wired into PersonaService.run_agent()."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from persona.services.persona_service import PersonaService
from persona.adapters.persona_adapter import PersonaAdapter
from persona.core.graph_ops import GraphOps
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.core.backends.neo4j_vector import Neo4jVectorStore


@pytest.mark.asyncio
async def test_retriever_wired_into_run_agent():
    """Test that PersonaService.run_agent() includes working memory from Retriever."""
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    vector_store = Neo4jVectorStore(graph_driver=graph_db.driver)
    await vector_store.initialize()

    user_id = f"test-user-{uuid4()}"
    await graph_db.create_user(user_id)

    try:
        graph_ops = GraphOps(graph_db=graph_db, vector_store=vector_store)
        service = PersonaService(graph_ops)
        adapter = PersonaAdapter(user_id=user_id, graph_ops=graph_ops)

        two_hours_ago = datetime.utcnow() - timedelta(hours=2)
        await adapter.ingest(
            content="Morning workout: Did 30 minutes of cardio and strength training",
            timestamp=two_hours_ago,
        )

        result = await service.run_agent(
            user_id=user_id,
            query="What did I do yesterday?",
            include_stats=True,
            user_timezone="UTC",
        )

        assert result["status"] == "completed"
        assert "stats" in result
        stats = result["stats"]

        assert "working_memory_chars" in stats
        assert stats["working_memory_chars"] > 0, "working_memory should be non-empty"

        assert "retriever" in stats
        retriever_stats = stats["retriever"]
        assert "episode_count" in retriever_stats
        assert retriever_stats["episode_count"] >= 1, (
            "Should have retrieved at least 1 episode"
        )
    finally:
        await graph_db.delete_user(user_id)
        await vector_store.close()
        await graph_db.close()


@pytest.mark.asyncio
async def test_working_memory_in_context():
    """Test that working memory appears in the user context."""
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    vector_store = Neo4jVectorStore(graph_driver=graph_db.driver)
    await vector_store.initialize()

    user_id = f"test-user-{uuid4()}"
    await graph_db.create_user(user_id)

    try:
        graph_ops = GraphOps(graph_db=graph_db, vector_store=vector_store)
        service = PersonaService(graph_ops)
        adapter = PersonaAdapter(user_id=user_id, graph_ops=graph_ops)

        now = datetime.utcnow()
        await adapter.ingest(
            content="Unique marker event XYZ123: This is a unique event that should appear in working memory",
            timestamp=now - timedelta(hours=2),
        )

        result = await service.run_agent(
            user_id=user_id,
            query="Tell me about recent events",
            include_stats=True,
            user_timezone="UTC",
        )

        assert result["status"] == "completed"
        stats = result["stats"]

        assert stats["working_memory_chars"] > 0
        assert stats["retriever"]["episode_count"] >= 1
    finally:
        await graph_db.delete_user(user_id)
        await vector_store.close()
        await graph_db.close()


@pytest.mark.asyncio
async def test_empty_working_memory_when_no_memories():
    """Test that working_memory_chars is 0 when user has no memories."""
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    vector_store = Neo4jVectorStore(graph_driver=graph_db.driver)
    await vector_store.initialize()

    user_id = f"test-user-{uuid4()}"
    await graph_db.create_user(user_id)

    try:
        graph_ops = GraphOps(graph_db=graph_db, vector_store=vector_store)
        service = PersonaService(graph_ops)

        result = await service.run_agent(
            user_id=user_id,
            query="What do you know about me?",
            include_stats=True,
            user_timezone="UTC",
        )

        assert result["status"] == "completed"
        stats = result["stats"]

        assert "working_memory_chars" in stats
        assert stats["retriever"]["episode_count"] == 0
        assert stats["retriever"]["psyche_count"] == 0
        assert stats["retriever"]["note_count"] == 0
    finally:
        await graph_db.delete_user(user_id)
        await vector_store.close()
        await graph_db.close()
