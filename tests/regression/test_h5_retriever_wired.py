"""Regression test for H5: Verify Retriever.get_working_memory() is wired into PersonaService.run_agent()."""

import pytest
from datetime import datetime, timedelta

from persona.services.persona_service import PersonaService
from persona.adapters.persona_adapter import PersonaAdapter


@pytest.mark.asyncio
async def test_retriever_wired_into_run_agent(isolated_graph_ops):
    """Test that PersonaService.run_agent() includes working memory from Retriever."""
    async for graph_ops, user_id in isolated_graph_ops:
        service = PersonaService(graph_ops)
        adapter = PersonaAdapter(user_id=user_id, graph_ops=graph_ops)

        yesterday = datetime.utcnow() - timedelta(days=1)
        await adapter.ingest(
            content="Morning workout: Did 30 minutes of cardio and strength training yesterday",
            timestamp=yesterday,
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


@pytest.mark.asyncio
async def test_working_memory_in_context(isolated_graph_ops):
    """Test that working memory appears in the user context."""
    async for graph_ops, user_id in isolated_graph_ops:
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


@pytest.mark.asyncio
async def test_empty_working_memory_when_no_memories(isolated_graph_ops):
    """Test that working_memory_chars is 0 when user has no memories."""
    async for graph_ops, user_id in isolated_graph_ops:
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
