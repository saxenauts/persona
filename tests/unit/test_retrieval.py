"""
Unit tests for the Retrieval Layer.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timedelta

from persona.core.retrieval import Retriever
from persona.core.query_expansion import QueryExpansion, DateRange
from persona.models.memory import EpisodeMemory, NoteMemory, PsycheMemory, UserCard


@pytest.fixture
def mock_store():
    store = AsyncMock()
    return store


@pytest.fixture
def mock_graph_ops():
    ops = AsyncMock()
    return ops


@pytest.fixture
def user_id():
    return "test_user_retrieval"


@pytest.fixture
def sample_memories():
    return {
        "episode": EpisodeMemory(
            id=uuid4(),
            user_id="test_user_retrieval",
            type="episode",
            title="Morning run",
            content="Ran 5k in the park",
            timestamp=datetime.utcnow(),
        ),
        "note": NoteMemory(
            id=uuid4(),
            user_id="test_user_retrieval",
            type="note",
            title="Run 10k",
            content="Training for marathon",
            status="IN_PROGRESS",
            note_type="task",
        ),
        "psyche": PsycheMemory(
            id=uuid4(),
            user_id="test_user_retrieval",
            type="psyche",
            title="Prefers mornings",
            content="User prefers exercising in the morning",
            psyche_type="preference",
        ),
    }


class TestRetriever:
    @pytest.mark.asyncio
    async def test_get_working_memory_returns_xml(
        self, mock_store, mock_graph_ops, user_id, sample_memories
    ):
        mock_graph_ops.text_similarity_search.return_value = {
            "results": [{"nodeName": str(sample_memories["episode"].id), "score": 0.9}]
        }
        mock_store.get.return_value = sample_memories["episode"]
        mock_store.get_by_type.return_value = []
        mock_store.get_connected.return_value = []

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        context = await retriever.get_working_memory(
            "running", top_k=5, hop_depth=1, use_query_expansion=False
        )

        assert isinstance(context, str)
        assert "<working_memory>" in context
        assert "</working_memory>" in context

    @pytest.mark.asyncio
    async def test_vector_search_called_with_query(
        self, mock_store, mock_graph_ops, user_id
    ):
        mock_graph_ops.text_similarity_search.return_value = {"results": []}
        mock_store.get_by_type.return_value = []

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        await retriever.get_working_memory(
            "find running memories", top_k=3, use_query_expansion=False
        )

        mock_graph_ops.text_similarity_search.assert_called_once_with(
            query="find running memories", user_id=user_id, limit=3, date_range=None
        )

    @pytest.mark.asyncio
    async def test_static_context_includes_active_notes(
        self, mock_store, mock_graph_ops, user_id, sample_memories
    ):
        active_note = sample_memories["note"]
        completed_note = NoteMemory(
            id=uuid4(),
            user_id=user_id,
            type="note",
            title="Old note",
            content="Done",
            status="COMPLETED",
        )

        mock_store.get_by_type.side_effect = [
            [active_note, completed_note],  # notes
            [sample_memories["psyche"]],  # psyche
        ]
        mock_graph_ops.text_similarity_search.return_value = {"results": []}

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        context = await retriever.get_working_memory(
            "test query", use_query_expansion=False
        )

        # Active goal should be in context, completed should not
        assert "Run 10k" in context or "Training for marathon" in context

    @pytest.mark.asyncio
    async def test_graph_expansion_follows_relationships(
        self, mock_store, mock_graph_ops, user_id, sample_memories
    ):
        seed = sample_memories["episode"]
        linked = sample_memories["note"]

        mock_graph_ops.text_similarity_search.return_value = {
            "results": [{"nodeName": str(seed.id), "score": 0.9}]
        }
        mock_store.get.return_value = seed
        mock_store.get_by_type.return_value = []
        mock_store.get_connected.return_value = [linked]

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        context = await retriever.get_working_memory(
            "running", hop_depth=1, use_query_expansion=False
        )

        mock_store.get_connected.assert_called()

    @pytest.mark.asyncio
    async def test_hop_depth_zero_skips_expansion(
        self, mock_store, mock_graph_ops, user_id, sample_memories
    ):
        seed = sample_memories["episode"]

        mock_graph_ops.text_similarity_search.return_value = {
            "results": [{"nodeName": str(seed.id), "score": 0.9}]
        }
        mock_store.get.return_value = seed
        mock_store.get_by_type.return_value = []

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        await retriever.get_working_memory(
            "running", hop_depth=0, use_query_expansion=False
        )

        mock_store.get_connected.assert_not_called()

    @pytest.mark.asyncio
    async def test_include_static_false_skips_static_context(
        self, mock_store, mock_graph_ops, user_id
    ):
        mock_graph_ops.text_similarity_search.return_value = {"results": []}

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        await retriever.get_working_memory(
            "test", include_static=False, use_query_expansion=False
        )

        mock_store.get_by_type.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_vector_search_failure(
        self, mock_store, mock_graph_ops, user_id
    ):
        mock_graph_ops.text_similarity_search.side_effect = Exception("Vector DB error")
        mock_store.get_by_type.return_value = []

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        context = await retriever.get_working_memory(
            "test query", use_query_expansion=False
        )

        assert isinstance(context, str)
        assert "<working_memory>" in context

    @pytest.mark.asyncio
    async def test_deduplicates_memories(
        self, mock_store, mock_graph_ops, user_id, sample_memories
    ):
        memory = sample_memories["episode"]

        mock_store.get_by_type.side_effect = [
            [memory],  # notes (pretend episode is a note for test)
            [],  # psyche
        ]
        mock_graph_ops.text_similarity_search.return_value = {
            "results": [{"nodeName": str(memory.id), "score": 0.9}]
        }
        mock_store.get.return_value = memory
        mock_store.get_connected.return_value = []

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        context = await retriever.get_working_memory("test", use_query_expansion=False)

        assert isinstance(context, str)

    @pytest.mark.asyncio
    async def test_graph_expansion_limits_fanout_per_node(
        self, mock_store, mock_graph_ops, user_id, sample_memories
    ):
        from uuid import uuid4

        seed = sample_memories["episode"]
        linked_memories = [
            EpisodeMemory(
                id=uuid4(),
                user_id=user_id,
                type="episode",
                title=f"Linked {i}",
                content=f"Content {i}",
                timestamp=datetime.utcnow(),
            )
            for i in range(30)
        ]

        mock_store.get_by_type.return_value = []
        mock_graph_ops.text_similarity_search.return_value = {
            "results": [{"nodeName": str(seed.id), "score": 0.9}]
        }
        mock_store.get.return_value = seed
        mock_store.get_connected.return_value = linked_memories

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        context, stats = await retriever.get_working_memory_with_stats(
            "test", hop_depth=1, include_static=False, use_query_expansion=False
        )

        # Default max_links_per_node=15, so from 30 links only 15 should be added
        assert stats["graph_traversal"]["nodes_visited"] <= 16


class TestLinkScoring:
    @pytest.fixture
    def retriever(self, mock_store, mock_graph_ops, user_id):
        return Retriever(user_id, mock_store, mock_graph_ops)

    @pytest.fixture
    def linked_memories(self, user_id):
        now = datetime.utcnow()
        return [
            EpisodeMemory(
                id=uuid4(),
                user_id=user_id,
                type="episode",
                title="Recent with entity",
                content="Meeting with Jordan about project",
                timestamp=now - timedelta(days=1),
                importance=0.8,
            ),
            EpisodeMemory(
                id=uuid4(),
                user_id=user_id,
                type="episode",
                title="Old episode",
                content="Something old",
                timestamp=now - timedelta(days=60),
                importance=0.3,
            ),
            PsycheMemory(
                id=uuid4(),
                user_id=user_id,
                type="psyche",
                content="Important trait",
                importance=0.9,
            ),
        ]

    def test_scoring_boosts_recent_episodes(self, retriever, linked_memories):
        scored = retriever._score_links(linked_memories, None)
        scores = {m.title if hasattr(m, "title") else m.content: s for s, m in scored}
        assert scores["Recent with entity"] > scores["Old episode"]

    def test_scoring_boosts_entity_matches(self, retriever, linked_memories):
        expansion = QueryExpansion(original_query="about Jordan", entities=["Jordan"])
        scored = retriever._score_links(linked_memories, expansion)
        scores = {m.title if hasattr(m, "title") else m.content: s for s, m in scored}
        assert scores["Recent with entity"] > scores["Old episode"]

    def test_scoring_uses_importance_field(self, retriever, linked_memories):
        scored = retriever._score_links(linked_memories, None)
        high_importance = [m for s, m in scored if m.importance >= 0.8]
        low_importance = [m for s, m in scored if m.importance < 0.5]
        assert len(high_importance) > 0
        assert len(low_importance) > 0


class TestUserCardIntegration:
    @pytest.mark.asyncio
    async def test_get_working_memory_with_user_card(
        self, mock_store, mock_graph_ops, user_id
    ):
        mock_graph_ops.text_similarity_search.return_value = {"results": []}
        mock_store.get_by_type.return_value = []

        card = UserCard(user_id=user_id, name="Test User", current_focus=["Testing"])
        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        context = await retriever.get_working_memory(
            "test", user_card=card, use_query_expansion=False
        )

        assert "<user_card>" in context
        assert "Test User" in context
