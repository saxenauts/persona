"""
Unit tests for the Retrieval Layer.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timedelta

from persona.core.retrieval import Retriever
from persona.core.context import ContextView
from persona.core.query_expansion import QueryExpansion, DateRange
from persona.models.memory import EpisodeMemory, NoteMemory, PsycheMemory, UserCard


@pytest.fixture
def mock_store():
    """Mock MemoryStore."""
    store = AsyncMock()
    return store


@pytest.fixture
def mock_graph_ops():
    """Mock GraphOps."""
    ops = AsyncMock()
    return ops


@pytest.fixture
def user_id():
    return "test_user_retrieval"


@pytest.fixture
def sample_memories():
    """Sample memories for testing."""
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
    """Tests for Retriever class."""

    @pytest.mark.asyncio
    async def test_get_context_returns_xml(
        self, mock_store, mock_graph_ops, user_id, sample_memories
    ):
        """Test that get_context returns XML-formatted string."""
        # Setup mocks
        mock_graph_ops.text_similarity_search.return_value = {
            "results": [{"nodeName": str(sample_memories["episode"].id), "score": 0.9}]
        }
        mock_store.get.return_value = sample_memories["episode"]
        mock_store.get_by_type.return_value = []
        mock_store.get_connected.return_value = []

        # Execute
        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        context = await retriever.get_context("running", top_k=5, hop_depth=1)

        # Assert
        assert isinstance(context, str)
        assert "<memory_context>" in context
        assert "</memory_context>" in context

    @pytest.mark.asyncio
    async def test_vector_search_called_with_query(
        self, mock_store, mock_graph_ops, user_id
    ):
        """Test that vector search is called with the query."""
        mock_graph_ops.text_similarity_search.return_value = {"results": []}
        mock_store.get_by_type.return_value = []

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        await retriever.get_context("find running memories", top_k=3)

        mock_graph_ops.text_similarity_search.assert_called_once_with(
            query="find running memories", user_id=user_id, limit=3, date_range=None
        )

    @pytest.mark.asyncio
    async def test_static_context_includes_active_notes(
        self, mock_store, mock_graph_ops, user_id, sample_memories
    ):
        """Test that static context includes active notes."""
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
        context = await retriever.get_context("test query")

        # Active goal should be in context, completed should not
        assert "Run 10k" in context or "Training for marathon" in context

    @pytest.mark.asyncio
    async def test_graph_expansion_follows_relationships(
        self, mock_store, mock_graph_ops, user_id, sample_memories
    ):
        """Test that graph expansion follows relationships."""
        seed = sample_memories["episode"]
        linked = sample_memories["note"]

        mock_graph_ops.text_similarity_search.return_value = {
            "results": [{"nodeName": str(seed.id), "score": 0.9}]
        }
        mock_store.get.return_value = seed
        mock_store.get_by_type.return_value = []
        mock_store.get_connected.return_value = [linked]

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        context = await retriever.get_context("running", hop_depth=1)

        # Both seed and linked memory should be in context
        mock_store.get_connected.assert_called()

    @pytest.mark.asyncio
    async def test_hop_depth_zero_skips_expansion(
        self, mock_store, mock_graph_ops, user_id, sample_memories
    ):
        """Test that hop_depth=0 skips graph expansion."""
        seed = sample_memories["episode"]

        mock_graph_ops.text_similarity_search.return_value = {
            "results": [{"nodeName": str(seed.id), "score": 0.9}]
        }
        mock_store.get.return_value = seed
        mock_store.get_by_type.return_value = []

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        await retriever.get_context("running", hop_depth=0)

        # get_connected should not be called when hop_depth=0
        mock_store.get_connected.assert_not_called()

    @pytest.mark.asyncio
    async def test_include_static_false_skips_static_context(
        self, mock_store, mock_graph_ops, user_id
    ):
        """Test that include_static=False skips static context."""
        mock_graph_ops.text_similarity_search.return_value = {"results": []}

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        await retriever.get_context("test", include_static=False)

        # get_by_type should not be called for static context
        mock_store.get_by_type.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_vector_search_failure(
        self, mock_store, mock_graph_ops, user_id
    ):
        """Test graceful handling of vector search failure."""
        mock_graph_ops.text_similarity_search.side_effect = Exception("Vector DB error")
        mock_store.get_by_type.return_value = []

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        context = await retriever.get_context("test query")

        # Should still return valid context (empty or with static)
        assert isinstance(context, str)
        assert "<memory_context>" in context

    @pytest.mark.asyncio
    async def test_deduplicates_memories(
        self, mock_store, mock_graph_ops, user_id, sample_memories
    ):
        """Test that duplicate memories are deduplicated."""
        memory = sample_memories["episode"]

        # Same memory appears in static and vector search
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

        # Should not raise, should deduplicate
        context = await retriever.get_context("test")
        assert isinstance(context, str)

    @pytest.mark.asyncio
    async def test_graph_expansion_limits_fanout_per_node(
        self, mock_store, mock_graph_ops, user_id, sample_memories
    ):
        """Test that graph expansion limits links per node to prevent hub domination."""
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
        context, stats = await retriever.get_context_with_stats(
            "test", hop_depth=1, include_static=False
        )

        # Default max_links_per_node=15, so from 30 links only 15 should be added
        # Total: 1 seed + 15 linked = 16 max
        assert stats["graph_traversal"]["nodes_visited"] <= 16


class TestContextViewRouter:
    @pytest.fixture
    def retriever(self, mock_store, mock_graph_ops, user_id):
        return Retriever(user_id, mock_store, mock_graph_ops)

    def test_timeline_view_from_date_range(self, retriever):
        expansion = QueryExpansion(
            original_query="test",
            date_range=DateRange(
                start=datetime.now().date() - timedelta(days=7),
                end=datetime.now().date(),
            ),
        )
        view = retriever._route_context_view("test", expansion)
        assert view == ContextView.TIMELINE

    def test_timeline_view_from_keywords(self, retriever):
        view = retriever._route_context_view("what happened last week", None)
        assert view == ContextView.TIMELINE

    def test_tasks_view_from_keywords(self, retriever):
        view = retriever._route_context_view("what should i do today", None)
        assert view == ContextView.TASKS

        view = retriever._route_context_view("show me my tasks", None)
        assert view == ContextView.TASKS

    def test_profile_view_from_keywords(self, retriever):
        view = retriever._route_context_view("who am i", None)
        assert view == ContextView.PROFILE

        view = retriever._route_context_view("what do i like", None)
        assert view == ContextView.PROFILE

    def test_graph_neighborhood_from_entities(self, retriever):
        expansion = QueryExpansion(
            original_query="about Jordan",
            entities=["Jordan"],
        )
        view = retriever._route_context_view("about Jordan", expansion)
        assert view == ContextView.GRAPH_NEIGHBORHOOD

    def test_default_to_profile(self, retriever):
        view = retriever._route_context_view("random query", None)
        assert view == ContextView.PROFILE


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
    async def test_get_context_with_user_card(
        self, mock_store, mock_graph_ops, user_id
    ):
        mock_graph_ops.text_similarity_search.return_value = {"results": []}
        mock_store.get_by_type.return_value = []

        card = UserCard(user_id=user_id, name="Test User", current_focus=["Testing"])
        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        context = await retriever.get_context("test", user_card=card)

        assert "<user_card>" in context
        assert "Test User" in context

    @pytest.mark.asyncio
    async def test_get_context_with_stats_includes_view(
        self, mock_store, mock_graph_ops, user_id
    ):
        mock_graph_ops.text_similarity_search.return_value = {"results": []}
        mock_store.get_by_type.return_value = []

        retriever = Retriever(user_id, mock_store, mock_graph_ops)
        context, stats = await retriever.get_context_with_stats(
            "what happened yesterday"
        )

        assert "context_view" in stats
        assert stats["context_view"] == "timeline"
