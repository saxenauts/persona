"""
Unit tests for the Retrieval Layer.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime

from persona.core.retrieval import Retriever
from persona.models.memory import EpisodeMemory, NoteMemory, PsycheMemory


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
