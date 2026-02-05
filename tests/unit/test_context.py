"""Unit tests for context formatting (prose format)."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from persona.core.context import (
    format_working_memory_prose,
    MemoryAdapter,
)
from persona.models.memory import (
    EpisodeMemory,
    PsycheMemory,
    NoteMemory,
    UserCard,
    MemoryLink,
)


class TestFormatWorkingMemoryProse:
    """Test the main prose formatting function."""

    @pytest.fixture
    def sample_episodes(self):
        now = datetime.now()
        return [
            EpisodeMemory(
                id=uuid4(),
                user_id="test",
                type="episode",
                title="Recent Episode",
                content="Had a great meeting with the team",
                event_time=now,
            ),
            EpisodeMemory(
                id=uuid4(),
                user_id="test",
                type="episode",
                title="Older Episode",
                content="Started the project",
                event_time=now - timedelta(days=7),
            ),
        ]

    @pytest.fixture
    def sample_psyche(self):
        return [
            PsycheMemory(
                id=uuid4(),
                user_id="test",
                type="psyche",
                psyche_type="trait",
                content="Values efficiency and direct communication",
            ),
            PsycheMemory(
                id=uuid4(),
                user_id="test",
                type="psyche",
                psyche_type="preference",
                content="Prefers morning meetings",
            ),
        ]

    @pytest.fixture
    def sample_notes(self):
        return [
            NoteMemory(
                id=uuid4(),
                user_id="test",
                type="note",
                title="Launch MVP",
                content="Ship the product by end of month",
                status="active",
            ),
            NoteMemory(
                id=uuid4(),
                user_id="test",
                type="note",
                title="Completed task",
                content="Already done",
                status="COMPLETED",
            ),
        ]

    def test_empty_context(self):
        """Test formatting with no memories."""
        result = format_working_memory_prose(
            user_card=None,
            episodes=[],
            psyche=[],
            active_notes=[],
        )
        assert result == ""

    def test_user_card_rendering(self):
        """Test that user card appears in output."""
        card = UserCard(
            user_id="test",
            timezone="PST",
            identity_prose="Alex is a founder building an AI startup.",
        )
        result = format_working_memory_prose(
            user_card=card,
            episodes=[],
            psyche=[],
            active_notes=[],
        )
        assert "<user>" in result
        assert "Alex is a founder" in result
        assert "</user>" in result

    def test_episodes_in_recent_context(self, sample_episodes):
        """Test episodes appear in recent_context section."""
        result = format_working_memory_prose(
            user_card=None,
            episodes=sample_episodes,
            psyche=[],
            active_notes=[],
        )
        assert "<recent_context>" in result
        assert "</recent_context>" in result
        assert "great meeting" in result
        assert "Started the project" in result

    def test_episodes_sorted_by_recency(self, sample_episodes):
        """Test that more recent episodes appear first."""
        result = format_working_memory_prose(
            user_card=None,
            episodes=sample_episodes,
            psyche=[],
            active_notes=[],
        )
        recent_pos = result.find("great meeting")
        older_pos = result.find("Started the project")
        assert recent_pos < older_pos

    def test_episode_links_rendered(self):
        """Test that link context is appended to episodes."""
        now = datetime.now()
        current = EpisodeMemory(
            id=uuid4(),
            user_id="test",
            type="episode",
            title="Current Episode",
            content="Wrapped up the sprint",
            event_time=now,
        )
        previous = EpisodeMemory(
            id=uuid4(),
            user_id="test",
            type="episode",
            title="Previous Episode",
            content="Started the sprint",
            event_time=now - timedelta(days=1),
        )
        link = MemoryLink(
            source_id=current.id,
            target_id=previous.id,
            relation="PREVIOUS",
        )
        result = format_working_memory_prose(
            user_card=None,
            episodes=[current, previous],
            psyche=[],
            active_notes=[],
            links=[link],
        )
        assert "preceded by" in result
        assert "Previous Episode" in result

    def test_psyche_in_active_context(self, sample_psyche):
        """Test psyche appears in active_context section."""
        result = format_working_memory_prose(
            user_card=None,
            episodes=[],
            psyche=sample_psyche,
            active_notes=[],
        )
        assert "<active_context>" in result
        assert "Trait:" in result
        assert "Preference:" in result

    def test_notes_filter_completed(self, sample_notes):
        """Test that completed notes are filtered out."""
        result = format_working_memory_prose(
            user_card=None,
            episodes=[],
            psyche=[],
            active_notes=sample_notes,
        )
        assert "Launch MVP" in result
        assert "Already done" not in result

    def test_full_context_structure(self, sample_episodes, sample_psyche, sample_notes):
        """Test full context with all memory types."""
        card = UserCard(
            user_id="test",
            timezone="PST",
            identity_prose="Test user identity",
        )
        result = format_working_memory_prose(
            user_card=card,
            episodes=sample_episodes,
            psyche=sample_psyche,
            active_notes=sample_notes,
        )
        user_pos = result.find("<user>")
        recent_pos = result.find("<recent_context>")
        active_pos = result.find("<active_context>")
        assert user_pos < recent_pos < active_pos


class TestMemoryAdapter:
    """Test MemoryAdapter for storage->model conversion."""

    def test_from_storage_episode(self):
        """Test converting raw storage dict to EpisodeMemory."""
        adapter = MemoryAdapter()
        raw = {
            "id": str(uuid4()),
            "user_id": "test",
            "type": "episode",
            "title": "Test Episode",
            "content": "Content here",
            "timestamp": "2025-12-27T10:00:00Z",
        }
        memory = adapter.from_storage(raw)
        assert memory.type == "episode"
        assert memory.title == "Test Episode"

    def test_from_storage_goal_to_note_migration(self):
        """Test that old 'goal' type is migrated to 'note'."""
        adapter = MemoryAdapter()
        raw = {
            "id": str(uuid4()),
            "user_id": "test",
            "type": "goal",
            "goal_type": "task",
            "title": "Old Goal",
            "content": "Content",
        }
        memory = adapter.from_storage(raw)
        assert memory.type == "note"
        assert hasattr(memory, "note_type")
        assert memory.note_type == "task"

    def test_from_storage_batch(self):
        """Test batch conversion."""
        adapter = MemoryAdapter()
        raw_list = [
            {
                "id": str(uuid4()),
                "user_id": "test",
                "type": "episode",
                "content": f"Episode {i}",
            }
            for i in range(3)
        ]
        memories = adapter.from_storage_batch(raw_list)
        assert len(memories) == 3


class TestLinkProseFormatting:
    """Test link context in prose output."""

    def test_links_appear_in_episode_context(self):
        """Test that links are rendered in episode prose."""
        ep1_id = uuid4()
        ep2_id = uuid4()
        episodes = [
            EpisodeMemory(
                id=ep1_id,
                user_id="test",
                type="episode",
                title="First event",
                content="Something happened",
                event_time=datetime.now(),
            ),
            EpisodeMemory(
                id=ep2_id,
                user_id="test",
                type="episode",
                title="Second event",
                content="Follow-up",
                event_time=datetime.now() - timedelta(hours=1),
            ),
        ]
        links = [
            MemoryLink(
                source_id=ep1_id,
                target_id=ep2_id,
                relation="led_to",
            )
        ]
        result = format_working_memory_prose(
            user_card=None,
            episodes=episodes,
            psyche=[],
            active_notes=[],
            links=links,
        )
        assert "led to" in result
