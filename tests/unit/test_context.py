"""Unit tests for context formatting with token budget."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from persona.core.context import ContextFormatter, ContextBudget, ContextView
from persona.models.memory import EpisodeMemory, PsycheMemory, NoteMemory, UserCard


class TestContextBudget:
    """Test token budget enforcement."""

    @pytest.fixture
    def formatter(self):
        return ContextFormatter()

    @pytest.fixture
    def sample_episodes(self):
        return [
            EpisodeMemory(
                id=uuid4(),
                user_id="test",
                type="episode",
                title=f"Episode {i}",
                content="x" * 200,
                timestamp=datetime.now(),
            )
            for i in range(10)
        ]

    def test_default_budget_values(self):
        """Test ContextBudget has sensible defaults."""
        budget = ContextBudget()
        assert budget.total_tokens == 4000
        assert budget.user_card_budget == 300
        assert budget.psyche_budget == 600
        assert budget.episode_budget == 2400
        assert budget.note_budget == 700

    def test_fit_to_budget_limits_content(self, formatter, sample_episodes):
        """Test that _fit_to_budget respects token limits."""
        small_budget = 100
        limited = formatter._fit_to_budget(sample_episodes, small_budget)
        assert len(limited) < len(sample_episodes)
        assert len(limited) > 0

    def test_format_context_with_budget(self, formatter, sample_episodes):
        """Test format_context respects budget parameter."""
        budget = ContextBudget(episode_budget=100)
        context = formatter.format_context(sample_episodes, budget=budget)
        assert "<memory_context>" in context
        assert "</memory_context>" in context
        episode_count = context.count("<episode")
        assert episode_count < len(sample_episodes)

    def test_format_context_without_budget(self, formatter, sample_episodes):
        """Test format_context includes all memories without budget."""
        context = formatter.format_context(sample_episodes)
        episode_count = context.count("<episode ")
        assert episode_count == len(sample_episodes)


class TestUserCard:
    @pytest.fixture
    def formatter(self):
        return ContextFormatter()

    def test_user_card_rendering(self, formatter):
        card = UserCard(
            user_id="test",
            name="Alex",
            timezone="PST",
            roles=["Founder", "Engineer"],
            current_focus=["Launch MVP", "Training"],
            core_values=["Speed", "Quality"],
        )
        output = formatter._format_user_card(card)
        assert "<user_card>" in output
        assert "Alex" in output
        assert "PST" in output
        assert "Current focus:" in output

    def test_user_card_in_context(self, formatter):
        card = UserCard(user_id="test", name="Test User")
        memories = [
            EpisodeMemory(
                id=uuid4(),
                user_id="test",
                type="episode",
                title="Test",
                content="Test content",
                timestamp=datetime.now(),
            )
        ]
        context = formatter.format_context(memories, user_card=card)
        assert context.index("<user_card>") < context.index("<episodes>")


class TestContextOrdering:
    @pytest.fixture
    def formatter(self):
        return ContextFormatter()

    @pytest.fixture
    def mixed_memories(self):
        now = datetime.now()
        return [
            EpisodeMemory(
                id=uuid4(),
                user_id="test",
                type="episode",
                title="Recent Episode",
                content="Recent",
                timestamp=now,
                importance=0.8,
            ),
            EpisodeMemory(
                id=uuid4(),
                user_id="test",
                type="episode",
                title="Old Episode",
                content="Old",
                timestamp=now - timedelta(days=30),
                importance=0.3,
            ),
            PsycheMemory(
                id=uuid4(),
                user_id="test",
                type="psyche",
                psyche_type="trait",
                content="Important trait",
                importance=0.9,
            ),
            PsycheMemory(
                id=uuid4(),
                user_id="test",
                type="psyche",
                psyche_type="preference",
                content="Less important pref",
                importance=0.4,
            ),
            NoteMemory(
                id=uuid4(),
                user_id="test",
                type="note",
                title="Active Task",
                content="Do this",
                status="active",
                importance=0.7,
            ),
        ]

    def test_profile_view_ordering(self, formatter, mixed_memories):
        context = formatter.format_context(mixed_memories, view=ContextView.PROFILE)
        psyche_pos = context.find("<psyche>")
        notes_pos = context.find("<notes>")
        episodes_pos = context.find("<episodes>")
        assert psyche_pos < notes_pos < episodes_pos

    def test_tasks_view_ordering(self, formatter, mixed_memories):
        context = formatter.format_context(mixed_memories, view=ContextView.TASKS)
        assert "<active_tasks>" in context

    def test_timeline_view_ordering(self, formatter, mixed_memories):
        context = formatter.format_context(mixed_memories, view=ContextView.TIMELINE)
        assert "<timeline>" in context

    def test_importance_sorting(self, formatter, mixed_memories):
        psyches = [m for m in mixed_memories if isinstance(m, PsycheMemory)]
        sorted_psyches = formatter._sort_by_importance(psyches)
        assert sorted_psyches[0].importance >= sorted_psyches[1].importance

    def test_recency_sorting(self, formatter, mixed_memories):
        episodes = [m for m in mixed_memories if isinstance(m, EpisodeMemory)]
        sorted_eps = formatter._sort_by_recency(episodes)
        assert sorted_eps[0].timestamp >= sorted_eps[1].timestamp


class TestContextViewEnum:
    def test_context_view_values(self):
        assert ContextView.PROFILE.value == "profile"
        assert ContextView.TIMELINE.value == "timeline"
        assert ContextView.TASKS.value == "tasks"
        assert ContextView.GRAPH_NEIGHBORHOOD.value == "graph_neighborhood"
