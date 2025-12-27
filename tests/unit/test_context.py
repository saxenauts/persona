"""Unit tests for context formatting with token budget."""

import pytest
from datetime import datetime
from uuid import uuid4

from persona.core.context import ContextFormatter, ContextBudget
from persona.models.memory import EpisodeMemory, PsycheMemory, NoteMemory


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
        assert budget.psyche_budget == 800
        assert budget.episode_budget == 2500
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
