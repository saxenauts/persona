from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Type, Union
from pydantic import BaseModel, TypeAdapter
from persona.models.memory import (
    Memory,
    MemoryLink,
    EpisodeMemory,
    PsycheMemory,
    NoteMemory,
    UserCard,
)


class ContextView(str, Enum):
    """Context presentation view based on query intent."""

    PROFILE = "profile"
    TIMELINE = "timeline"
    TASKS = "tasks"
    GRAPH_NEIGHBORHOOD = "graph_neighborhood"


class ContextBudget(BaseModel):
    """Token budget allocation for context building."""

    total_tokens: int = 4000
    user_card_budget: int = 300
    psyche_budget: int = 600
    episode_budget: int = 2400
    note_budget: int = 700


# =============================================================================
# MemoryAdapter: Storage -> Domain Model Conversion
# =============================================================================


class MemoryAdapter:
    """
    Converts between storage format (flat dict from Neo4j) and domain Memory models.
    """

    _memory_adapter = TypeAdapter(Memory)

    def from_storage(self, raw: Dict[str, Any]) -> Memory:
        """
        Convert a raw storage dict (from Neo4j) to a polymorphic Memory model.
        """
        # Pydantic's Memory union (with discriminator='type') handles the mapping
        # if the input dict has the 'type' field and matches the model fields.

        # Ensure ID is a UUID if it's a string
        if "id" in raw and isinstance(raw["id"], str):
            from uuid import UUID

            try:
                raw["id"] = UUID(raw["id"])
            except ValueError:
                pass
        elif "name" in raw and "id" not in raw:
            # Fallback for older nodes where 'name' was the ID
            from uuid import UUID

            try:
                raw["id"] = UUID(raw["name"])
            except ValueError:
                pass

        # Handle timestamp strings from Neo4j
        if "timestamp" in raw and isinstance(raw["timestamp"], str):
            try:
                raw["timestamp"] = datetime.fromisoformat(
                    raw["timestamp"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        # Handle legacy 'goal' type -> 'note' migration
        if raw.get("type") == "goal":
            raw["type"] = "note"
            # Migrate goal_type to note_type if present
            if "goal_type" in raw and "note_type" not in raw:
                raw["note_type"] = raw.pop("goal_type")

        return self._memory_adapter.validate_python(raw)

    def from_storage_batch(self, raw_nodes: List[Dict[str, Any]]) -> List[Memory]:
        """Convert multiple storage dicts to Memory models."""
        return [self.from_storage(r) for r in raw_nodes]


# =============================================================================
# ContextFormatter: Memory -> LLM Context (Semantic-First, No IDs)
# =============================================================================


class ContextFormatter:
    """Formats retrieved memories into LLM-optimized context."""

    CHARS_PER_TOKEN = 4

    def format_context(
        self,
        memories: List[Memory],
        links: List[MemoryLink] = None,
        max_nodes: int = 50,
        budget: Optional[ContextBudget] = None,
        user_card: Optional[UserCard] = None,
        view: ContextView = ContextView.PROFILE,
    ) -> str:
        """
        Build LLM context from memories with research-based ordering.

        Ordering based on "Lost in the Middle" research:
        1. User Card FIRST (primacy anchor)
        2. Query-relevant psyche/notes (middle - sorted by importance)
        3. Episodes LAST (recency anchor - most attention here)
        4. Identity checksum (optional recap at very end)
        """
        limited_memories = memories[:max_nodes]

        episodes = [m for m in limited_memories if isinstance(m, EpisodeMemory)]
        psyches = [m for m in limited_memories if isinstance(m, PsycheMemory)]
        notes = [m for m in limited_memories if isinstance(m, NoteMemory)]

        if budget:
            episodes = self._fit_to_budget(episodes, budget.episode_budget)
            psyches = self._fit_to_budget(psyches, budget.psyche_budget)
            notes = self._fit_to_budget(notes, budget.note_budget)

        psyches = self._sort_by_importance(psyches)
        notes = self._sort_by_importance(notes)
        episodes = self._sort_by_recency(episodes)

        lines = ["<memory_context>"]

        if user_card:
            lines.append(self._format_user_card(user_card))

        if view == ContextView.TASKS:
            lines.extend(self._format_tasks_view(notes, psyches, episodes))
        elif view == ContextView.TIMELINE:
            lines.extend(self._format_timeline_view(episodes, psyches, notes))
        else:
            lines.extend(self._format_profile_view(psyches, notes, episodes))

        lines.append("</memory_context>")
        return "\n".join(lines)

    def _format_profile_view(
        self,
        psyches: List[PsycheMemory],
        notes: List[NoteMemory],
        episodes: List[EpisodeMemory],
    ) -> List[str]:
        """Default view: psyche first, notes, then episodes last (recency)."""
        lines = []

        if psyches:
            lines.append("<psyche>")
            for p in psyches:
                lines.append(self._format_psyche(p))
            lines.append("</psyche>")

        if notes:
            lines.append("<notes>")
            for n in notes:
                lines.append(self._format_note(n))
            lines.append("</notes>")

        if episodes:
            lines.append("<episodes>")
            for ep in episodes:
                lines.append(self._format_episode(ep))
            lines.append("</episodes>")

        return lines

    def _format_tasks_view(
        self,
        notes: List[NoteMemory],
        psyches: List[PsycheMemory],
        episodes: List[EpisodeMemory],
    ) -> List[str]:
        """Tasks view: notes first, supporting psyche, recent episodes last."""
        lines = []

        active_notes = [n for n in notes if n.status != "COMPLETED"]
        completed_notes = [n for n in notes if n.status == "COMPLETED"]

        if active_notes:
            lines.append("<active_tasks>")
            for n in active_notes:
                lines.append(self._format_note(n))
            lines.append("</active_tasks>")

        if psyches:
            lines.append("<context>")
            for p in psyches[:3]:
                lines.append(self._format_psyche(p))
            lines.append("</context>")

        if episodes:
            lines.append("<recent_activity>")
            for ep in episodes[:5]:
                lines.append(self._format_episode(ep))
            lines.append("</recent_activity>")

        return lines

    def _format_timeline_view(
        self,
        episodes: List[EpisodeMemory],
        psyches: List[PsycheMemory],
        notes: List[NoteMemory],
    ) -> List[str]:
        """Timeline view: chronological episodes, minimal psyche/notes."""
        lines = []

        if psyches:
            lines.append("<identity>")
            for p in psyches[:2]:
                lines.append(self._format_psyche(p))
            lines.append("</identity>")

        if episodes:
            sorted_eps = sorted(episodes, key=lambda e: e.timestamp)
            lines.append("<timeline>")
            for ep in sorted_eps:
                lines.append(self._format_episode(ep))
            lines.append("</timeline>")

        return lines

    def _format_user_card(self, card: UserCard) -> str:
        """Format UserCard as compact identity anchor."""
        parts = []

        header = []
        if card.name:
            header.append(card.name)
        if card.timezone:
            header.append(card.timezone)
        if card.roles:
            header.extend(card.roles[:3])
        if header:
            parts.append(" | ".join(header))

        if card.summary:
            parts.append(card.summary)

        if card.current_focus:
            focus_items = ", ".join(card.current_focus[:5])
            parts.append(f"Current focus: {focus_items}")

        if card.core_values:
            values = ", ".join(card.core_values[:3])
            parts.append(f"Values: {values}")

        if card.key_relationships:
            rels = ", ".join(card.key_relationships[:3])
            parts.append(f"Key people: {rels}")

        if card.communication_style:
            parts.append(f"Style: {card.communication_style}")

        if card.uncertainties:
            uncertain = ", ".join(card.uncertainties[:2])
            parts.append(f"[Uncertain: {uncertain}]")

        content = "\n".join(parts)
        return f"<user_card>\n{content}\n</user_card>"

    def _sort_by_importance(self, memories: list) -> list:
        """Sort memories by importance score (highest first)."""
        return sorted(
            memories, key=lambda m: getattr(m, "importance", 0.5), reverse=True
        )

    def _sort_by_recency(self, memories: list) -> list:
        """Sort memories by timestamp (most recent first)."""
        return sorted(memories, key=lambda m: m.timestamp, reverse=True)

    def _fit_to_budget(self, memories: list, token_budget: int) -> list:
        """Select memories that fit within token budget (FIFO by position)."""
        result = []
        char_budget = token_budget * self.CHARS_PER_TOKEN
        used = 0

        for m in memories:
            content = getattr(m, "content", "") or getattr(m, "title", "") or ""
            size = len(content) + 50
            if used + size <= char_budget:
                result.append(m)
                used += size
            else:
                break

        return result

    def _format_episode(self, node: EpisodeMemory) -> str:
        """Format episode with date and semantic content."""
        date_str = node.timestamp.strftime("%Y-%m-%d") if node.timestamp else ""
        title = self._escape(node.title) if node.title else ""
        content = self._escape(node.content or node.summary or "")[:500]

        # Build attributes
        attrs = []
        if date_str:
            attrs.append(f'date="{date_str}"')
        if title:
            attrs.append(f'title="{title}"')

        attr_str = " " + " ".join(attrs) if attrs else ""
        return f"<episode{attr_str}>{content}</episode>"

    def _format_psyche(self, node: PsycheMemory) -> str:
        """Format psyche with subtype tag and content."""
        subtype = node.psyche_type or "trait"
        content = self._escape(node.content)[:300]
        return f"<{subtype}>{content}</{subtype}>"

    def _format_note(self, node: NoteMemory) -> str:
        """Format note with type, status, and content."""
        subtype = node.note_type or "task"
        status = node.status or "active"

        # Use title if content is empty, otherwise prefer content
        text = self._escape(node.content or node.title)[:300]

        return f'<{subtype} status="{status}">{text}</{subtype}>'

    def _escape(self, text: str) -> str:
        """Escape XML special characters."""
        if not text:
            return ""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("\n", " ")
        )


# =============================================================================
# Singleton instances for convenience
# =============================================================================

_adapter = MemoryAdapter()
_formatter = ContextFormatter()


def convert_to_memories(raw_nodes: List[Dict[str, Any]]) -> List[Memory]:
    """Convenience function to convert storage dicts to domain Memory models."""
    return _adapter.from_storage_batch(raw_nodes)


def format_memories_for_llm(
    memories: List[Memory], links: List[MemoryLink] = None
) -> str:
    """Convenience function to format memories as XML context."""
    return _formatter.format_context(memories, links)
