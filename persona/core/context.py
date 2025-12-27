from datetime import datetime
from typing import Optional, List, Dict, Any, Type, Union
from pydantic import BaseModel, TypeAdapter
from persona.models.memory import (
    Memory,
    MemoryLink,
    EpisodeMemory,
    PsycheMemory,
    NoteMemory,
)


class ContextBudget(BaseModel):
    """Token budget allocation for context building.

    Inspired by SillyTavern's "golden reserve" pattern - always include
    core identity traits, then fill remaining budget by recency/relevance.
    """

    total_tokens: int = 4000
    psyche_budget: int = 800
    episode_budget: int = 2500
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
    ) -> str:
        """Build LLM context from memories with optional token budget."""
        limited_memories = memories[:max_nodes]

        episodes = [m for m in limited_memories if isinstance(m, EpisodeMemory)]
        psyches = [m for m in limited_memories if isinstance(m, PsycheMemory)]
        notes = [m for m in limited_memories if isinstance(m, NoteMemory)]

        if budget:
            episodes = self._fit_to_budget(episodes, budget.episode_budget)
            psyches = self._fit_to_budget(psyches, budget.psyche_budget)
            notes = self._fit_to_budget(notes, budget.note_budget)

        lines = ["<memory_context>"]

        if episodes:
            lines.append("<episodes>")
            for ep in episodes:
                lines.append(self._format_episode(ep))
            lines.append("</episodes>")

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

        lines.append("</memory_context>")
        return "\n".join(lines)

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
