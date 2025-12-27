from datetime import datetime
from typing import Optional, List, Dict, Any, Type, Union
from pydantic import TypeAdapter
from persona.models.memory import (
    Memory,
    MemoryLink,
    EpisodeMemory,
    PsycheMemory,
    NoteMemory,
)


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
    """
    Formats retrieved memories into LLM-optimized context.

    Design principles:
    1. Semantic content first - no IDs for general consumption
    2. Shallow, flat structure with explicit XML tags
    3. Group by type for easy scanning
    """

    def format_context(
        self,
        memories: List[Memory],
        links: List[MemoryLink] = None,
        max_nodes: int = 50,
    ) -> str:
        """
        Build LLM context from memories.
        """
        limited_memories = memories[:max_nodes]

        # Group by type
        episodes = [m for m in limited_memories if isinstance(m, EpisodeMemory)]
        psyches = [m for m in limited_memories if isinstance(m, PsycheMemory)]
        notes = [m for m in limited_memories if isinstance(m, NoteMemory)]

        lines = ["<memory_context>"]

        # Episodes - temporal memories
        if episodes:
            lines.append("<episodes>")
            for ep in episodes:
                lines.append(self._format_episode(ep))
            lines.append("</episodes>")

        # Psyche - identity/preference memories
        if psyches:
            lines.append("<psyche>")
            for p in psyches:
                lines.append(self._format_psyche(p))
            lines.append("</psyche>")

        # Notes - structured/unstructured items (tasks, facts, lists, etc.)
        if notes:
            lines.append("<notes>")
            for n in notes:
                lines.append(self._format_note(n))
            lines.append("</notes>")

        lines.append("</memory_context>")
        return "\n".join(lines)

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
