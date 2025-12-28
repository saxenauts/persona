"""Context Formatting: Memory -> LLM Working Memory (XML format)."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, TypeAdapter
from persona.models.memory import (
    Memory,
    MemoryLink,
    EpisodeMemory,
    PsycheMemory,
    NoteMemory,
    UserCard,
)


class ContextBudget(BaseModel):
    total_tokens: int = 4000
    user_card_budget: int = 300
    psyche_budget: int = 600
    episode_budget: int = 2400
    note_budget: int = 700


class MemoryAdapter:
    _memory_adapter = TypeAdapter(Memory)

    def from_storage(self, raw: Dict[str, Any]) -> Memory:
        if "id" in raw and isinstance(raw["id"], str):
            from uuid import UUID

            try:
                raw["id"] = UUID(raw["id"])
            except ValueError:
                pass
        elif "name" in raw and "id" not in raw:
            from uuid import UUID

            try:
                raw["id"] = UUID(raw["name"])
            except ValueError:
                pass

        if "timestamp" in raw and isinstance(raw["timestamp"], str):
            try:
                raw["timestamp"] = datetime.fromisoformat(
                    raw["timestamp"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        if raw.get("type") == "goal":
            raw["type"] = "note"
            if "goal_type" in raw and "note_type" not in raw:
                raw["note_type"] = raw.pop("goal_type")

        return self._memory_adapter.validate_python(raw)

    def from_storage_batch(self, raw_nodes: List[Dict[str, Any]]) -> List[Memory]:
        return [self.from_storage(r) for r in raw_nodes]


class ContextFormatter:
    CHARS_PER_TOKEN = 4

    def format_working_memory(
        self,
        memories: List[Memory],
        links: Optional[List[MemoryLink]] = None,
        max_nodes: int = 50,
        budget: Optional[ContextBudget] = None,
        user_card: Optional[UserCard] = None,
    ) -> str:
        """Format memories into XML working memory for LLM."""
        sorted_memories = sorted(
            memories,
            key=lambda m: (
                getattr(m, "importance", 0.5),
                getattr(m, "timestamp", datetime.min),
            ),
            reverse=True,
        )
        limited_memories = sorted_memories[:max_nodes]

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

        lines = ["<working_memory>"]

        if user_card:
            lines.append(self._format_user_card(user_card))

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

        lines.append("</working_memory>")
        return "\n".join(lines)

    def _format_user_card(self, card: UserCard) -> str:
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
        return sorted(
            memories, key=lambda m: getattr(m, "importance", 0.5), reverse=True
        )

    def _sort_by_recency(self, memories: list) -> list:
        return sorted(memories, key=lambda m: m.timestamp, reverse=True)

    def _fit_to_budget(self, memories: list, token_budget: int) -> list:
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
        date_str = node.timestamp.strftime("%Y-%m-%d") if node.timestamp else ""
        title = self._escape(node.title) if node.title else ""
        content = self._escape(node.content or node.summary or "")[:500]

        attrs = []
        if date_str:
            attrs.append(f'date="{date_str}"')
        if title:
            attrs.append(f'title="{title}"')

        attr_str = " " + " ".join(attrs) if attrs else ""
        return f"<episode{attr_str}>{content}</episode>"

    def _format_psyche(self, node: PsycheMemory) -> str:
        subtype = node.psyche_type or "trait"
        content = self._escape(node.content)[:300]
        return f"<{subtype}>{content}</{subtype}>"

    def _format_note(self, node: NoteMemory) -> str:
        subtype = node.note_type or "task"
        status = node.status or "active"
        text = self._escape(node.content or node.title)[:300]
        return f'<{subtype} status="{status}">{text}</{subtype}>'

    def _escape(self, text: str) -> str:
        if not text:
            return ""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("\n", " ")
        )


_adapter = MemoryAdapter()
_formatter = ContextFormatter()


def format_working_memory(
    memories: List[Memory],
    links: Optional[List[MemoryLink]] = None,
    user_card: Optional[UserCard] = None,
) -> str:
    return _formatter.format_working_memory(memories, links, user_card=user_card)
