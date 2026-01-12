"""Context Formatting: Memory -> LLM Working Memory (Prose format with links)."""

from datetime import datetime
from typing import Optional, List, Dict, Sequence
from uuid import UUID
from pydantic import BaseModel, TypeAdapter
from persona.models.memory import (
    Memory,
    MemoryLink,
    EpisodeMemory,
    PsycheMemory,
    NoteMemory,
    UserCard,
)


class MemoryAdapter:
    _memory_adapter = TypeAdapter(Memory)

    def from_storage(self, raw: Dict) -> Memory:
        if "id" in raw and isinstance(raw["id"], str):
            from uuid import UUID as UUIDType

            try:
                raw["id"] = UUIDType(raw["id"])
            except ValueError:
                pass
        elif "name" in raw and "id" not in raw:
            from uuid import UUID as UUIDType

            try:
                raw["id"] = UUIDType(raw["name"])
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

    def from_storage_batch(self, raw_nodes: List[Dict]) -> List[Memory]:
        return [self.from_storage(r) for r in raw_nodes]


LINK_PROSE_TEMPLATES = {
    "caused_by": "caused by",
    "led_to": "led to",
    "related_to": "related to",
    "derived_from": "based on",
    "contradicts": "contradicts earlier",
    "NEXT": "followed by",
    "PREVIOUS": "preceded by",
    "supports": "supports",
    "part_of": "part of",
}


def _link_to_prose(relation: str, target: Memory) -> str:
    template = LINK_PROSE_TEMPLATES.get(relation, relation)
    target_text = target.title if target.title else target.content[:40]
    return f"{template} {target_text}"


def _build_link_context_map(
    memories: Sequence[Memory],
    links: List[MemoryLink],
) -> Dict[UUID, str]:
    memory_map = {m.id: m for m in memories}
    link_map: Dict[UUID, List[str]] = {}

    for link in links:
        if link.source_id in memory_map and link.target_id in memory_map:
            target = memory_map[link.target_id]
            prose = _link_to_prose(link.relation, target)
            if link.source_id not in link_map:
                link_map[link.source_id] = []
            link_map[link.source_id].append(prose)

    return {k: "; ".join(v) for k, v in link_map.items()}


def _format_episodes_prose(
    episodes: List[EpisodeMemory],
    link_map: Dict[UUID, str],
) -> str:
    lines = []
    for ep in sorted(episodes, key=lambda e: e.event_time, reverse=True):
        # Include year in timestamp for temporal disambiguation
        date_str = ep.event_time.strftime("%Y-%m-%d") if ep.event_time else "recent"
        content = ep.content or ep.summary or ep.title or ""
        line = f"[{date_str}] {content}"
        if ep.id in link_map:
            line += f" ({link_map[ep.id]})"
        lines.append(line)
    return "\n".join(lines)


def _format_psyche_prose(psyche: List[PsycheMemory]) -> str:
    if not psyche:
        return ""
    parts = []
    for p in sorted(psyche, key=lambda x: x.event_time, reverse=True):
        ptype = p.psyche_type or "trait"
        parts.append(f"{ptype.capitalize()}: {p.content}")
    return " ".join(parts)


def _format_notes_prose(notes: List[NoteMemory]) -> str:
    if not notes:
        return ""
    active = [n for n in notes if n.status != "COMPLETED"]
    if not active:
        return ""

    grouped: Dict[str, List[NoteMemory]] = {}
    for note in active:
        key = note.note_type or "note"
        grouped.setdefault(key, []).append(note)

    parts = []
    for note_type, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x.event_time, reverse=True)
        titles = [n.title or n.content[:50] for n in items_sorted]
        parts.append(f"{note_type.capitalize()}s: {', '.join(titles)}")

    return " ".join(parts)


def format_working_memory_prose(
    user_card: Optional[UserCard],
    episodes: List[EpisodeMemory],
    psyche: List[PsycheMemory],
    active_notes: List[NoteMemory],
    links: Optional[List[MemoryLink]] = None,
) -> str:
    sections = []
    all_memories: Sequence[Memory] = [*episodes, *psyche, *active_notes]
    link_map = _build_link_context_map(all_memories, links or [])

    if user_card and user_card.identity_prose:
        sections.append(f"<user>\n{user_card.identity_prose}\n</user>")

    if episodes:
        episode_prose = _format_episodes_prose(episodes, link_map)
        sections.append(f"<recent_context>\n{episode_prose}\n</recent_context>")

    active_parts = []
    if psyche:
        active_parts.append(_format_psyche_prose(psyche))
    if active_notes:
        notes_prose = _format_notes_prose(active_notes)
        if notes_prose:
            active_parts.append(notes_prose)

    if active_parts:
        sections.append(
            f"<active_context>\n{chr(10).join(active_parts)}\n</active_context>"
        )

    return "\n\n".join(sections)


_adapter = MemoryAdapter()
