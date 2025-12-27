"""User management and UserCard generation service."""

import re
from collections import Counter
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.models.memory import UserCard, Memory
from persona.llm.client_factory import get_chat_client
from persona.llm.providers.base import ChatMessage
from server.logging_config import get_logger

logger = get_logger(__name__)


USERCARD_SYSTEM_PROMPT = """You are analyzing a user's identity from their stored memories.
Given a list of psyche items (traits, values, preferences, beliefs) and active notes,
synthesize a compact user profile.

Return valid JSON with these fields:
{
  "name": "string or null - user's name if known",
  "roles": ["list of roles/identities - e.g. 'software engineer', 'parent', 'runner'"],
  "core_values": ["list of 3-5 core values - e.g. 'work-life balance', 'continuous learning'"],
  "current_focus": ["list of current priorities/projects - from active notes"],
  "key_relationships": ["list of important people mentioned - e.g. 'partner Sarah', 'mentor John'"],
  "communication_style": "string or null - how they prefer to communicate",
  "summary": "1-2 sentence summary of who this person is",
  "identity_summary": "1-2 sentences about who they are and core values",
  "current_themes": "1-2 sentences about active life threads, projects they're working on",
  "preferences_summary": "1-2 sentences about key likes/dislikes, behavioral patterns",
  "entity_aliases": {"alias": "canonical name" - common ways user refers to people/things}
}

Be concise. Only include fields you have evidence for. Empty arrays for unknown."""


class UserService:
    @staticmethod
    async def create_user(user_id: str, graph_ops: GraphOps):
        if await graph_ops.user_exists(user_id):
            return {"message": f"User {user_id} already exists", "status": "exists"}

        await graph_ops.create_user(user_id)
        return {"message": f"User {user_id} created successfully", "status": "created"}

    @staticmethod
    async def delete_user(user_id: str, graph_ops: GraphOps):
        await graph_ops.delete_user(user_id)
        return {"message": f"User {user_id} deleted successfully"}


class UserCardService:
    def __init__(self, store: MemoryStore, graph_ops: Optional[GraphOps] = None):
        self.store = store
        self.graph_ops = graph_ops
        self.chat_client = get_chat_client()

    async def generate(
        self,
        user_id: str,
        timezone: Optional[str] = None,
    ) -> UserCard:
        psyche_memories = await self.store.get_by_type("psyche", user_id, limit=20)
        note_memories = await self.store.get_by_type("note", user_id, limit=10)
        episode_memories = await self.store.get_by_type("episode", user_id, limit=30)
        active_notes = [
            n for n in note_memories if getattr(n, "status", "active") != "COMPLETED"
        ]

        if not psyche_memories and not active_notes:
            logger.info(f"No memories for user {user_id}, returning empty UserCard")
            return UserCard(user_id=user_id, timezone=timezone)

        psyche_text = self._format_psyche(psyche_memories)
        notes_text = self._format_notes(active_notes)
        episodes_text = self._format_episodes(episode_memories[:10])

        # Compute fuzzy index fields
        all_memories = psyche_memories + note_memories + episode_memories
        dominant_memory_types = self._compute_memory_type_distribution(all_memories)
        dominant_link_types = await self._compute_link_type_distribution(user_id)
        keyword_hints = self._extract_keyword_hints(all_memories)
        temporal_anchors = self._extract_temporal_anchors(episode_memories)
        pinned_memories = self._compute_pinned_memories(all_memories)

        try:
            card_data = await self._synthesize(psyche_text, notes_text, episodes_text)
            return UserCard(
                user_id=user_id,
                timezone=timezone,
                name=card_data.get("name"),
                roles=card_data.get("roles", []),
                core_values=card_data.get("core_values", []),
                current_focus=card_data.get("current_focus", []),
                key_relationships=card_data.get("key_relationships", []),
                communication_style=card_data.get("communication_style"),
                summary=card_data.get("summary"),
                # New prose paragraphs
                identity_summary=card_data.get("identity_summary"),
                current_themes=card_data.get("current_themes"),
                preferences_summary=card_data.get("preferences_summary"),
                # Fuzzy index fields
                dominant_memory_types=dominant_memory_types,
                dominant_link_types=dominant_link_types,
                keyword_hints=keyword_hints,
                pinned_memories=pinned_memories,
                temporal_anchors=temporal_anchors,
                entity_aliases=card_data.get("entity_aliases", {}),
                updated_at=datetime.utcnow(),
                version=2,
            )
        except Exception as e:
            logger.warning(f"UserCard synthesis failed: {e}, returning basic card")
            return self._fallback_card(
                user_id,
                timezone,
                psyche_memories,
                active_notes,
                dominant_memory_types,
                dominant_link_types,
                keyword_hints,
                pinned_memories,
                temporal_anchors,
            )

    def _format_psyche(self, memories: List[Memory]) -> str:
        if not memories:
            return "No psyche memories."
        lines = []
        for m in memories:
            ptype = getattr(m, "psyche_type", "trait")
            lines.append(f"- [{ptype}] {m.content}")
        return "\n".join(lines)

    def _format_notes(self, notes: List[Memory]) -> str:
        if not notes:
            return "No active notes."
        lines = []
        for n in notes:
            ntype = getattr(n, "note_type", "task")
            lines.append(f"- [{ntype}] {n.title}: {n.content}"[:200])
        return "\n".join(lines)

    def _format_episodes(self, episodes: List[Memory]) -> str:
        if not episodes:
            return "No recent episodes."
        lines = []
        for e in episodes:
            ts = e.timestamp.strftime("%Y-%m-%d") if e.timestamp else "unknown"
            lines.append(f"- [{ts}] {e.title}: {e.content}"[:200])
        return "\n".join(lines)

    async def _synthesize(
        self, psyche_text: str, notes_text: str, episodes_text: str = ""
    ) -> dict:
        user_prompt = f"""Psyche memories:
{psyche_text}

Active notes:
{notes_text}

Recent episodes:
{episodes_text}

Synthesize into a user profile JSON."""

        messages = [
            ChatMessage(role="system", content=USERCARD_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = await self.chat_client.chat(
            messages, response_format={"type": "json_object"}
        )

        import json

        return json.loads(response.content)

    def _compute_memory_type_distribution(
        self, memories: List[Memory]
    ) -> Dict[str, float]:
        if not memories:
            return {}
        type_counts = Counter(getattr(m, "type", "unknown") for m in memories)
        total = sum(type_counts.values())
        return {t: round(c / total, 2) for t, c in type_counts.items()}

    async def _compute_link_type_distribution(self, user_id: str) -> Dict[str, float]:
        if not self.graph_ops:
            return {}
        try:
            relationships = await self.graph_ops.graph_db.get_all_relationships(user_id)
            if not relationships:
                return {}
            relation_counts = Counter(
                r.get("relation", "UNKNOWN") for r in relationships
            )
            total = sum(relation_counts.values())
            return {r: round(c / total, 2) for r, c in relation_counts.items()}
        except Exception as e:
            logger.warning(f"Failed to compute link distribution: {e}")
            return {}

    def _extract_keyword_hints(
        self, memories: List[Memory], min_importance: float = 0.5
    ) -> Dict[str, List[str]]:
        keyword_map: Dict[str, List[str]] = {}
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "about",
            "that",
            "this",
            "these",
            "those",
            "i",
            "me",
            "my",
            "you",
            "your",
            "we",
            "our",
            "they",
            "them",
            "their",
            "it",
            "its",
            "and",
            "or",
            "but",
            "if",
            "then",
            "so",
            "because",
            "when",
            "where",
            "what",
            "who",
            "how",
            "which",
            "all",
            "each",
            "every",
            "some",
            "any",
        }

        important_memories = [
            m for m in memories if getattr(m, "importance", 0.5) >= min_importance
        ]

        for mem in important_memories:
            text = f"{mem.title} {mem.content}".lower()
            words = re.findall(r"\b[a-z]{4,}\b", text)
            unique_words = set(words) - stop_words

            for word in list(unique_words)[:5]:
                if word not in keyword_map:
                    keyword_map[word] = []
                if str(mem.id) not in keyword_map[word]:
                    keyword_map[word].append(str(mem.id))

        sorted_keywords = sorted(
            keyword_map.items(), key=lambda x: len(x[1]), reverse=True
        )
        return dict(sorted_keywords[:50])

    def _extract_temporal_anchors(self, episodes: List[Memory]) -> Dict[str, str]:
        anchors: Dict[str, str] = {}
        event_patterns = [
            r"(wedding|marriage|married)",
            r"(started|began|joined).*(job|work|company|position)",
            r"(moved|relocat|moving).*(to|from)",
            r"(graduated|graduation)",
            r"(born|birthday|birth)",
            r"(trip|travel|vacation).*(to)",
            r"(surgery|operation|hospital)",
            r"(promotion|promoted)",
        ]

        for episode in episodes:
            if not episode.timestamp:
                continue
            text = f"{episode.title} {episode.content}".lower()
            date_str = episode.timestamp.strftime("%Y-%m-%d")

            for pattern in event_patterns:
                if re.search(pattern, text):
                    anchor_name = self._extract_anchor_name(text, pattern)
                    if anchor_name and anchor_name not in anchors:
                        anchors[anchor_name] = date_str
                        break

            if getattr(episode, "importance", 0.5) >= 0.8:
                title_key = re.sub(r"[^a-z0-9]+", "_", episode.title.lower()).strip("_")
                if title_key and len(title_key) > 3 and title_key not in anchors:
                    anchors[title_key] = date_str

        return dict(list(anchors.items())[:20])

    def _extract_anchor_name(self, text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text)
        if not match:
            return None
        matched = match.group(0)
        key = re.sub(r"[^a-z0-9]+", "_", matched).strip("_")
        return key if len(key) > 3 else None

    def _compute_pinned_memories(self, memories: List[Memory]) -> Dict[str, List[str]]:
        pinned: Dict[str, List[str]] = {}

        high_importance = [m for m in memories if getattr(m, "importance", 0.5) >= 0.8]

        for mem in high_importance:
            mem_type = getattr(mem, "type", "unknown")
            tag = f"high_importance_{mem_type}"
            if tag not in pinned:
                pinned[tag] = []
            pinned[tag].append(str(mem.id))

        for mem in memories:
            if getattr(mem, "type", "") == "note":
                note_type = getattr(mem, "note_type", "task")
                if note_type in ("goal", "project"):
                    tag = f"active_{note_type}s"
                    if tag not in pinned:
                        pinned[tag] = []
                    pinned[tag].append(str(mem.id))

        return {k: v[:10] for k, v in pinned.items()}

    def _fallback_card(
        self,
        user_id: str,
        timezone: Optional[str],
        psyche: List[Memory],
        notes: List[Memory],
        dominant_memory_types: Optional[Dict[str, float]] = None,
        dominant_link_types: Optional[Dict[str, float]] = None,
        keyword_hints: Optional[Dict[str, List[str]]] = None,
        pinned_memories: Optional[Dict[str, List[str]]] = None,
        temporal_anchors: Optional[Dict[str, str]] = None,
    ) -> UserCard:
        values = [
            m.content for m in psyche if getattr(m, "psyche_type", "") == "value"
        ][:3]
        traits = [
            m.content for m in psyche if getattr(m, "psyche_type", "") == "trait"
        ][:3]
        focus = [n.title for n in notes][:3]

        return UserCard(
            user_id=user_id,
            timezone=timezone,
            core_values=values,
            roles=traits,
            current_focus=focus,
            dominant_memory_types=dominant_memory_types or {},
            dominant_link_types=dominant_link_types or {},
            keyword_hints=keyword_hints or {},
            pinned_memories=pinned_memories or {},
            temporal_anchors=temporal_anchors or {},
            updated_at=datetime.utcnow(),
            version=2,
        )
