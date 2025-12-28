"""User management and UserCard generation service."""

from datetime import datetime
from typing import Optional, List

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.models.memory import UserCard, Memory
from persona.llm.client_factory import get_chat_client
from persona.llm.providers.base import ChatMessage
from server.logging_config import get_logger

logger = get_logger(__name__)


USERCARD_SYSTEM_PROMPT = """Synthesize a 2-3 sentence identity summary from the user's memories.

Write in third person, present tense. Focus on:
- Who they are (role, context)
- What matters to them (values, priorities)
- Current life threads (projects, goals, themes)

Be concise and natural. Write prose, not lists."""


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

    async def generate(self, user_id: str, timezone: Optional[str] = None) -> UserCard:
        psyche = await self.store.get_by_type("psyche", user_id, limit=15)
        notes = await self.store.get_by_type("note", user_id, limit=10)
        episodes = await self.store.get_by_type("episode", user_id, limit=10)

        active_notes = [
            n for n in notes if getattr(n, "status", "active") != "COMPLETED"
        ]

        if not psyche and not active_notes and not episodes:
            logger.info(f"No memories for user {user_id}, returning empty UserCard")
            return UserCard(user_id=user_id, timezone=timezone)

        try:
            identity_prose = await self._synthesize_prose(
                psyche, active_notes, episodes
            )
            return UserCard(
                user_id=user_id,
                timezone=timezone,
                identity_prose=identity_prose,
                updated_at=datetime.utcnow(),
                version=2,
            )
        except Exception as e:
            logger.warning(f"UserCard synthesis failed: {e}, returning fallback")
            return self._fallback_card(user_id, timezone, psyche, active_notes)

    async def _synthesize_prose(
        self,
        psyche: List[Memory],
        notes: List[Memory],
        episodes: List[Memory],
    ) -> str:
        memory_text = self._format_memories(psyche, notes, episodes)

        messages = [
            ChatMessage(role="system", content=USERCARD_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"Memories:\n{memory_text}\n\nWrite identity summary:",
            ),
        ]

        response = await self.chat_client.chat(messages)
        return response.content.strip()

    def _format_memories(
        self,
        psyche: List[Memory],
        notes: List[Memory],
        episodes: List[Memory],
    ) -> str:
        lines = []

        for m in psyche[:10]:
            ptype = getattr(m, "psyche_type", "trait")
            lines.append(f"[{ptype}] {m.content}")

        for n in notes[:5]:
            ntype = getattr(n, "note_type", "task")
            lines.append(f"[{ntype}] {n.title}: {n.content}"[:150])

        for e in episodes[:5]:
            ts = e.timestamp.strftime("%Y-%m-%d") if e.timestamp else ""
            lines.append(f"[{ts}] {e.content}"[:150])

        return "\n".join(lines)

    def _fallback_card(
        self,
        user_id: str,
        timezone: Optional[str],
        psyche: List[Memory],
        notes: List[Memory],
    ) -> UserCard:
        parts = []

        traits = [
            m.content for m in psyche if getattr(m, "psyche_type", "") == "trait"
        ][:2]
        if traits:
            parts.append(f"Traits: {', '.join(traits)}.")

        focus = [n.title for n in notes][:2]
        if focus:
            parts.append(f"Current focus: {', '.join(focus)}.")

        return UserCard(
            user_id=user_id,
            timezone=timezone,
            identity_prose=" ".join(parts) if parts else "",
            updated_at=datetime.utcnow(),
            version=2,
        )
