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
  "summary": "1-2 sentence summary of who this person is"
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
    def __init__(self, store: MemoryStore):
        self.store = store
        self.chat_client = get_chat_client()

    async def generate(
        self,
        user_id: str,
        timezone: Optional[str] = None,
    ) -> UserCard:
        psyche_memories = await self.store.get_by_type("psyche", user_id, limit=20)
        note_memories = await self.store.get_by_type("note", user_id, limit=10)
        active_notes = [
            n for n in note_memories if getattr(n, "status", "active") != "COMPLETED"
        ]

        if not psyche_memories and not active_notes:
            logger.info(f"No memories for user {user_id}, returning empty UserCard")
            return UserCard(user_id=user_id, timezone=timezone)

        psyche_text = self._format_psyche(psyche_memories)
        notes_text = self._format_notes(active_notes)

        try:
            card_data = await self._synthesize(psyche_text, notes_text)
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
                updated_at=datetime.utcnow(),
            )
        except Exception as e:
            logger.warning(f"UserCard synthesis failed: {e}, returning basic card")
            return self._fallback_card(user_id, timezone, psyche_memories, active_notes)

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

    async def _synthesize(self, psyche_text: str, notes_text: str) -> dict:
        user_prompt = f"""Psyche memories:
{psyche_text}

Active notes:
{notes_text}

Synthesize into a user profile JSON."""

        messages = [
            ChatMessage(role="system", content=USERCARD_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = await self.chat_client.chat(messages, json_mode=True)

        import json

        return json.loads(response.content)

    def _fallback_card(
        self,
        user_id: str,
        timezone: Optional[str],
        psyche: List[Memory],
        notes: List[Memory],
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
            updated_at=datetime.utcnow(),
        )
