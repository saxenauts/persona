"""Unified Persona Service for memory-augmented dialogue."""

import time
from typing import Optional, Dict, Any

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.adapters.persona_adapter import PersonaAdapter
from persona.models.memory import UserCard, Memeplex
from persona.services.consolidation_service import get_or_generate_usercard
from persona.llm.client_factory import get_chat_client
from persona.llm.providers.base import ChatMessage
from persona.llm.prompts import PERSONAL_AI_SYSTEM_PROMPT
from persona.tools.runner import AgentRunner, REGISTRY
from persona.tools.context import ToolContext
from persona.tools.schemas import MEMORY_TOOLS
from server.logging_config import get_logger

logger = get_logger(__name__)


class PersonaService:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops
        self._memory_store: Optional[MemoryStore] = None

    @property
    def memory_store(self) -> MemoryStore:
        if self._memory_store is None:
            self._memory_store = MemoryStore(self.graph_ops.graph_db)
        return self._memory_store

    async def run_agent(
        self,
        user_id: str,
        query: str,
        include_stats: bool = False,
        user_timezone: str = "UTC",
        session_id: Optional[str] = None,
        max_turns: Optional[int] = None,
        timeout: Optional[float] = None,
        output_schema: Optional[dict] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        user_card = await self._get_user_card(user_id, user_timezone)
        memeplex = await self._get_memeplex(user_id)

        user_card_present = bool(user_card and user_card.identity_prose)
        memeplex_present = bool(memeplex)

        world_model, user_context = self._build_user_context(user_card, memeplex)
        system_prompt = PERSONAL_AI_SYSTEM_PROMPT.format(
            world_model=world_model, user_context=user_context
        )

        adapter = PersonaAdapter(user_id=user_id, graph_ops=self.graph_ops)

        ctx = ToolContext(
            user_id=user_id,
            graph_ops=self.graph_ops,
            store=self.memory_store,
            adapter=adapter,
            session_id=session_id,
            user_timezone=user_timezone,
            user_card=user_card,
        )

        llm = get_chat_client()
        runner = AgentRunner(llm=llm, tools=MEMORY_TOOLS, registry=REGISTRY)

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=query),
        ]

        agent_result = await runner.run(
            messages,
            ctx=ctx,
            temperature=0.7,
            max_turns=max_turns,
            timeout=timeout,
        )

        if output_schema and agent_result.status == "completed":
            final_result = await self._finalize_with_schema(
                conversation=agent_result.content,
                query=query,
                schema=output_schema,
            )
            result_content = final_result
        else:
            result_content = agent_result.content

        total_ms = (time.time() - start_time) * 1000

        response: Dict[str, Any] = {
            "answer": result_content,
            "status": agent_result.status,
        }

        if agent_result.can_resume:
            response["state"] = agent_result.state

        if include_stats:
            response["stats"] = {
                "tool_calls_made": agent_result.tool_calls_made,
                "turns": agent_result.turns,
                "usage": agent_result.usage,
                "total_ms": total_ms,
                "tool_results": agent_result.tool_results,
                "user_card_present": user_card_present,
                "memeplex_present": memeplex_present,
                "world_model_chars": len(world_model) if world_model else 0,
                "user_context_chars": len(user_context) if user_context else 0,
            }

        return response

    async def _get_user_card(
        self, user_id: str, user_timezone: str
    ) -> Optional[UserCard]:
        try:
            user_card = await get_or_generate_usercard(
                user_id=user_id,
                graph_ops=self.graph_ops,
                user_timezone=user_timezone,
            )
            if user_card:
                logger.info(
                    f"UserCard for {user_id}: {user_card.identity_prose[:50] if user_card.identity_prose else 'empty'}"
                )
            return user_card
        except Exception as e:
            logger.warning(f"UserCard fetch failed: {e}")
            return None

    async def _get_memeplex(self, user_id: str) -> Optional[Memeplex]:
        try:
            memeplex = await self.memory_store.get_memeplex(user_id)
            if memeplex:
                logger.debug(
                    f"Loaded Memeplex for {user_id}: {len(memeplex.index)} chars"
                )
            return memeplex
        except Exception as e:
            logger.warning(f"Memeplex load failed: {e}")
            return None

    def _build_user_context(
        self, user_card: Optional[UserCard], memeplex: Optional[Memeplex]
    ) -> tuple[str, str]:
        world_model = ""
        user_context = ""

        if memeplex:
            world_model = memeplex.to_system_prompt()

        if user_card and user_card.identity_prose:
            user_context = f"## Who They Are\n\n{user_card.identity_prose}"

        return world_model, user_context

    async def ask(
        self,
        user_id: str,
        query: str,
        output_schema: dict,
        user_timezone: str = "UTC",
    ) -> Dict[str, Any]:
        """Agentic retrieval + structured output extraction.

        Uses run_agent() for retrieval, then extracts structured data.
        This ensures consistent retrieval path across all query types.
        """
        result = await self.run_agent(
            user_id=user_id,
            query=query,
            user_timezone=user_timezone,
            output_schema=output_schema,
        )

        # Transform response to match AskResponse format
        answer = result.get("answer", {})
        if isinstance(answer, dict):
            return {"result": answer}
        else:
            # If answer is a string (structured extraction failed), wrap it
            return {"result": {"raw": answer}}

    async def _finalize_with_schema(
        self,
        conversation: str,
        query: str,
        schema: dict,
    ) -> Dict[str, Any]:
        import json

        llm = get_chat_client()

        finalization_prompt = f"""Based on the following conversation, extract structured data.

User Query: {query}

Conversation: {conversation}

Extract the requested information as JSON matching: {json.dumps(schema)}"""

        messages = [ChatMessage(role="user", content=finalization_prompt)]

        if llm.supports_json_mode():
            result = await llm.chat(messages, response_format={"type": "json_object"})
            content = result.content or "{}"
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON: {content}")
                return {"error": "Invalid JSON from LLM"}
        else:
            result = await llm.chat(messages)
            content = result.content or ""
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse structured output: {content}")
                return {"error": "Failed to extract structured data"}
