"""Unified Persona Service for memory-augmented dialogue."""

import json
import time
from typing import Optional, Dict, Any

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.core.retrieval import Retriever
from persona.models.memory import UserCard
from persona.services.user_service import UserCardService
from persona.llm.client_factory import get_chat_client
from persona.llm.providers.base import ChatMessage
from persona.llm.llm_graph import (
    generate_response_with_context,
    generate_response_with_context_with_stats,
)
from persona.tools.runner import AgentRunner, REGISTRY
from persona.tools.context import ToolContext
from persona.tools.schemas import MEMORY_TOOLS
from server.logging_config import get_logger

logger = get_logger(__name__)

AGENT_SYSTEM_PROMPT = """You are a helpful assistant with access to the user's personal memory graph.

## Memory Graph Schema
- **Episode**: Events, conversations, experiences (what happened)
- **Psyche**: Traits, preferences, values, personality (who they are)
- **Note**: Goals, tasks, reminders (what they want to do)
- **Relationships**: LED_TO, CAUSED_BY, NEXT, PREVIOUS, RELATES_TO

## Available Tools

**recall(query)** - Search memories by semantic similarity + time cues.
Include time references ("yesterday", "last week", "2024-01-15") and topic keywords.
Call multiple times in parallel for different queries.

**record(text)** - Store new information. System infers type (episode/psyche/note) and creates links automatically.

**expand_neighbors(memory_id, relationship_types?)** - Explore graph connections from a memory.
Use after recall() to find related memories via graph relationships.
Optional: filter by relationship types like ["LED_TO", "CAUSED_BY"].

**follow_relationship(source_id, relation_type, limit?)** - Trace specific relationship chains.
Use to follow causal chains (LED_TO), temporal sequences (NEXT/PREVIOUS), or thematic connections.

## Strategy
1. If you have enough context from conversation, answer directly
2. For questions about the user, use recall with relevant time/topic cues
3. For deep exploration, use expand_neighbors or follow_relationship on interesting results
4. When user shares important facts/preferences/experiences, use record
5. Reference specific memories when relevant to build trust"""


class PersonaService:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops
        self._memory_store: Optional[MemoryStore] = None
        self._user_card_service: Optional[UserCardService] = None

    @property
    def memory_store(self) -> MemoryStore:
        if self._memory_store is None:
            self._memory_store = MemoryStore(self.graph_ops.graph_db)
        return self._memory_store

    @property
    def user_card_service(self) -> UserCardService:
        if self._user_card_service is None:
            self._user_card_service = UserCardService(
                self.memory_store, graph_ops=self.graph_ops
            )
        return self._user_card_service

    async def query(
        self,
        user_id: str,
        query: str,
        include_stats: bool = False,
        user_timezone: str = "UTC",
    ) -> Dict[str, Any]:
        wm_start = time.time()

        user_card = await self._get_user_card(user_id, user_timezone)
        retriever = Retriever(
            user_id=user_id, store=self.memory_store, graph_ops=self.graph_ops
        )
        working_memory = await retriever.get_working_memory(
            user_card=user_card,
            user_timezone=user_timezone,
        )
        if not isinstance(working_memory, str):
            working_memory = working_memory[0]

        wm_ms = (time.time() - wm_start) * 1000
        logger.info(f"Working memory for query: {working_memory[:200]}...")

        if include_stats:
            gen_start = time.time()
            answer, llm_stats = await generate_response_with_context_with_stats(
                query, working_memory
            )
            gen_ms = (time.time() - gen_start) * 1000

            return {
                "answer": answer,
                "model": llm_stats.get("model"),
                "usage": llm_stats.get("usage"),
                "temperature": llm_stats.get("temperature"),
                "prompt_tokens": llm_stats.get("prompt_tokens"),
                "completion_tokens": llm_stats.get("completion_tokens"),
                "working_memory_chars": len(working_memory),
                "working_memory_ms": wm_ms,
                "generation_ms": gen_ms,
            }

        answer = await generate_response_with_context(query, working_memory)
        return {"answer": answer}

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

        # Create per-request ToolContext (not cached - context is stateless)
        ctx = ToolContext(
            user_id=user_id,
            graph_ops=self.graph_ops,
            store=self.memory_store,
            session_id=session_id,
            user_timezone=user_timezone,
            user_card=user_card,
        )

        llm = get_chat_client()
        runner = AgentRunner(llm=llm, tools=MEMORY_TOOLS, registry=REGISTRY)

        messages = [
            ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT),
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
            }

        return response

    async def _get_user_card(
        self, user_id: str, user_timezone: str
    ) -> Optional[UserCard]:
        try:
            user_card = await self.user_card_service.generate(
                user_id, timezone=user_timezone
            )
            logger.info(
                f"Generated UserCard for {user_id}: {user_card.identity_prose[:50] if user_card.identity_prose else 'empty'}"
            )
            return user_card
        except Exception as e:
            logger.warning(f"UserCard generation failed: {e}")
            return None

    async def ask(
        self,
        user_id: str,
        query: str,
        output_schema: dict,
        user_timezone: str = "UTC",
    ) -> Dict[str, Any]:
        """Direct retrieval + structured LLM extraction (no agent loop, no tools)."""
        import json

        user_card = await self._get_user_card(user_id, user_timezone)
        retriever = Retriever(
            user_id=user_id, store=self.memory_store, graph_ops=self.graph_ops
        )
        working_memory = await retriever.get_working_memory(
            user_card=user_card,
            user_timezone=user_timezone,
        )
        if not isinstance(working_memory, str):
            working_memory = working_memory[0]

        llm = get_chat_client()

        extraction_prompt = f"""Extract structured data from the user's memory.

User Query: {query}

User Memory Context:
{working_memory}

Return a JSON object matching this structure: {json.dumps(output_schema)}

Extract only factual information found in the memory. If information is not found, use null."""

        messages = [ChatMessage(role="user", content=extraction_prompt)]

        if llm.supports_json_mode():
            result = await llm.chat(messages, response_format={"type": "json_object"})
            content = result.content or "{}"
            try:
                parsed = json.loads(content)
                return {"result": parsed}
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {content}")
                return {"result": {"error": "Invalid JSON from LLM", "raw": content}}
        else:
            result = await llm.chat(messages)
            content = result.content or ""
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                parsed = json.loads(content)
                return {"result": parsed}
            except json.JSONDecodeError:
                logger.error(f"Failed to parse structured output: {content}")
                return {
                    "result": {
                        "error": "Failed to extract structured data",
                        "raw": content,
                    }
                }

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
