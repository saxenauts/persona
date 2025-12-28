"""RAG Service for query handling."""

import time
from typing import Optional, Dict, Any

from persona.core.rag_interface import RAGInterface
from persona.llm.client_factory import get_chat_client
from persona.llm.providers.base import ChatMessage
from persona.tools.runner import AgentRunner, create_memory_tool_registry
from persona.tools.schemas import MEMORY_TOOLS
from server.logging_config import get_logger

logger = get_logger(__name__)

AGENT_SYSTEM_PROMPT = """You are a helpful assistant with access to the user's personal memory.

You have two tools:
- recall: Ask the user's memory in natural language. Include time/topic cues in your query.
- record: Store important information the user shares. System infers type and links.

When answering questions:
1. If you have enough context from the conversation, answer directly
2. If you need to recall something about the user, use recall
3. When the user shares important facts, preferences, or experiences, use record

Be conversational and helpful. Reference specific memories when relevant."""


class RAGService:
    @staticmethod
    async def query(
        user_id: str,
        query: str,
        retrieval_query: Optional[str] = None,
        include_stats: bool = False,
    ):
        async with RAGInterface(user_id) as rag:
            response = await rag.query(
                query,
                retrieval_query=retrieval_query,
                include_stats=include_stats,
            )
            logger.debug(f"RAG service response: {response}")
            return response

    @staticmethod
    async def query_with_agent(
        user_id: str,
        query: str,
        include_stats: bool = False,
        user_timezone: str = "UTC",
        session_id: Optional[str] = None,
        max_turns: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        async with RAGInterface(user_id, user_timezone=user_timezone) as rag:
            registry = create_memory_tool_registry(
                user_id=user_id,
                graph_ops=rag.graph_ops,
                store=rag._memory_store,
                user_card=await rag._get_user_card(),
                user_timezone=user_timezone,
                session_id=session_id,
            )

            llm = get_chat_client()
            runner = AgentRunner(llm=llm, tools=MEMORY_TOOLS, registry=registry)

            messages = [
                ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT),
                ChatMessage(role="user", content=query),
            ]

            result = await runner.run(
                messages,
                temperature=0.7,
                max_turns=max_turns,
                timeout=timeout,
            )

            total_ms = (time.time() - start_time) * 1000

            response: Dict[str, Any] = {
                "answer": result.content,
                "status": result.status,
            }

            if result.can_resume:
                response["state"] = result.state

            if include_stats:
                response["stats"] = {
                    "tool_calls_made": result.tool_calls_made,
                    "turns": result.turns,
                    "usage": result.usage,
                    "total_ms": total_ms,
                }

            return response
