"""RAG Interface for Persona - simplified for agent tool pattern."""

from typing import Dict, Any, Optional, Tuple, Union
import time
from persona.core.graph_ops import GraphOps
from persona.core.retrieval import Retriever
from persona.core.memory_store import MemoryStore
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.core.context import format_working_memory
from persona.models.memory import UserCard
from persona.services.user_service import UserCardService
from persona.llm.llm_graph import (
    generate_response_with_context,
    generate_response_with_context_with_stats,
)
from server.logging_config import get_logger

logger = get_logger(__name__)


class RAGInterface:
    def __init__(self, user_id: str, user_timezone: str = "UTC"):
        self.user_id = user_id
        self.user_timezone = user_timezone
        self.graph_ops: Optional[GraphOps] = None
        self._memory_store: Optional[MemoryStore] = None
        self._retriever: Optional[Retriever] = None
        self._graph_db: Optional[Neo4jGraphDatabase] = None
        self._user_card: Optional[UserCard] = None
        self._user_card_service: Optional[UserCardService] = None

    async def __aenter__(self):
        self.graph_ops = await GraphOps().__aenter__()

        self._graph_db = Neo4jGraphDatabase()
        await self._graph_db.initialize()
        self._memory_store = MemoryStore(self._graph_db)

        self._retriever = Retriever(
            user_id=self.user_id, store=self._memory_store, graph_ops=self.graph_ops
        )
        self._user_card_service = UserCardService(
            self._memory_store, graph_ops=self.graph_ops
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.graph_ops:
            await self.graph_ops.__aexit__(exc_type, exc_val, exc_tb)
        if self._graph_db:
            await self._graph_db.close()

    async def get_working_memory(
        self,
        query: str,
        top_k: int = 5,
        hop_depth: int = 1,
        include_static: bool = True,
    ) -> str:
        if not self._retriever:
            await self.__aenter__()
        assert self._retriever is not None

        result = await self._retriever.get_working_memory(
            query=query, top_k=top_k, hop_depth=hop_depth, include_static=include_static
        )
        working_memory = result if isinstance(result, str) else result[0]

        logger.info(
            f"RAGInterface: got {len(working_memory)} chars working memory for query: {query[:50]}..."
        )
        return working_memory

    async def _get_user_card(self) -> Optional[UserCard]:
        if self._user_card:
            return self._user_card
        if self._user_card_service:
            try:
                self._user_card = await self._user_card_service.generate(
                    self.user_id, timezone=self.user_timezone
                )
                logger.info(
                    f"Generated UserCard for {self.user_id}: {self._user_card.summary or 'no summary'}"
                )
            except Exception as e:
                logger.warning(f"UserCard generation failed: {e}")
        return self._user_card

    async def query(
        self,
        query: str,
        retrieval_query: Optional[str] = None,
        include_stats: bool = False,
    ) -> Dict[str, Any]:
        if not self._retriever:
            await self.__aenter__()
        assert self._retriever is not None

        user_card = await self._get_user_card()
        search_query = retrieval_query or query

        retrieval_stats: Optional[Dict[str, Any]] = None
        retrieval_start = time.time()
        if include_stats:
            result = await self._retriever.get_working_memory(
                query=search_query,
                user_card=user_card,
                user_timezone=self.user_timezone,
                collect_stats=True,
            )
            working_memory, retrieval_stats = result  # type: ignore
        else:
            result = await self._retriever.get_working_memory(
                query=search_query,
                user_card=user_card,
                user_timezone=self.user_timezone,
            )
            working_memory = result if isinstance(result, str) else result[0]
        retrieval_ms = (time.time() - retrieval_start) * 1000

        logger.info(f"Working memory for RAG query: {working_memory[:200]}...")

        if include_stats:
            generation_start = time.time()
            answer, llm_stats = await generate_response_with_context_with_stats(
                query, working_memory
            )
            generation_ms = (time.time() - generation_start) * 1000
            retrieval_stats = retrieval_stats or {}
            retrieval_stats["working_memory_preview"] = working_memory[:1000]
            return {
                "answer": answer,
                "model": llm_stats.get("model"),
                "usage": llm_stats.get("usage"),
                "temperature": llm_stats.get("temperature"),
                "prompt_tokens": llm_stats.get("prompt_tokens"),
                "completion_tokens": llm_stats.get("completion_tokens"),
                "working_memory_chars": len(working_memory),
                "retrieval": retrieval_stats,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
            }

        answer = await generate_response_with_context(query, working_memory)
        return {"answer": answer}

    async def close(self):
        await self.__aexit__(None, None, None)
