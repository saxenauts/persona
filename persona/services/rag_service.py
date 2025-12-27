"""
RAG Service for query handling.

Uses RAGInterface which internally uses the new Retriever.
"""

from typing import Optional
from persona.core.rag_interface import RAGInterface
from server.logging_config import get_logger

logger = get_logger(__name__)


class RAGService:
    @staticmethod
    async def query(
        user_id: str,
        query: str,
        retrieval_query: Optional[str] = None,
        include_stats: bool = False,
        use_router: bool = False,
    ):
        """
        Query with RAG retrieval.

        Args:
            user_id: User identifier.
            query: Natural language query.
            retrieval_query: Optional override for retrieval (uses query if None).
            include_stats: Return detailed stats.
            use_router: Use IntentRouter for spray-and-pray retrieval.
        """
        async with RAGInterface(user_id) as rag:
            response = await rag.query(
                query,
                retrieval_query=retrieval_query,
                include_stats=include_stats,
                use_router=use_router,
            )
            logger.debug(f"RAG service response: {response}")
            return response
