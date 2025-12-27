"""
RAG Service for query handling.

Uses RAGInterface which internally uses the new Retriever.
"""

from persona.core.rag_interface import RAGInterface
from server.logging_config import get_logger

logger = get_logger(__name__)


class RAGService:
    @staticmethod
    async def query(
        user_id: str,
        query: str,
        retrieval_query: str = None,
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
