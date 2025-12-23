"""
RAG Service for query handling.

Uses RAGInterface which internally uses the new Retriever.
"""

from persona.core.rag_interface import RAGInterface
from server.logging_config import get_logger

logger = get_logger(__name__)


class RAGService:
    @staticmethod
    async def query(user_id: str, query: str, include_stats: bool = False):
        """
        Execute a RAG query for a user.
        
        Uses RAGInterface which handles its own resource management.
        """
        async with RAGInterface(user_id) as rag:
            response = await rag.query(query, include_stats=include_stats)
            logger.debug(f"RAG service response: {response}")
            return response
