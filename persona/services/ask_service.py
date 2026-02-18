"""
Ask Service for structured insight queries.

Uses RAGInterface which internally uses the new Retriever.
"""

from persona.core.rag_interface import RAGInterface
from persona.models.schema import AskRequest, AskResponse
from persona.llm.llm_graph import generate_structured_insights


class AskService:
    @staticmethod
    async def ask_insights(user_id: str, ask_request: AskRequest) -> AskResponse:
        """
        Generate structured insights based on the requested schema.
        
        Uses RAGInterface which handles its own resource management.
        """
        async with RAGInterface(user_id) as rag:
            # Get context using new Retriever
            context = await rag.get_context(ask_request.query)
            
            # Generate structured response
            structured_response = await generate_structured_insights(ask_request, context)
            
            return AskResponse(result=structured_response)