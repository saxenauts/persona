from persona.core.rag_interface import RAGInterface
from persona.models.schema import AskRequest, AskResponse
from persona.core.graph_ops import GraphOps
from persona.llm.llm_graph import generate_structured_insights
from typing import Dict, Any
import instructor
from openai import AsyncOpenAI

class AskService:
    @staticmethod
    async def ask_insights(ask_request: AskRequest) -> AskResponse:
        """
        Generate structured insights based on the requested schema
        """
        async with RAGInterface(ask_request.user_id) as rag:
            # Get context using existing RAG functionality
            context = await rag.get_context(ask_request.query)
            
            instructor_response = await generate_structured_insights(ask_request, context)
            
            # Convert the instructor response to a dictionary
            return AskResponse(result=instructor_response)