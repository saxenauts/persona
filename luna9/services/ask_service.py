from luna9.core.rag_interface import RAGInterface
from luna9.models.schema import AskRequest, AskResponse
from luna9.core.graph_ops import GraphOps
from luna9.llm.llm_graph import generate_structured_insights
from typing import Dict, Any
import instructor
from openai import AsyncOpenAI

class AskService:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops
        self.rag = RAGInterface(None)  # Will be set per request
        
    async def ask_insights(self, ask_request: AskRequest) -> AskResponse:
        """
        Generate structured insights based on the requested schema
        """
        self.rag.user_id = ask_request.user_id
        
        # Get context using existing RAG functionality
        context = await self.rag.get_context(ask_request.query)
        
        instructor_response = await generate_structured_insights(ask_request, context)
        
        # Convert the instructor response to a dictionary
        return AskResponse(result=instructor_response)