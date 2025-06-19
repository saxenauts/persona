from persona.core.rag_interface import RAGInterface
from persona.models.schema import AskRequest, AskResponse
from persona.core.graph_ops import GraphOps, GraphContextRetriever
from persona.llm.llm_graph import generate_structured_insights
from typing import Dict, Any
import instructor
from openai import AsyncOpenAI

class AskService:
    @staticmethod
    async def ask_insights(user_id: str, ask_request: AskRequest, graph_ops: GraphOps) -> AskResponse:
        """
        Generate structured insights based on the requested schema
        """
        rag = RAGInterface(user_id)
        rag.graph_ops = graph_ops
        rag.graph_context_retriever = GraphContextRetriever(graph_ops)
        
        # Get context using existing RAG functionality
        context = await rag.get_context(ask_request.query)
        
        instructor_response = await generate_structured_insights(ask_request, context)
        
        # Convert the instructor response to a dictionary
        return AskResponse(result=instructor_response)