"""Ask Service for structured insight queries."""

from persona.core.rag_interface import RAGInterface
from persona.models.schema import AskRequest, AskResponse
from persona.llm.llm_graph import generate_structured_insights


class AskService:
    @staticmethod
    async def ask_insights(user_id: str, ask_request: AskRequest) -> AskResponse:
        async with RAGInterface(user_id) as rag:
            working_memory = await rag.get_working_memory(ask_request.query)
            structured_response = await generate_structured_insights(
                ask_request, working_memory
            )
            return AskResponse(result=structured_response)
