from persona_graph.core.graph_ops import GraphOps, GraphContextRetriever
from persona_graph.llm.llm_graph import generate_response_with_context
from persona_graph.models.schema import AskRequest, AskResponse

# TODO: Feed a schema to the LLM to generate the response. 

class AskService:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops
        self.graph_context_retriever = GraphContextRetriever(self.graph_ops)

    async def ask_insights(self, ask_request: AskRequest) -> AskResponse:
        user_id = ask_request.user_id
        query = ask_request.query

        # Retrieve relevant context from the graph
        context = await self.graph_context_retriever.get_graph_context(query)

        # Generate structured insights using LLM
        insights = await generate_response_with_context(query, context)

        return AskResponse(insights=insights)