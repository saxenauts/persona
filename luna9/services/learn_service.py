from luna9.llm.llm_graph import get_nodes, get_relationships
from luna9.core.graph_ops import GraphOps
from luna9.models.schema import LearnRequest, LearnResponse, GraphUpdateModel

# TODO: This is not the right implementation. Make the graph constructor take in the schema
# and then use that schema to extract the nodes and relationships from the user's interactions.


class LearnService:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops

    async def learn_user(self, learn_request: LearnRequest) -> LearnResponse:
        """
        Store a new schema for graph construction.
        The schema will be used by the constructor when processing user data.
        """
        # Store the new schema
        schema_id = await self.graph_ops.store_schema(learn_request.graph_schema)
        
        return LearnResponse(
            status="Success",
            schema_id=schema_id,
            details=f"Schema '{learn_request.graph_schema.name}' stored successfully. It will be used in future graph construction."
        )