from persona_graph.core.graph_ops import GraphOps
from persona_graph.models.schema import PersonalizeRequest, PersonalizeResponse

# TODO: Constraint the personalization to a schema. Use an example. 


class PersonalizeServiceBYOA:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops

    async def personalize_user(self, personalize_request: PersonalizeRequest) -> PersonalizeResponse:
        user_id = personalize_request.user_id
        criteria = personalize_request.personalization_criteria

        # Apply personalization algorithms based on criteria
        # This could involve updating user preferences, modifying graph nodes/relationships, etc.
        # Placeholder for actual implementation
        # What do we even want to do here?
        # await self.graph_ops.apply_personalization(user_id, criteria)

        return PersonalizeResponse(status="Success", details="Personalization criteria applied.")