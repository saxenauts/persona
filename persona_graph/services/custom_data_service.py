from persona_graph.core.graph_ops import GraphOps
from persona_graph.models.schema import CustomGraphUpdate, CustomNodeData, CustomRelationshipData, NodesAndRelationshipsResponse
from persona_graph.models.schema import NodeModel, RelationshipModel
from typing import Dict, Any
    
class CustomDataService:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops
    
    async def update_custom_data(self, user_id: str, update: CustomGraphUpdate) -> Dict[str, Any]:
        """
        Add or update custom structured data using existing GraphOps
        """
        try:
            # Convert CustomNodeData to NodeModel
            nodes = [
                NodeModel(
                    name=node.name,
                    perspective=node.perspective,
                    properties=node.properties
                ) for node in update.nodes
            ]
            
            # Convert CustomRelationshipData to RelationshipModel
            relationships = [
                RelationshipModel(
                    source=rel.source,
                    target=rel.target,
                    relation=rel.relation_type  # Map relation_type to relation
                ) for rel in update.relationships
            ]
            
            # Use existing GraphOps to update the graph
            await self.graph_ops.update_graph(
                NodesAndRelationshipsResponse(
                    nodes=nodes,
                    relationships=relationships
                ),
                user_id
            )
            
            return {
                "status": "success",
                "message": f"Updated {len(nodes)} nodes and {len(relationships)} relationships"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }