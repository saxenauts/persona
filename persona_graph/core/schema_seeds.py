from persona_graph.core.graph_ops import GraphOps
from persona_graph.models.schema import GraphSchema

# Original static schemas become our seed data
SEED_SCHEMAS = [
    GraphSchema(
        name="Core Psychology",
        description="Basic psychological traits and interests schema",
        attributes=[
            'CORE_PSYCHE',
            'STABLE_INTEREST', 
            'TEMPORAL_INTEREST',
            'ACTIVE_INTEREST'
        ],
        relationships=[
            'PART_OF',          
            'RELATES_TO',       
            'LEADS_TO',         
            'INFLUENCED_BY',    
            'SIMILAR_TO'        
        ],
        is_seed=True
    )
]

async def ensure_seed_schemas(graph_ops: GraphOps):
    """Ensure default schemas exist in Neo4j"""
    existing_schemas = await graph_ops.get_all_schemas()
    existing_names = {s.name for s in existing_schemas}
    
    for schema in SEED_SCHEMAS:
        if schema.name not in existing_names:
            await graph_ops.store_schema(schema)