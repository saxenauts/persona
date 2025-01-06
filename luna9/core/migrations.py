from luna9.models.schema import GraphSchema
from luna9.core.graph_ops import GraphOps

# Seed schema constants
NODE_TYPES = [
    'CORE_PSYCHE',
    'STABLE_INTEREST', 
    'TEMPORAL_INTEREST',
    'ACTIVE_INTEREST'
]

RELATIONSHIP_TYPES = [
    'PART_OF',          # Hierarchical relationships
    'RELATES_TO',       # Cross-domain connections
    'LEADS_TO',         # Learning progression
    'INFLUENCED_BY',    # Impact relationships
    'SIMILAR_TO'        # Similarity connections
]



SEED_SCHEMAS = [
    GraphSchema(
        name="Core Psychology",
        description="Basic psychological traits and interests",
        attributes=NODE_TYPES,  # From original static schema
        relationships=RELATIONSHIP_TYPES,  # From original static schema
        is_seed=True
    ),
    # Add more seed schemas as needed
]

async def ensure_seed_schemas(graph_ops: GraphOps):
    """Ensure seed schemas exist in the database"""
    existing_schemas = await graph_ops.get_all_schemas()
    existing_names = {s.name for s in existing_schemas}
    
    for schema in SEED_SCHEMAS:
        if schema.name not in existing_names:
            await graph_ops.store_schema(schema)