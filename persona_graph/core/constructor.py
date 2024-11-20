from persona_graph.core.graph_ops import GraphOps, GraphContextRetriever
from persona_graph.llm.llm_graph import get_nodes, get_relationships
from persona_graph.llm.embeddings import generate_embeddings
from persona_graph.models.schema import (
    NodeModel, RelationshipModel, GraphUpdateModel,
    UnstructuredData, NodesAndRelationshipsResponse, Node, Relationship
)
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import asyncio

# Schema constants
# Static for now
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


class GraphConstructor:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.graph_ops = None

    async def __aenter__(self):
        self.graph_ops = await GraphOps().__aenter__()
        self.graph_context_retriever = GraphContextRetriever(self.graph_ops)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.graph_ops.__aexit__(exc_type, exc_val, exc_tb)

    async def clean_graph(self):
        await self.graph_ops.clean_graph()

    async def ingest_unstructured_data_to_graph(self, data: UnstructuredData):
        """
        Ingest unstructured data into the graph.
        Preprocess the data, extract nodes, generate relationships, and update the graph.
        """
        text = self.preprocess_data(data)
        nodes = await self.extract_nodes(text)
        relationships = await self.generate_relationships(nodes)
        
        if not nodes and not relationships:
            print("No nodes or relationships generated from the unstructured data.")
            return
        
        nodes = [NodeModel(name=node.name, perspective=node.perspective) for node in nodes]
        
        relationships = [RelationshipModel(source=rel.source, target=rel.target, relation=rel.relation) 
                         for rel in relationships]
        
        graph_update = NodesAndRelationshipsResponse(
            nodes=nodes,
            relationships=relationships
        )
        
        # Update the graph
        await self.graph_ops.update_graph(graph_update, self.user_id)

    def preprocess_data(self, data: UnstructuredData) -> str:
        """
        Preprocess the data, combine relevant fields into a single string.
        """
        preprocessed = f"{data.title}\n{data.content}\n"
        if data.metadata:
            preprocessed += "\n".join([f"{k}: {v}" for k, v in data.metadata.items()])
        return preprocessed
    
    async def get_schema_context(self) -> str:
        """
        Returns the static graph schema types and existing core/stable nodes as LLM context.
        """
        # Get existing core and stable nodes
        print(f"Getting schema context for user {self.user_id}")
        
        context = "# Graph Schema\n\n"
        
        # Add node types
        context += "## Node Types\n"
        for node_type in NODE_TYPES:
            context += f"- {node_type}\n"
        
        # Add relationship types
        context += "\n## Relationship Types\n"
        for rel_type in RELATIONSHIP_TYPES:
            context += f"- {rel_type}\n"
        
        return context

    async def extract_nodes(self, text: str) -> List[Node]:
        """
        Extract nodes from the unstructured text referring to the graph schema context.
        """
        schema_context = await self.get_schema_context()
        nodes_response = await get_nodes(text, schema_context)
        return nodes_response

    async def generate_relationships(self, nodes: List[Node]) -> List[Relationship]:
        """
        Generate relationships from the nodes based on the schema context and graph context.
        """
        schema_context = await self.get_schema_context()
        graph_context = await self.get_relevant_graph_context(user_id=self.user_id, nodes=nodes)
        relationships = await get_relationships(nodes, schema_context, graph_context)
        return relationships
    
    async def get_relevant_graph_context(self, user_id: str, nodes: List[Node], max_hops: int = 2) -> str:
        """
        Get relevant subgraph context for the given nodes.
        """
        return await self.graph_context_retriever.get_relevant_graph_context(nodes=nodes, user_id=user_id, max_hops=max_hops)

    async def close(self):
        print("Closing Neo4j connection...")
        await self.graph_ops.close()



    
