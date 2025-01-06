from luna9.core.graph_ops import GraphOps, GraphContextRetriever
from luna9.llm.llm_graph import get_nodes, get_relationships
from luna9.llm.embeddings import generate_embeddings
from luna9.models.schema import (
    NodeModel, RelationshipModel, GraphUpdateModel,
    UnstructuredData, NodesAndRelationshipsResponse, Node, Relationship
)
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import asyncio

class GraphConstructor:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.graph_ops = None
        self.schemas = []

    async def __aenter__(self):
        self.graph_ops = await GraphOps().__aenter__()
        self.graph_context_retriever = GraphContextRetriever(self.graph_ops)
        # Load all schemas
        self.schemas = await self.graph_ops.get_all_schemas()
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
        """Returns all schemas as LLM context"""
        context = "# Graph Schemas\n\n"
        
        for schema in self.schemas:
            context += f"## Schema: {schema.name}\n"
            context += f"Description: {schema.description}\n\n"
            
            context += "### Attributes (Node Types)\n"
            for attr in schema.attributes:
                context += f"- {attr}\n"
            
            context += "\n### Relationships\n"
            for rel in schema.relationships:
                context += f"- {rel}\n"
            
            context += "\n---\n\n"
        
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



    
