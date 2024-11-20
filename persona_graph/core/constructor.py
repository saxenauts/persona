from persona_graph.core.graph_ops import GraphOps
from persona_graph.llm.llm_graph import get_nodes, get_relationships
from persona_graph.llm.embeddings import generate_embeddings
from persona_graph.models.schema import NodeModel, RelationshipModel, GraphUpdateModel, UnstructuredData, NodesAndRelationshipsResponse, Node, Relationship
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import asyncio

# Schema constants
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
        self.graph_ops = GraphOps()
        # Ensure the vector index is created
        await self.graph_ops.neo4j_manager.ensure_vector_index()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.graph_ops.close()

    async def clean_graph(self):
        await self.graph_ops.clean_graph()
        # Recreate the vector index after cleaning
        await self.graph_ops.neo4j_manager.ensure_vector_index()

    async def ingest_unstructured_data_to_graph(self, data: UnstructuredData):
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
        # Combine relevant fields into a single string
        preprocessed = f"{data.title}\n{data.content}\n"
        if data.metadata:
            preprocessed += "\n".join([f"{k}: {v}" for k, v in data.metadata.items()])
        return preprocessed
    
    async def get_schema_context(self) -> str:
        """Returns the static schema types and existing core/stable nodes as context"""
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
        print(f"Extracting nodes from text...")
        schema_context = await self.get_schema_context()
        nodes_response = await get_nodes(text, schema_context)
        print(f"Extracted nodes: {nodes_response}")
        return nodes_response

    async def generate_relationships(self, nodes: List[Node]) -> List[Relationship]:
        print(f"Generating relationships from nodes...")
        # graph_context = await self.get_entire_graph_context()
        schema_context = await self.get_schema_context()
        graph_context = await self.get_relevant_graph_context(nodes)
        relationships = await get_relationships(nodes, schema_context, graph_context)
        print(f"Generated relationships: {relationships}")
        return relationships
    
    async def get_relevant_graph_context(self, nodes: List[Node], max_hops: int = 2) -> str:
        """Get relevant subgraph context for the given nodes"""
        context = "# Relevant Graph Context\n\n"
        
        # Check if there are any existing nodes in the graph
        existing_nodes = await self.graph_ops.get_all_nodes(self.user_id)
        if not existing_nodes:
            print("No existing nodes in graph (t=0). Skipping context retrieval.")
            return context
        
        print(f"Found {len(existing_nodes)} existing nodes in graph")
        # print(f"Current nodes being processed: {[n.name for n in nodes]}")
        
        # Generate embeddings for all nodes in batch
        node_names = [node.name for node in nodes]
        print(f"Generating embeddings for {len(node_names)} nodes in batch")
        embeddings = generate_embeddings(node_names)
        
        # Perform similarity searches concurrently but safely
        similar_nodes = []
        tasks = []
        for node_name, embedding in zip(node_names, embeddings):
            task = asyncio.create_task(
                self.graph_ops.perform_similarity_search(
                    query=node_name,
                    embedding=embedding,
                    user_id=self.user_id,
                    limit=5
                )
            )
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks)
            for result in results:
                if result and result.get('results'):
                    print(f"Found similar nodes for {result['query']}")
                    similar_nodes.extend(result['results'])
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return context
        
        if not similar_nodes:
            print("No similar nodes found in existing graph.")
            return context
        
        # Crawl the graph from similar nodes
        subgraph = {}
        for node in similar_nodes:
            await self._explore_node(node['nodeName'], subgraph, max_hops)
        
        # Format the subgraph context
        if subgraph:
            context += "## Related Nodes and Relationships\n"
            for node_name, data in subgraph.items():
                context += f"\n### {node_name}\n"
                context += f"Perspective: {data['perspective']}\n"
                if data['relationships']:
                    context += "Relationships:\n"
                    for rel in data['relationships']:
                        context += f"- {rel}\n"
        
        return context

    async def _explore_node(self, node_name: str, subgraph: Dict, hops_left: int):
        """Helper method to explore node relationships for context building"""
        if node_name in subgraph or hops_left < 0:
            return
        
        node_data = await self.graph_ops.get_node_data(node_name, self.user_id)
        relationships = await self.graph_ops.get_node_relationships(node_name, self.user_id)
        
        subgraph[node_name] = {
            'perspective': node_data.perspective,
            'relationships': []
        }
        
        for rel in relationships:
            subgraph[node_name]['relationships'].append(
                f"{rel.source} {rel.relation} {rel.target}"
            )
            if hops_left > 0:
                next_node = rel.target if rel.source == node_name else rel.source
                await self._explore_node(next_node, subgraph, hops_left - 1)

    async def get_entire_graph_context(self) -> str:
        """
        Get graph context relative to the nodes, and user psyche.
        Currently gets the entire graph context, which is not efficient.
        """
        # TODO: This is not efficient, only get the context for the nodes that are relevant.
        # TODO: Use more efficient sophisticated context retrieval techniques like graph traversal, etc.
        # TODO: Use vector search to get the context for the nodes.
    
        nodes = await self.graph_ops.get_all_nodes(self.user_id)
        relationships = await self.graph_ops.get_all_relationships(self.user_id)
       
        context = "# Current Knowledge Graph\n\n## Nodes\n"
        for node in nodes:
            context += f"- {node.name}: {node.perspective}\n"
        
        context += "\n## Relationships\n"
        for rel in relationships:
            context += f"- {rel.source} {rel.relation} {rel.target}\n"
        
        return context

    async def get_graph_context(self, query: str):
        print(f"Getting graph context for query: {query}")
        results = await self.graph_ops.perform_similarity_search(query=query, user_id=self.user_id)
        return results

    async def close(self):
        print("Closing Neo4j connection...")
        await self.graph_ops.close()


class GraphContextRetriever:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops

    async def get_rich_context(self, query: str, user_id: str, top_k: int = 5, max_hops: int = 2) -> str:
        similar_nodes = await self.graph_ops.text_similarity_search(query=query, user_id=user_id, limit=top_k, index_name="embeddings_index")
        context = await self.crawl_graph(similar_nodes['results'], max_hops, user_id)
        return self.format_separated_context(context)

    async def crawl_graph(self, start_nodes, max_hops, user_id):
        context = {}
        for node in start_nodes:
            print(node)
            await self.explore_node(node['nodeName'], context, max_hops, user_id)
        return context

    async def explore_node(self, node_name, context, hops_left, user_id):
        if node_name in context or hops_left < 0:
            return
        
        node_data = await self.graph_ops.get_node_data(node_name, user_id)
        relationships = await self.graph_ops.get_node_relationships(node_name, user_id)
        
        context[node_name] = {
            'perspective': node_data.perspective,
            'properties': node_data.properties,
            'relationships': []
        }

        for rel in relationships:
            context[node_name]['relationships'].append(f"{rel.relation} -> {rel.target}")
            if hops_left > 0:
                await self.explore_node(rel.target, context, hops_left - 1, user_id)

    def format_separated_context(self, context):
        graph_structure = []
        node_perspectives = []

        for node, data in context.items():
            for relationship in data['relationships']:
                graph_structure.append(f"{node} -> {relationship.split(' -> ')[1]}")
            
            node_perspectives.append(f"### {node}")
            node_perspectives.append(data['perspective'])
            node_perspectives.append("")

        formatted = "# User Knowledge Graph\n\n## Graph Structure\n```\n"
        formatted += "\n".join(graph_structure)
        formatted += "\n```\n\n## Node Perspectives\n\n"
        formatted += "\n".join(node_perspectives)

        return formatted
    
