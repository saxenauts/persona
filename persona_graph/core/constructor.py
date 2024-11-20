from persona_graph.core.graph_ops import GraphOps
from persona_graph.llm.llm_graph import get_entities, get_nodes_and_relationships
from persona_graph.models.schema import NodeModel, RelationshipModel, GraphUpdateModel, UnstructuredData, NodesAndRelationshipsResponse
from typing import List, Dict, Any, Tuple
from collections import defaultdict

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

    async def process_unstructured_data(self, data: UnstructuredData):
        structured_data = self.preprocess_data(data)
        entities = await self.extract_entities(structured_data)
        nodes, relationships = await self.generate_nodes_and_relationships(entities)
        
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

    async def extract_entities(self, text: str) -> List[str]:
        print(f"Extracting entities from text...")
        entities_response = await get_entities(text)
        entities = entities_response.get('entities', [])
        print(f"Extracted entities: {entities}")
        return entities

    async def generate_nodes_and_relationships(self, entities: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        print(f"Generating nodes and relationships from entities...")
        graph_context = await self.get_entire_graph_context()
        nodes, relationships = await get_nodes_and_relationships(entities, graph_context)
        
        print(f"Generated nodes: {nodes}")
        print(f"Generated relationships: {relationships}")
        return nodes, relationships

    async def get_entire_graph_context(self) -> str:
        nodes = await self.graph_ops.get_all_nodes(self.user_id)
        relationships = await self.graph_ops.get_all_relationships(self.user_id)
       
        print (nodes)
        print (relationships)
        
        context = "# Current Knowledge Graph\n\n## Nodes\n"
        for node in nodes:
            context += f"- {node.name}: {node.perspective}\n"
        
        context += "\n## Relationships\n"
        for rel in relationships:
            context += f"- {rel.source} {rel.relation} {rel.target}\n"
        
        return context

    async def add_nodes_and_relationships(self, nodes: List[Dict[str, Any]], relationships: List[Dict[str, Any]]):
        print(f"Adding nodes and relationships to the graph...")
        node_models = [NodeModel(name=node['name'], label=node.get('label', 'Entity')) for node in nodes] if nodes else []
        relationship_models = [RelationshipModel(source=rel['source'], target=rel['target'], relation=rel['relation']) for rel in relationships] if relationships else []
        graph_update = GraphUpdateModel(nodes=node_models, relationships=relationship_models)
        await self.graph_ops.update_graph(graph_update, self.user_id)

    async def get_graph_context(self, query: str):
        print(f"Getting graph context for query: {query}")
        results = await self.graph_ops.perform_similarity_search(query, self.user_id)
        return results

    async def close(self):
        print("Closing Neo4j connection...")
        await self.graph_ops.close()





class GraphContextRetriever:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops

    async def get_rich_context(self, query: str, user_id: str, top_k: int = 5, max_hops: int = 2) -> str:
        similar_nodes = await self.graph_ops.perform_similarity_search(query=query, user_id=user_id)
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
    
