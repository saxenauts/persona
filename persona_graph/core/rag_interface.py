from typing import List, Dict, Any
from persona_graph.core.graph_ops import GraphOps
from persona_graph.llm.llm_graph import generate_response_with_context

class RAGInterface:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.graph_ops = GraphOps()

    async def get_context(self, query: str, top_k: int = 5, max_hops: int = 2) -> str:
        similar_nodes = await self.graph_ops.perform_similarity_search(query, self.user_id, limit=top_k)
        context = await self.expand_context(similar_nodes['results'], max_hops)
        return self.format_context(context)

    async def expand_context(self, start_nodes: List[Dict[str, Any]], max_hops: int) -> Dict[str, Any]:
        context = {}
        for node in start_nodes:
            await self.explore_node(node['nodeName'], context, max_hops)
        return context

    async def explore_node(self, node_name: str, context: Dict[str, Any], hops_left: int):
        if node_name in context or hops_left < 0:
            return
        
        node_data = await self.graph_ops.get_node_data(node_name, self.user_id)
        if node_data is None:
            return  # Skip if node data is not found
        
        relationships = await self.graph_ops.get_node_relationships(node_name, self.user_id)
        
        context[node_name] = {
            'perspective': node_data.perspective,
            'properties': node_data.properties,
            'relationships': []
        }

        for rel in relationships:
            related_node = rel.target if rel.source == node_name else rel.source
            related_node_data = await self.graph_ops.get_node_data(related_node, self.user_id)
            
            if related_node_data is not None:
                relationship_info = {
                    'relation': rel.relation,
                    'related_node': related_node,
                    'related_node_perspective': related_node_data.perspective,
                    'related_node_properties': related_node_data.properties,
                    'value': getattr(rel, 'value', '')  # Use getattr in case 'value' is not present
                }
                
                context[node_name]['relationships'].append(relationship_info)
                
                if hops_left > 0:
                    await self.explore_node(related_node, context, hops_left - 1)

    def format_context(self, context: Dict[str, Any]) -> str:
        formatted = "# Knowledge Graph Context\n\n"
        for node, data in context.items():
            formatted += f"## {node}\n"
            formatted += f"Perspective: {data['perspective']}\n"
            formatted += f"Properties: {', '.join([f'{k}: {v}' for k, v in data['properties'].items()])}\n"
            formatted += "Relationships:\n"
            for rel in data['relationships']:
                formatted += f"- {rel['relation']} {rel['related_node']}\n"
                formatted += f"  Perspective: {rel['related_node_perspective']}\n"
                formatted += f"  Properties: {', '.join([f'{k}: {v}' for k, v in rel['related_node_properties'].items()])}\n"
                formatted += f"  Value: {rel['value']}\n"
            formatted += "\n"
        return formatted

    async def query(self, query: str) -> str:
        context = await self.get_context(query)
        response = await generate_response_with_context(query, context)
        return response

    async def close(self):
        await self.graph_ops.close()

    async def get_vector_context(self, query: str, top_k: int = 5) -> str:
        similar_nodes = await self.graph_ops.perform_similarity_search(query, self.user_id, limit=top_k)
        return await self.format_vector_context(similar_nodes['results'])

    async def format_vector_context(self, similar_nodes: List[Dict[str, Any]]) -> str:
        formatted = "# Vector Search Context\n\n"
        for node in similar_nodes:
            formatted += f"## {node['nodeName']}\n"
            formatted += f"Similarity Score: {node['score']}\n"
            node_data = await self.graph_ops.get_node_data(node['nodeName'], self.user_id)
            formatted += f"Perspective: {node_data.perspective}\n"
            formatted += f"Properties: {', '.join([f'{k}: {v}' for k, v in node_data.properties.items()])}\n\n"
        print(f"Vector context: {formatted}")
        return formatted

    async def query_vector_only(self, query: str) -> str:
        similar_nodes = await self.graph_ops.perform_similarity_search(query, self.user_id)
        context = await self.format_vector_context(similar_nodes['results'])
        response = await generate_response_with_context(query, context)
        return response