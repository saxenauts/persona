from persona_graph.core.neo4j_database import Neo4jConnectionManager
from persona_graph.llm.embeddings import generate_embeddings
from persona_graph.models.schema import (
    NodeModel, RelationshipModel, GraphUpdateModel, NodesAndRelationshipsResponse, 
    CommunityStructure, Subgraph, Node, Relationship
)
from typing import List, Dict, Any
import asyncio
import json
from persona_graph.llm.llm_graph import detect_communities
from collections import defaultdict

class GraphOps:
    def __init__(self):
        self.neo4j_manager = Neo4jConnectionManager()

    async def __aenter__(self):
        await self.neo4j_manager.ensure_vector_index()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def clean_graph(self):
        # Clean the graph
        await self.neo4j_manager.clean_graph()

    async def add_nodes(self, nodes: List[NodeModel], user_id: str):
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot add nodes.")
            return

        node_dicts = [
            {
                "name": node.name,
                "perspective": node.perspective or "",
                "properties": node.properties.dict() if node.properties else {}
            }
            for node in nodes
        ]
        await self.neo4j_manager.create_nodes(node_dicts, user_id)
        
        # Generate and add embeddings for new nodes
        await self.add_nodes_batch_embeddings(nodes, user_id)


    async def add_nodes_batch_embeddings(self, nodes: List[NodeModel], user_id: str):
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot add nodes.")
            return

        # Create all nodes
        node_dicts = [
            {
                "name": node.name,
                "perspective": node.perspective or "",
                "properties": node.properties.dict() if node.properties else {}
            }
            for node in nodes
        ]
        await self.neo4j_manager.create_nodes(node_dicts, user_id)
        
        # Generate embeddings for all nodes in one batch
        node_names = [node.name for node in nodes]
        embeddings = generate_embeddings(node_names)
        
        # Add embeddings to nodes
        for node_name, embedding in zip(node_names, embeddings):
            if embedding:
                await self.neo4j_manager.add_embedding_to_vector_index(node_name, embedding, user_id)
            else:
                print(f"Failed to generate embedding for node: {node_name}")


    async def add_relationships(self, relationships: List[RelationshipModel], user_id: str):
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot add relationships.")
            return

        relationship_dicts = [rel.dict() for rel in relationships]
        await self.neo4j_manager.create_relationships(relationship_dicts, user_id)

    async def get_node_data(self, node_name: str, user_id: str) -> NodeModel:
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot get node data.")
            return NodeModel(name=node_name, perspective="", properties={})

        node_data = await self.neo4j_manager.get_node_data(node_name, user_id)
        if node_data:
            return NodeModel(
                name=node_data["name"],
                perspective=node_data["perspective"],
                properties=node_data["properties"]
            )
        return NodeModel(name=node_name, perspective="", properties={})

    async def get_node_relationships(self, node_name: str, user_id: str) -> List[RelationshipModel]:
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot get node relationships.")
            return []

        relationships = await self.neo4j_manager.get_node_relationships(node_name, user_id)
        return [RelationshipModel(source=rel["source"], target=rel["target"], relation=rel["relation"]) 
                for rel in relationships]

    async def text_similarity_search(self, query: str, user_id: str, limit: int = 5, index_name: str = "embeddings_index") -> Dict[str, Any]:
        """
        Perform a similarity search on the graph based on a text query. 
        """
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot perform similarity search.")
            return {"query": query, "results": []}

        print(f"Generating embedding for query: '{query}' for user ID: '{user_id}'")
        query_embeddings = generate_embeddings([query])
        if not query_embeddings[0]:
            return {"query": query, "results": []}

        print(f"Performing similarity search for the query: '{query}' for user ID: '{user_id}'")
        results = await self.neo4j_manager.query_text_similarity(query_embeddings[0], user_id)

        return {
            "query": query,
            "results": [
                {
                    "nodeId": result["nodeId"],
                    "nodeName": result["nodeName"],
                    "score": result["score"]
                } for result in results
            ]
        }
    
    async def update_graph(self, graph_update: NodesAndRelationshipsResponse, user_id: str):
        """
        Update the graph with new nodes and relationships
        """
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot update graph.")
            return

        if graph_update.nodes:
            await self.add_nodes(graph_update.nodes, user_id)
        if graph_update.relationships:
            await self.add_relationships(graph_update.relationships, user_id)
        if not graph_update.nodes and not graph_update.relationships:
            print("No nodes or relationships to update.")        

    async def close(self):
        print("Closing Neo4j connection...")
        await self.neo4j_manager.close()

    async def get_all_nodes(self, user_id: str) -> List[NodeModel]:
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot get all nodes.")
            return []

        nodes = await self.neo4j_manager.get_all_nodes(user_id)
        return [NodeModel(name=node['name'], perspective=node['perspective']) for node in nodes]

    async def get_all_relationships(self, user_id: str) -> List[RelationshipModel]:
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot get all relationships.")
            return []

        relationships = await self.neo4j_manager.get_all_relationships(user_id)
        return [RelationshipModel(source=rel['source'], target=rel['target'], relation=rel['relation']) for rel in relationships]
    
    async def create_user(self, user_id: str) -> None:
        await self.neo4j_manager.create_user(user_id)

    async def delete_user(self, user_id: str) -> None:
        await self.neo4j_manager.delete_user(user_id)

    async def user_exists(self, user_id: str) -> bool:
        return await self.neo4j_manager.user_exists(user_id)

    async def perform_similarity_search(
        self, 
        query: str, 
        embedding: List[float],
        user_id: str, 
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Perform similarity search using pre-computed embedding
        """
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot perform similarity search.")
            return {"query": query, "results": []}

        try:
            print(f"Performing similarity search for: {query}")
            results = await self.neo4j_manager.query_text_similarity(embedding, user_id)
            print(f"Found {len(results)} similar nodes for '{query}'")
            
            return {
                "query": query,
                "results": [
                    {
                        "nodeId": result["nodeId"],
                        "nodeName": result["nodeName"],
                        "score": result["score"]
                    } for result in results
                ]
            }
        except Exception as e:
            print(f"Error in similarity search for {query}: {str(e)}")
            return {"query": query, "results": []}
        

    async def get_ranked_subgraphs(self, user_id: str) -> List[Subgraph]:
        """Get all subgraphs in the graph, ranked by size and influence"""
        subgraphs = []
        visited_nodes = set()

        # Retrieve all nodes for the user
        all_nodes = await self.neo4j_manager.get_all_nodes(user_id)

        for node in all_nodes:
            node_name = node['name']
            if node_name not in visited_nodes:
                # Retrieve all relationships for this node
                relationships = await self.neo4j_manager.get_node_relationships(node_name, user_id)

                # Collect all connected nodes
                connected_nodes = {node_name}
                for rel in relationships:
                    connected_nodes.add(rel['source'])
                    connected_nodes.add(rel['target'])

                # Mark nodes as visited
                visited_nodes.update(connected_nodes)

                # Calculate central nodes based on degree
                central_nodes = await self._get_central_nodes(list(connected_nodes), relationships)

                subgraphs.append(Subgraph(
                    id=len(subgraphs),  # Assign a unique ID based on the order
                    nodes=list(connected_nodes),
                    relationships=relationships,
                    size=len(connected_nodes),
                    central_nodes=central_nodes
                ))

        # Sort subgraphs by size in descending order
        subgraphs.sort(key=lambda sg: sg.size, reverse=True)

        return subgraphs

    async def _get_central_nodes(self, nodes: List[str], relationships: List[Dict]) -> List[str]:
        """Calculate central nodes based on degree centrality"""
        if not nodes:  # Handle empty subgraph case
            return []
            
        degree_count = defaultdict(int)
        for rel in relationships:
            degree_count[rel['source']] += 1
            degree_count[rel['target']] += 1
        
        # If we have nodes with relationships, return the most central one
        if degree_count:
            sorted_nodes = sorted(degree_count.items(), key=lambda x: x[1], reverse=True)
            # Return only the node name (first element of the tuple)
            return [sorted_nodes[0][0]]
        
        # If no relationships exist, return the first node as representative
        return [nodes[0]]
    

    async def format_subgraphs_for_llm(self, subgraphs: List[Subgraph]) -> str:
        """Format subgraphs into a string for LLM input"""
        formatted = "# Graph Structure Analysis\n\n"
        
        for subgraph in subgraphs:
            formatted += f"\n## Subgraph {subgraph.id} (Size: {subgraph.size})\n"
            formatted += f"Central Nodes: {', '.join(subgraph.central_nodes)}\n"
            formatted += "\nNodes:\n"
            for node in subgraph.nodes:
                formatted += f"- {node}\n"
            
            formatted += "\nRelationships:\n"
            for rel in subgraph.relationships:
                formatted += f"- {rel['source']} {rel['relation']} {rel['target']}\n"
        
        return formatted

    async def make_communities(self, user_id: str, community_structure: CommunityStructure, subgraphs: List[Subgraph]) -> None:
        """Create community structure in the graph"""
        for header in community_structure.communityHeaders:
            # Create header node
            await self.neo4j_manager.create_nodes([{
                'name': header.header,
                'perspective': "Community header representing " + header.header,
                'type': 'community_head'
            }], user_id)
            
            for subheader in header.subheaders:
                # Create subheader node
                await self.neo4j_manager.create_nodes([{
                    'name': subheader.subheader,
                    'perspective': f"Subheader under {header.header}",
                    'type': 'subheader'
                }], user_id)
                
                # Connect header to subheader
                await self.neo4j_manager.create_relationships([{
                    'source': header.header,
                    'target': subheader.subheader,
                    'relation': 'HAS_SUBHEADER'
                }], user_id)
                
                # Connect representative nodes from subgraphs to this subheader
                for subgraph_id in subheader.subgraph_ids:
                    if subgraph_id < len(subgraphs):
                        subgraph = subgraphs[subgraph_id]
                        for central_node in subgraph.central_nodes:
                            await self.neo4j_manager.create_relationships([{
                                'source': central_node,
                                'target': subheader.subheader,
                                'relation': 'BELONGS_TO'
                            }], user_id)

    async def community_detection(self, user_id) -> None:
        """Main orchestrator for community detection process"""
        # Get ranked subgraphs
        subgraphs = await self.get_ranked_subgraphs(user_id)
        
        # Format for LLM
        subgraphs_text = await self.format_subgraphs_for_llm(subgraphs)
        
        # Get community structure from LLM
        community_structure = await detect_communities(subgraphs_text)
        
        # Create community structure in graph
        await self.make_communities(user_id, community_structure, subgraphs)



class GraphContextRetriever:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops

    async def get_rich_context(self, query: str, user_id: str, top_k: int = 5, max_hops: int = 2) -> str:
        """
        Get rich context from the graph based on a text query.
        """
        similar_nodes = await self.graph_ops.text_similarity_search(query=query, user_id=user_id, limit=top_k, index_name="embeddings_index")
        context = await self.crawl_graph(similar_nodes['results'], max_hops, user_id)
        return self.format_separated_context(context)

    async def crawl_graph(self, start_nodes, max_hops, user_id):
        """
        Crawl the graph to get rich context.
        """
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
    
    async def get_relevant_graph_context(self, nodes: List[Node], user_id: str, max_hops: int = 2) -> str:
        """
        Get relevant subgraph context for the given nodes.
        Similarity Search, then Graph Crawl to get subgraph based context.
        """
        context = "# Relevant Graph Context\n\n"
        
        # Check if there are any existing nodes in the graph
        existing_nodes = await self.graph_ops.get_all_nodes(user_id)
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
                    user_id=user_id,
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
            await self._explore_node(node_name = node['nodeName'], subgraph=subgraph, user_id=user_id, max_hops=max_hops)
        
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

    async def _explore_node(self, node_name: str, subgraph: Dict, user_id: str, max_hops: int):
        """Helper method to explore node relationships for context building"""
        if node_name in subgraph or max_hops < 0:
            return
        
        node_data = await self.graph_ops.get_node_data(node_name, user_id)
        relationships = await self.graph_ops.get_node_relationships(node_name, user_id)
        
        subgraph[node_name] = {
            'perspective': node_data.perspective,
            'relationships': []
        }
        
        for rel in relationships:
            subgraph[node_name]['relationships'].append(
                f"{rel.source} {rel.relation} {rel.target}"
            )
            if max_hops > 0:
                next_node = rel.target if rel.source == node_name else rel.source
                await self._explore_node(next_node, subgraph, user_id, max_hops - 1)

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