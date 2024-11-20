from persona_graph.core.neo4j_database import Neo4jConnectionManager
from persona_graph.llm.embeddings import generate_embeddings
from persona_graph.models.schema import NodeModel, RelationshipModel, GraphUpdateModel, NodesAndRelationshipsResponse, CommunityStructure, Subgraph
from typing import List, Dict, Any
import asyncio
import json
from persona_graph.llm.llm_graph import detect_communities
from collections import defaultdict

class GraphOps:
    def __init__(self):
        self.neo4j_manager = Neo4jConnectionManager()
        # self.ensure_index_task = asyncio.create_task(self.neo4j_manager.ensure_vector_index())

    async def __aenter__(self):
        await self.neo4j_manager.ensure_vector_index()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def clean_graph(self):
        # Clean the graph
        async with self.neo4j_manager.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        print("Graph cleaned.")

        # Clean the vector index if it exists
        await self.neo4j_manager.drop_vector_index("embeddings_index")

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
        # for node in nodes:
        #     await self.add_node_embedding(node.name, user_id)
        await self.add_nodes_batch_embeddings(nodes, user_id)

    async def add_node_embedding(self, node_name: str, user_id: str):
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot add node with embedding.")
            return

        print(f"Generating embedding for node: {node_name}")
        embeddings = generate_embeddings([node_name])
        if not embeddings[0]:
            print(f"Failed to generate embeddings for node: {node_name}")
            return

        await self.neo4j_manager.add_embedding_to_vector_index(node_name, embeddings[0], user_id)
        print(f"Added embedding for node: {node_name}")

    async def add_nodes_batch_embeddings(self, nodes: List[NodeModel], user_id: str):
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot add nodes.")
            return

        # First create all nodes
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
        print(f"Generating embeddings for {len(node_names)} nodes in batch")
        embeddings = generate_embeddings(node_names)
        
        # Add embeddings to nodes
        for node_name, embedding in zip(node_names, embeddings):
            if embedding:
                await self.neo4j_manager.add_embedding_to_vector_index(node_name, embedding, user_id)
                print(f"Added embedding for node: {node_name}")
            else:
                print(f"Failed to generate embedding for node: {node_name}")


    async def add_relationships(self, relationships: List[RelationshipModel], user_id: str):
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot add relationships.")
            return

        print(f"Adding relationships to the graph for user ID: {user_id}")
        relationship_dicts = [rel.dict() for rel in relationships]
        await self.neo4j_manager.create_relationships(relationship_dicts, user_id)

    async def add_node_with_embedding(self, node_name: str, user_id: str):
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot add node with embedding.")
            return

        print(f"Generating embedding for node: {node_name}")
        embeddings = generate_embeddings([node_name])
        if not embeddings[0]:
            print("Failed to generate embeddings.")
            return

        node = NodeModel(
            name=node_name,
            properties={},  # Additional properties can be added here
            embedding=embeddings[0]
        )
        await self.add_nodes([node], user_id)
        await self.neo4j_manager.add_embedding_to_vector_index(node_name, embeddings[0], user_id)

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
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot update graph.")
            return

        print(f"Updating graph with new nodes and relationships for user ID: {user_id}")
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
        print("Starting community detection")
        # Get ranked subgraphs
        subgraphs = await self.get_ranked_subgraphs(user_id)
        print(f"Found {len(subgraphs)} subgraphs")
        
        # Format for LLM
        subgraphs_text = await self.format_subgraphs_for_llm(subgraphs)
        
        # Get community structure from LLM
        community_structure = await detect_communities(subgraphs_text)
        
        # Create community structure in graph
        await self.make_communities(user_id, community_structure, subgraphs)
        print("Community structure created")