from persona_graph.core.neo4j_database import Neo4jConnectionManager
from persona_graph.llm.embeddings import generate_embeddings
from persona_graph.models.schema import NodeModel, RelationshipModel, GraphUpdateModel, NodesAndRelationshipsResponse
from typing import List, Dict, Any
import asyncio
import json

class GraphOps:
    def __init__(self):
        self.neo4j_manager = Neo4jConnectionManager()
        self.ensure_index_task = asyncio.create_task(self.neo4j_manager.ensure_vector_index())

    async def __aenter__(self):
        await self.ensure_index_task
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
        for node in nodes:
            await self.add_node_embedding(node.name, user_id)

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

    async def perform_similarity_search(self, query: str, user_id: str, limit: int = 5, index_name: str = "embeddings_index") -> Dict[str, Any]:
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