from typing import List, Dict, Any, Union, Tuple
from neo4j import AsyncGraphDatabase, basic_auth
import asyncio
import time
from persona.llm.embeddings import generate_embeddings
import json
from app_server.config import config


class Neo4jConnectionManager:
    def __init__(self):
        self.uri = config.NEO4J.URI
        self.username = config.NEO4J.USER
        self.password = config.NEO4J.PASSWORD
        print("URI: ", self.uri)
        print("Username: ", self.username)
        print("Password: ", self.password)
        self.driver = None
        self.ensure_vector_index_task = None

    async def initialize(self):
        """Initialize the connection and wait for Neo4j to be ready"""
        await self.connect()
        await self.wait_for_neo4j()
        await self.ensure_vector_index()

    async def connect(self):
        """Create the driver connection"""
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=basic_auth(self.username, self.password),
            max_connection_lifetime=3600
        )

    async def wait_for_neo4j(self, timeout=60):
        """Wait for Neo4j to be ready"""
        start_time = time.time()
        while True:
            try:
                if not self.driver:
                    await self.connect()
                async with self.driver.session() as session:
                    await session.run("RETURN 1")
                    print("Neo4j is ready.")
                    return
            except Exception as e:
                print(f"Waiting for Neo4j... {str(e)}")
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    print(f"Failed to connect to Neo4j after {timeout} seconds.")
                    raise e
                await asyncio.sleep(2)  # Increased sleep time

    async def close(self):
        if self.driver:
            await self.driver.close()
            self.driver = None

    async def clean_graph(self) -> None:
        # Delete all nodes and relationships
        async with self.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")

        # Drop the vector index
        await self.drop_vector_index("embeddings_index")

    async def check_node_exists(self, node_name: str, node_type: str, user_id: str) -> bool:
        query = """
        MATCH (n {name: $node_name, NodeType: $node_type, UserId: $user_id})
        RETURN n.name AS NodeName
        """
        async with self.driver.session() as session:
            result = await session.run(query, node_name=node_name, node_type=node_type, user_id=user_id)
            return result.single() is not None

    async def clean_graph(self) -> None:
        async with self.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        await self.drop_vector_index("embeddings_index")

    async def drop_vector_index(self, index_name: str) -> None:
        if await self.index_exists(index_name):
            query = f"DROP INDEX `{index_name}`"
            async with self.driver.session() as session:
                await session.run(query)
            print(f"Vector index '{index_name}' dropped.")
        else:
            print(f"Vector index '{index_name}' does not exist. Skipping drop operation.")

    async def create_nodes(self, nodes: List[Dict[str, Any]], user_id: str) -> None:
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot create nodes.")
            return
        async with self.driver.session() as session:
            for node in nodes:
                query = (
                    "MERGE (n:NodeName {name: $name, UserId: $user_id}) "
                    "SET n.perspective = $perspective, n.properties = $properties"
                )
                properties = json.dumps(node.get("properties", {}))  # Serialize properties to JSON string
                await session.run(query, {
                    "name": node["name"],
                    "user_id": user_id,
                    "perspective": node.get("perspective", ""),
                    "properties": properties
                })

    async def create_relationships(self, relationships: List[Dict[str, Any]], user_id: str) -> None:
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot create relationships.")
            return
        async with self.driver.session() as session:
            for relationship in relationships:
                await self._create_relationship(session, relationship, user_id)

    async def _create_relationship(self, session, relationship: Dict[str, Any], user_id: str) -> None:
        query = (
            "MATCH (source {UserId: $user_id}), (target {UserId: $user_id}) "
            "WHERE source.name = $source AND target.name = $target "
            "MERGE (source)-[r:`{relation}`]->(target) "
            "SET r.value = $relation"
        )
        params = {
            "source": relationship["source"],
            "target": relationship["target"],
            "relation": relationship["relation"],
            "user_id": user_id
        }

        await session.run(query, params)

    async def create_vector_index(self, index_name: str) -> None:
        # Check if the index already exists
        existing_indexes_query = "SHOW VECTOR INDEXES"
        async with self.driver.session() as session:
            existing_indexes = await session.run(existing_indexes_query)
            index_exists = any(index['name'] == index_name for index in await existing_indexes.data())
    
        if not index_exists:
            query = f"""
            CREATE VECTOR INDEX `{index_name}`
            FOR (n:NodeName) ON (n.embedding)
            OPTIONS {{indexConfig: {{`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}}}
            """
            async with self.driver.session() as session:
                await session.run(query)
            print(f"Vector index '{index_name}' created.")
        else:
            print(f"Vector index '{index_name}' already exists.")

    async def add_embedding_to_vector_index(self, node_name: str, embedding: List[float], user_id: str) -> None:
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot add embedding.")
            return
        query = """
        MATCH (n {name: $node_name, UserId: $user_id})
        CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
        """
        async with self.driver.session() as session:
            await session.run(query, node_name=node_name, embedding=embedding, user_id=user_id)

    async def index_exists(self, index_name: str) -> bool:
        async with self.driver.session() as session:
            existing_indexes = await session.run("SHOW VECTOR INDEXES")
            index_exists = any(index['name'] == index_name for index in await existing_indexes.data())
            return index_exists

    async def ensure_vector_index(self) -> None:
        async with self.driver.session() as session:
            # Check if the index exists
            result = await session.run("SHOW VECTOR INDEXES")
            indexes = await result.data()
            index_exists = any(index['name'] == 'embeddings_index' for index in indexes)

            if not index_exists:
                # Create the index if it doesn't exist
                query = """
                CREATE VECTOR INDEX embeddings_index IF NOT EXISTS
                FOR (n:NodeName)
                ON (n.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
                """
                await session.run(query)
                print("Vector index 'embeddings_index' created.")
            else:
                print("Vector index 'embeddings_index' already exists.")

    async def query_text_similarity(self, keyword_embedding: List[float], user_id: str, index_name: str = "embeddings_index") -> List[Dict[str, Any]]:
        """
        Query the Neo4j vector index to find the top 5 nodes similar to a given text keyword embedding, filtered by user ID.

        Args:
        - keyword_embedding (List[float]): The embedding of the text keyword as a list of floats.
        - user_id (str): The user ID to filter the nodes by.
        - index_name (str): The name of the vector index used for querying.

        Returns:
        - List[Dict[str, Any]]: A list of dictionaries containing the node ID, node name, and their similarity scores.
        """
        query = """
        CALL db.index.vector.queryNodes($indexName, 5, $embedding)
        YIELD node, score
        WHERE node.UserId = $user_id
        RETURN elementId(node) AS nodeId, node.name AS nodeName, score
        ORDER BY score DESC
        """
        results = []
        async with self.driver.session() as session:
            tx = await session.begin_transaction()
            try:
                result = await tx.run(query, indexName=index_name, embedding=keyword_embedding, user_id=user_id)
                async for record in result:
                    results.append({
                        'nodeId': record['nodeId'],
                        'nodeName': record['nodeName'],
                        'score': record['score']
                    })
                await tx.commit()
            except Exception as e:
                await tx.rollback()
                raise
            finally:
                await tx.close()
        return results


    async def update_node_embeddings(self, node_name: str, embedding: List[float], user_id: str) -> None:
        if not await self.user_exists(user_id):
            print(f"User {user_id} does not exist. Cannot update embeddings.")
            return
        if not self._validate_embedding(embedding):
            print(f"Invalid embedding format for node {node_name}. Embedding must be a list of floats.")
            return

        async with self.driver.session() as session:
            success_flag = await self._set_node_embedding(session=session, embedding=embedding, node_name=node_name, user_id=user_id)

    async def _set_node_embedding(self, session, embedding: List[float], node_name: str, user_id: str) -> bool:
        query = """
        MATCH (n {name: $node_name, UserId: $user_id})
        SET n.embedding = $embedding
        RETURN n.embedding IS NOT NULL AS successFlag
        """
        print("embedding", embedding)
        result = await session.run(query, embedding=embedding, node_name=node_name, user_id=user_id)
        result_data = await result.single()
        return result_data['successFlag'] if result_data else False

    @staticmethod
    def _validate_embedding(embedding: List[float]) -> bool:
        return isinstance(embedding, list) and all(isinstance(item, float) for item in embedding)

    async def get_node_data(self, node_name: str, user_id: str) -> Dict[str, Any]:
        query = """
        MATCH (n:NodeName {name: $node_name, UserId: $user_id})
        RETURN n.name AS name, n.perspective AS perspective, n.properties AS properties
        """
        async with self.driver.session() as session:
            result = await session.run(query, node_name=node_name, user_id=user_id)
            record = await result.single()
            if record:
                return {
                    "name": record["name"],
                    "perspective": record["perspective"],
                    "properties": json.loads(record["properties"]) if record["properties"] else {}
                }
            return None

    async def get_node_relationships(self, node_name: str, user_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (n:NodeName {name: $node_name, UserId: $user_id})-[r]-(m:NodeName)
        RETURN type(r) AS relation, m.name AS related_node, r.value AS value,
               CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS direction
        """
        async with self.driver.session() as session:
            result = await session.run(query, node_name=node_name, user_id=user_id)
            return [
                {
                    "source": node_name if record["direction"] == "outgoing" else record["related_node"],
                    "target": record["related_node"] if record["direction"] == "outgoing" else node_name,
                    "relation": record["relation"],
                    "value": record["value"]
                } 
                for record in await result.data()
            ]

    async def get_all_nodes(self, user_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (n:NodeName {UserId: $user_id})
        RETURN n.name AS name, n.perspective AS perspective
        """
        async with self.driver.session() as session:
            result = await session.run(query, user_id=user_id)
            return await result.data()

    async def get_all_relationships(self, user_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (source:NodeName {UserId: $user_id})-[r]->(target:NodeName {UserId: $user_id})
        RETURN source.name AS source, type(r) AS relation, target.name AS target
        """
        async with self.driver.session() as session:
            result = await session.run(query, user_id=user_id)
            return await result.data()

    async def create_user(self, user_id: str) -> None:
        query = """
        MERGE (u:User {id: $user_id})
        """
        print("URI: ", self.uri)
        async with self.driver.session() as session:
            await session.run(query, user_id=user_id)
        print(f"User {user_id} created successfully.")

    async def user_exists(self, user_id: str) -> bool:
        query = """
        MATCH (u:User {id: $user_id})
        RETURN COUNT(u) > 0 AS exists
        """
        async with self.driver.session() as session:
            result = await session.run(query, user_id=user_id)
            record = await result.single()
            return record and record['exists']

    async def delete_user(self, user_id: str) -> None:
        query = """
        MATCH (n {UserId: $user_id})
        DETACH DELETE n
        """
        async with self.driver.session() as session:
            await session.run(query, user_id=user_id)
        print(f"User {user_id} and all associated nodes deleted successfully.")


