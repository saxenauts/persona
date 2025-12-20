"""Neo4j implementation of the GraphDatabase interface."""

from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase, basic_auth
import asyncio
import time
import json

from persona.core.interfaces import GraphDatabase
from server.config import config
from server.logging_config import get_logger

logger = get_logger(__name__)


class Neo4jGraphDatabase(GraphDatabase):
    """Neo4j implementation of GraphDatabase interface."""
    
    def __init__(self):
        self.uri = config.NEO4J.URI
        self.username = config.NEO4J.USER
        self.password = config.NEO4J.PASSWORD
        self.driver = None
    
    async def initialize(self) -> None:
        """Initialize the connection and wait for Neo4j to be ready."""
        await self._connect()
        await self._wait_for_ready()
    
    async def _connect(self) -> None:
        """Create the driver connection."""
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=basic_auth(self.username, self.password),
            max_connection_lifetime=3600
        )
    
    async def _wait_for_ready(self, timeout: int = 60) -> None:
        """Wait for Neo4j to be ready."""
        start_time = time.time()
        while True:
            try:
                if not self.driver:
                    await self._connect()
                async with self.driver.session() as session:
                    await session.run("RETURN 1")
                    logger.info("Neo4j is ready.")
                    return
            except Exception as e:
                logger.debug(f"Waiting for Neo4j... {str(e)}")
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    logger.error(f"Failed to connect to Neo4j after {timeout} seconds.")
                    raise e
                await asyncio.sleep(2)
    
    async def close(self) -> None:
        if self.driver:
            await self.driver.close()
            self.driver = None
    
    async def clean_graph(self) -> None:
        async with self.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
    
    # Node Operations
    async def create_nodes(self, nodes: List[Dict[str, Any]], user_id: str) -> None:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot create nodes.")
            return
        
        async with self.driver.session() as session:
            for node in nodes:
                node_type = node.get("type", "").replace(" ", "").replace("/", "")
                labels = "NodeName"
                if node_type:
                    labels = f"NodeName:{node_type}"
                
                query = (
                    f"MERGE (n:{labels} {{name: $name, UserId: $user_id}}) "
                    "SET n.type = $type, n.properties = $properties"
                )
                properties = json.dumps(node.get("properties", {}))
                await session.run(query, {
                    "name": node["name"],
                    "user_id": user_id,
                    "type": node.get("type", ""),
                    "properties": properties
                })
    
    async def get_node(self, node_name: str, user_id: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (n:NodeName {name: $node_name, UserId: $user_id})
        RETURN n.name AS name, n.type AS type, n.properties AS properties
        """
        async with self.driver.session() as session:
            result = await session.run(query, node_name=node_name, user_id=user_id)
            record = await result.single()
            if record:
                return {
                    "name": record["name"],
                    "type": record["type"],
                    "properties": json.loads(record["properties"]) if record["properties"] else {}
                }
            return None
    
    async def get_all_nodes(self, user_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (n:NodeName {UserId: $user_id})
        RETURN n.name AS name, n.type AS type, n.properties AS properties
        """
        async with self.driver.session() as session:
            result = await session.run(query, user_id=user_id)
            data = await result.data()
            for record in data:
                if record.get('properties'):
                    record['properties'] = json.loads(record['properties'])
                else:
                    record['properties'] = {}
            return data
    
    async def check_node_exists(self, node_name: str, node_type: str, user_id: str) -> bool:
        query = """
        MATCH (n {name: $node_name, NodeType: $node_type, UserId: $user_id})
        RETURN n.name AS NodeName
        """
        async with self.driver.session() as session:
            result = await session.run(query, node_name=node_name, node_type=node_type, user_id=user_id)
            return result.single() is not None
    
    # Relationship Operations
    async def create_relationships(self, relationships: List[Dict[str, Any]], user_id: str) -> None:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot create relationships.")
            return
        
        async with self.driver.session() as session:
            for relationship in relationships:
                query = (
                    "MATCH (source {UserId: $user_id}), (target {UserId: $user_id}) "
                    "WHERE source.name = $source AND target.name = $target "
                    "MERGE (source)-[r:`{relation}`]->(target) "
                    "SET r.value = $relation"
                )
                await session.run(query, {
                    "source": relationship["source"],
                    "target": relationship["target"],
                    "relation": relationship["relation"],
                    "user_id": user_id
                })
    
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
    
    async def get_all_relationships(self, user_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (source:NodeName {UserId: $user_id})-[r]->(target:NodeName {UserId: $user_id})
        RETURN source.name AS source, type(r) AS relation, target.name AS target
        """
        async with self.driver.session() as session:
            result = await session.run(query, user_id=user_id)
            return await result.data()
    
    # User Management
    async def create_user(self, user_id: str) -> None:
        query = """
        MERGE (u:User {id: $user_id})
        """
        logger.debug(f"Creating user {user_id} with URI: {self.uri}")
        async with self.driver.session() as session:
            await session.run(query, user_id=user_id)
        logger.info(f"User {user_id} created successfully.")
    
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
        query1 = """
        MATCH (n {UserId: $user_id})
        DETACH DELETE n
        """
        query2 = """
        MATCH (u:User {id: $user_id})
        DELETE u
        """
        async with self.driver.session() as session:
            await session.run(query1, user_id=user_id)
            await session.run(query2, user_id=user_id)
        logger.info(f"User {user_id} and all associated nodes deleted successfully.")
