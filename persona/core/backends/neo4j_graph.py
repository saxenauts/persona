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

        clean_uid = user_id.replace("-", "_").replace(" ", "_")
        user_label = f"User_{clean_uid}"

        grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
        for node in nodes:
            node_type = node.get("type", "").replace(" ", "").replace("/", "")
            labels = f"NodeName:{user_label}"
            if node_type:
                labels += f":{node_type}"

            props = {}
            for k, v in node.items():
                if k == "name" or v is None:
                    continue

                is_complex = isinstance(v, dict)
                if isinstance(v, list) and v:
                    if any(isinstance(item, (dict, list)) for item in v):
                        is_complex = True

                props[k] = json.dumps(v) if is_complex else v

            grouped_rows.setdefault(labels, []).append({
                "name": node["name"],
                "props": props
            })

        async with self.driver.session() as session:
            for labels, rows in grouped_rows.items():
                if not rows:
                    continue
                query = f"""
                UNWIND $rows AS row
                MERGE (n:{labels} {{name: row.name, UserId: $user_id}})
                SET n += row.props
                """
                await session.run(query, rows=rows, user_id=user_id)
    
    async def get_node(self, node_name: str, user_id: str) -> Optional[Dict[str, Any]]:
        # Return all node properties as flat dict
        query = """
        MATCH (n:NodeName {name: $node_name, UserId: $user_id})
        RETURN n
        """
        async with self.driver.session() as session:
            result = await session.run(query, node_name=node_name, user_id=user_id)
            record = await result.single()
            if record:
                node = dict(record["n"])
                return node
            return None
    
    async def get_all_nodes(self, user_id: str) -> List[Dict[str, Any]]:
        # Return all node properties as flat dicts
        query = """
        MATCH (n:NodeName {UserId: $user_id})
        RETURN n
        """
        async with self.driver.session() as session:
            result = await session.run(query, user_id=user_id)
            records = await result.data()
            nodes = []
            for record in records:
                node = dict(record["n"])
                nodes.append(node)
            return nodes
    
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

        grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
        for relationship in relationships:
            relation_type = relationship["relation"].upper().replace(" ", "_")
            grouped_rows.setdefault(relation_type, []).append({
                "source": relationship["source"],
                "target": relationship["target"],
                "value": relationship.get("value")
            })

        async with self.driver.session() as session:
            for relation_type, rows in grouped_rows.items():
                if not rows:
                    continue
                query = f"""
                    UNWIND $rows AS row
                    MATCH (source {{UserId: $user_id, name: row.source}})
                    MATCH (target {{UserId: $user_id, name: row.target}})
                    MERGE (source)-[r:{relation_type}]->(target)
                    SET r.created_at = datetime()
                    FOREACH (_ IN CASE WHEN row.value IS NULL THEN [] ELSE [1] END |
                        SET r.value = row.value
                    )
                """
                await session.run(query, rows=rows, user_id=user_id)
    
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
