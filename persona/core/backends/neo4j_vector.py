"""Neo4j implementation of the VectorStore interface."""

from typing import List, Dict, Any

from persona.core.interfaces import VectorStore
from server.config import config
from server.logging_config import get_logger
from neo4j import AsyncGraphDatabase, basic_auth

logger = get_logger(__name__)


class Neo4jVectorStore(VectorStore):
    """Neo4j implementation of VectorStore interface.
    
    Uses Neo4j's built-in vector index for similarity search.
    """
    
    def __init__(self, graph_driver=None):
        """Initialize Neo4jVectorStore.
        
        Args:
            graph_driver: Optional existing Neo4j driver to share connection.
                          If None, creates its own connection.
        """
        self.uri = config.NEO4J.URI
        self.username = config.NEO4J.USER
        self.password = config.NEO4J.PASSWORD
        self._shared_driver = graph_driver is not None
        self.driver = graph_driver
        self.index_name = "embeddings_index"
    
    async def initialize(self) -> None:
        """Initialize and ensure vector index exists."""
        if not self.driver:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=basic_auth(self.username, self.password),
                max_connection_lifetime=3600
            )
        await self._ensure_vector_index()
    
    async def close(self) -> None:
        """Close connection if we own it."""
        if not self._shared_driver and self.driver:
            await self.driver.close()
            self.driver = None
    
    async def _ensure_vector_index(self) -> None:
        """Create vector index if it doesn't exist."""
        async with self.driver.session() as session:
            result = await session.run("SHOW VECTOR INDEXES")
            indexes = await result.data()
            index_exists = any(idx['name'] == self.index_name for idx in indexes)
            
            if not index_exists:
                query = """
                CREATE VECTOR INDEX embeddings_index
                FOR (n:NodeName)
                ON (n.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
                """
                try:
                    await session.run(query)
                    logger.info(f"Vector index '{self.index_name}' created.")
                except Exception as e:
                    if "EquivalentSchemaRuleAlreadyExists" in str(e):
                        logger.debug(f"Vector index '{self.index_name}' already exists.")
                    else:
                        raise e
            else:
                logger.debug(f"Vector index '{self.index_name}' already exists.")
    
    async def add_embedding(self, node_name: str, embedding: List[float], user_id: str) -> None:
        """Add or update embedding for a node."""
        if not self._validate_embedding(embedding):
            logger.error(f"Invalid embedding format for node {node_name}.")
            return
        
        query = """
        MATCH (n {name: $node_name, UserId: $user_id})
        CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
        """
        async with self.driver.session() as session:
            await session.run(query, node_name=node_name, embedding=embedding, user_id=user_id)
    
    async def search_similar(
        self, 
        embedding: List[float], 
        user_id: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Returns:
            List of dicts with keys: node_name, score
        """
        query = """
        CALL db.index.vector.queryNodes($indexName, $limit, $embedding)
        YIELD node, score
        WHERE node.UserId = $user_id
        RETURN node.name AS node_name, score
        ORDER BY score DESC
        """
        results = []
        async with self.driver.session() as session:
            tx = await session.begin_transaction()
            try:
                result = await tx.run(
                    query, 
                    indexName=self.index_name, 
                    embedding=embedding, 
                    user_id=user_id,
                    limit=limit
                )
                async for record in result:
                    results.append({
                        'node_name': record['node_name'],
                        'score': record['score']
                    })
                await tx.commit()
            except Exception as e:
                await tx.rollback()
                raise
            finally:
                await tx.close()
        return results
    
    @staticmethod
    def _validate_embedding(embedding: List[float]) -> bool:
        """Validate embedding format."""
        return isinstance(embedding, list) and all(isinstance(item, float) for item in embedding)
    
    async def drop_index(self) -> None:
        """Drop the vector index (for testing)."""
        async with self.driver.session() as session:
            result = await session.run("SHOW VECTOR INDEXES")
            indexes = await result.data()
            if any(idx['name'] == self.index_name for idx in indexes):
                await session.run(f"DROP INDEX `{self.index_name}`")
                logger.info(f"Vector index '{self.index_name}' dropped.")
