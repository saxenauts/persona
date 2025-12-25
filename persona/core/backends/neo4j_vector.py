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
        # Global index is deprecated in favor of per-user indexes
        # self.index_name = "embeddings_index" 
    
    async def initialize(self) -> None:
        """Initialize connection."""
        if not self.driver:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=basic_auth(self.username, self.password),
                max_connection_lifetime=3600
            )
        # No global index initialization needed anymore
    
    async def close(self) -> None:
        """Close connection if we own it."""
        if not self._shared_driver and self.driver:
            await self.driver.close()
            self.driver = None
            
    def _get_user_label(self, user_id: str) -> str:
        """Get the dynamic label for a user's nodes."""
        # Sanitize simple just in case, though backticks handle most
        clean_id = user_id.replace("-", "_").replace(" ", "_")
        return f"User_{clean_id}"

    def _get_index_name(self, user_id: str) -> str:
        """Get the dynamic vector index name for a user."""
        clean_id = user_id.replace("-", "_").replace(" ", "_")
        return f"vector_idx_{clean_id}"
    
    async def _ensure_user_index(self, user_id: str) -> None:
        """Ensure a vector index exists for this specific user."""
        index_name = self._get_index_name(user_id)
        user_label = self._get_user_label(user_id)
        
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{user_label})
        ON (n.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        async with self.driver.session() as session:
            try:
                await session.run(query)
                # logger.debug(f"Ensured vector index '{index_name}' exists.")
            except Exception as e:
                logger.error(f"Failed to create index {index_name}: {e}")
                raise e
    
    async def add_embedding(self, node_name: str, embedding: List[float], user_id: str) -> None:
        """Add or update embedding for a node.
        
        This also:
        1. Adds the User_{ID} label to the node (required for the index).
        2. Ensures the user's specific vector index exists.
        """
        if not self._validate_embedding(embedding):
            logger.error(f"Invalid embedding format for node {node_name}.")
            return
        
        # 1. Ensure index exists for this user
        await self._ensure_user_index(user_id)
        
        user_label = self._get_user_label(user_id)
        
        # 2. Add embedding AND proper user label
        query = f"""
        MATCH (n {{name: $node_name, UserId: $user_id}})
        SET n:{user_label}
        CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
        """
        async with self.driver.session() as session:
            await session.run(query, node_name=node_name, embedding=embedding, user_id=user_id)

    async def add_embeddings(self, rows: List[Dict[str, Any]], user_id: str) -> None:
        """Add or update embeddings for multiple nodes in a batch."""
        if not rows:
            return

        valid_rows = []
        for row in rows:
            embedding = row.get("embedding")
            node_name = row.get("node_name")
            if not node_name or not self._validate_embedding(embedding):
                logger.error(f"Invalid embedding format for node {node_name}.")
                continue
            valid_rows.append({"node_name": node_name, "embedding": embedding})

        if not valid_rows:
            return

        await self._ensure_user_index(user_id)
        user_label = self._get_user_label(user_id)

        query = f"""
        UNWIND $rows AS row
        MATCH (n {{name: row.node_name, UserId: $user_id}})
        SET n:{user_label}
        CALL db.create.setNodeVectorProperty(n, 'embedding', row.embedding)
        """
        async with self.driver.session() as session:
            await session.run(query, rows=valid_rows, user_id=user_id)
    
    async def search_similar(
        self, 
        embedding: List[float], 
        user_id: str, 
        limit: int = 5,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using User-Specific Index with optional filtering.
        
        This provides perfect isolation. We query ONLY the index for this user.
        No global competition ("crowding out") is possible.
        """
        index_name = self._get_index_name(user_id)
        
        # Build WHERE clause if filters exist
        where_clauses = []
        params = {
            "indexName": index_name,
            "embedding": embedding,
            "limit": limit
        }

        if filters:
            if "date_range" in filters:
                start, end = filters["date_range"]
                where_clauses.append("node.created_at >= $start_date")
                where_clauses.append("node.created_at <= $end_date")
                params["start_date"] = start.isoformat() if hasattr(start, 'isoformat') else start
                params["end_date"] = end.isoformat() if hasattr(end, 'isoformat') else end
            
            # Add other filters here as needed

        where_clause = " AND ".join(where_clauses)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        else:
            where_clause = ""
        
        # Note: If the user has no index yet (no data), this might fail or return empty.
        # We handle this gracefully.
        
        query = f"""
        CALL db.index.vector.queryNodes($indexName, $limit, $embedding)
        YIELD node, score
        {where_clause}
        RETURN node.name AS node_name, score
        ORDER BY score DESC
        """
        results = []
        async with self.driver.session() as session:
            # Check if index exists first to avoid error? 
            # Or just try/catch the procedure call.
            
            tx = await session.begin_transaction()
            try:
                result = await tx.run(query, **params)
                async for record in result:
                    results.append({
                        'node_name': record['node_name'],
                        'score': record['score']
                    })
                await tx.commit()
            except Exception as e:
                await tx.rollback()
                # If index doesn't exist, it means user has no vector data yet.
                if "Procedure call provided invalid name" in str(e) or "Index not found" in str(e):
                    logger.debug(f"Vector search for user {user_id} returned no results (index missing).")
                    return []
                raise e
            finally:
                await tx.close()
        return results
    
    @staticmethod
    def _validate_embedding(embedding: List[float]) -> bool:
        """Validate embedding format."""
        return isinstance(embedding, list) and all(isinstance(item, float) for item in embedding)
    
    async def drop_index(self, user_id: str = None) -> None:
        """Drop vector index.
        
        Args:
            user_id: If provided, drops ONLY that user's index.
                     If None, attempts to drop the legacy global index.
        """
        async with self.driver.session() as session:
            if user_id:
                index_name = self._get_index_name(user_id)
                await session.run(f"DROP INDEX {index_name} IF EXISTS")
                logger.info(f"Dropped vector index for user {user_id}")
            else:
                # Legacy global index drop
                await session.run("DROP INDEX embeddings_index IF EXISTS")
