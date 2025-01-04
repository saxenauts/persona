# app/db.py
from neo4j import AsyncGraphDatabase, basic_auth, Neo4jDriver
from contextlib import asynccontextmanager

class GraphDatabaseConfig:
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "password"

driver: Neo4jDriver = AsyncGraphDatabase.driver(
    GraphDatabaseConfig.NEO4J_URI,
    auth=basic_auth(GraphDatabaseConfig.NEO4J_USERNAME, GraphDatabaseConfig.NEO4J_PASSWORD)
)

@asynccontextmanager
async def get_graph_db():
    async with driver.session() as session:
        yield session
