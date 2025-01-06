import os
import pytest
import asyncio
from fastapi.testclient import TestClient
from app_server.main import app
from luna9.core.neo4j_database import Neo4jConnectionManager
from app_server.config import config

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_client():
    return TestClient(app)

@pytest.fixture(scope="session")
async def neo4j_manager():
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    
    manager = Neo4jConnectionManager(
        uri=neo4j_uri,
        user=config.NEO4J.USER,
        password=config.NEO4J.PASSWORD
    )
    
    # Test connection and wait for Neo4j
    try:
        await manager.wait_for_neo4j()
    except Exception as e:
        pytest.fail(f"Neo4j connection failed: {str(e)}")
    
    yield manager
    await manager.close()

@pytest.fixture(scope="function")
async def test_user(neo4j_manager):
    user_id = "test_user"
    # Clean up any existing test user
    await neo4j_manager.run_query("MATCH (u:User {id: $user_id}) DETACH DELETE u", {"user_id": user_id})
    yield user_id
    # Cleanup after test
    await neo4j_manager.run_query("MATCH (u:User {id: $user_id}) DETACH DELETE u", {"user_id": user_id})