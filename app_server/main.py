# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from persona_graph.core.neo4j_database import Neo4jConnectionManager

from app_server.routers.graph_api import router as graph_ops_router
from app_server.config import BaseConfig

config = BaseConfig()

app = FastAPI(
    title=config.INFO.title,
    description=config.INFO.description,
    version=config.INFO.version
)

@asynccontextmanager
async def app_lifespan(app):
    neo4j_manager = Neo4jConnectionManager()
    try:
        await neo4j_manager.wait_for_neo4j()
        yield
    finally:
        await neo4j_manager.close()

app.lifespan = app_lifespan

app.include_router(graph_ops_router, prefix="/api/v1")