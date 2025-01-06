# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from luna9.core.neo4j_database import Neo4jConnectionManager
from luna9.core.graph_ops import GraphOps
from luna9.core.migrations import ensure_seed_schemas

from app_server.routers.graph_api import router as graph_ops_router
from app_server.routers.chat_api import router as chat_api_router
from app_server.config import BaseConfig

config = BaseConfig()

app = FastAPI(
    title=config.INFO.title,
    description=config.INFO.description,
    version=config.INFO.version
)

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    graph_ops = await GraphOps().__aenter__()
    try:
        await ensure_seed_schemas(graph_ops)
    finally:
        await graph_ops.__aexit__(None, None, None)


app.include_router(graph_ops_router, prefix="/api/v1")
app.include_router(chat_api_router, prefix="/api/v1")