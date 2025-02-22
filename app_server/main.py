# main.py
from fastapi import FastAPI
from persona.core.neo4j_database import Neo4jConnectionManager
from persona.core.graph_ops import GraphOps
from fastapi.middleware.cors import CORSMiddleware
from app_server.routers.graph_api import router as graph_api_router
import asyncio

from app_server.config import BaseConfig

config = BaseConfig()

app = FastAPI(
    title=config.INFO.title,
    description=config.INFO.description,
    version=config.INFO.version
)

@app.on_event("startup")
async def startup_event():
    # Initialize Neo4j connection
    neo4j_manager = Neo4jConnectionManager()
    await neo4j_manager.initialize()
    
    # Initialize GraphOps with the connected manager
    app.state.graph_ops = GraphOps(neo4j_manager)

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "graph_ops"):
        await app.state.graph_ops.close()

app.include_router(graph_api_router, prefix="/api/v1")