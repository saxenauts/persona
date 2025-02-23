# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from persona.core.neo4j_database import Neo4jConnectionManager
from persona.core.graph_ops import GraphOps
from fastapi.middleware.cors import CORSMiddleware
from server.routers.graph_api import router as graph_api_router
import asyncio

from server.config import BaseConfig

config = BaseConfig()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Neo4j connection
    neo4j_manager = Neo4jConnectionManager()
    await neo4j_manager.initialize()
    
    # Initialize GraphOps with the connected manager
    app.state.graph_ops = GraphOps(neo4j_manager)
    
    yield
    
    # Cleanup
    if hasattr(app.state, "graph_ops"):
        await app.state.graph_ops.close()

app = FastAPI(
    title=config.INFO.title,
    description=config.INFO.description,
    version=config.INFO.version,
    lifespan=lifespan
)

app.include_router(graph_api_router, prefix="/api/v1")