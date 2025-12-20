# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from persona.core import GraphOps
from fastapi.middleware.cors import CORSMiddleware
from server.routers.graph_api import router as graph_api_router
from server.logging_config import setup_logging, get_logger
import asyncio

from server.config import BaseConfig

config = BaseConfig()

# Initialize logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph_ops = GraphOps()
    await app.state.graph_ops.initialize()
    yield
    if hasattr(app.state, "graph_ops"):
        await app.state.graph_ops.close()

app = FastAPI(
    title=config.INFO.title,
    description=config.INFO.description,
    version=config.INFO.version,
    lifespan=lifespan
)

app.include_router(graph_api_router, prefix="/api/v1")