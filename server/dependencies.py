from fastapi import Depends, Request
from persona.core.graph_ops import GraphOps
from typing import Annotated

def get_graph_ops(request: Request) -> GraphOps:
    """Dependency to get the global GraphOps instance from app.state"""
    return request.app.state.graph_ops

# Type alias for easier usage in service methods
GraphOpsDep = Annotated[GraphOps, Depends(get_graph_ops)] 