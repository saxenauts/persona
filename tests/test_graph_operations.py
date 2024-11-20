import pytest
from persona_graph.core.graph_ops import GraphOps

@pytest.mark.asyncio
async def test_graph_operations(neo4j_manager, test_user):
    graph_ops = GraphOps(neo4j_manager)
    
    # Test node creation
    await graph_ops.add_node_with_embedding(
        "Python",
        test_user,
        {"description": "Programming language"}
    )
    
    # Test node retrieval
    node = await graph_ops.get_node("Python", test_user)
    assert node is not None
    assert node["name"] == "Python"
    
    # Test relationship creation
    await graph_ops.add_node_with_embedding(
        "FastAPI",
        test_user,
        {"description": "Web framework"}
    )
    await graph_ops.add_relationship(
        "Python",
        "FastAPI",
        "HAS_FRAMEWORK",
        test_user
    )
    
    # Test relationship retrieval
    rel = await graph_ops.get_relationship(
        "Python",
        "FastAPI",
        "HAS_FRAMEWORK",
        test_user
    )
    assert rel is not None