from fastapi import APIRouter, HTTPException, status, Path, Depends, Body, Response
from persona.core.graph_ops import GraphOps, GraphContextRetriever
from persona.models.schema import NodeModel, RelationshipModel, GraphUpdateModel
from persona.core.constructor import GraphConstructor
from persona.llm.prompts import sample_statements, ASTRONAUT_PROMPT, SPACE_SCHOOL_CHAT
from persona.models.schema import UnstructuredData
from persona.core.constructor import GraphContextRetriever
from persona.core.rag_interface import RAGInterface
from persona.models.schema import UserCreate, RAGQuery, RAGResponse
from persona.services.user_service import UserService
from persona.services.ingest_service import IngestService
from persona.services.rag_service import RAGService
from persona.services.ask_service import AskService
from persona.services.custom_data_service import CustomDataService
from persona.models.schema import LearnRequest, LearnResponse, AskRequest, AskResponse, GraphSchema, CustomGraphUpdate, CustomNodeData, CustomRelationshipData
from server.dependencies import get_graph_ops


router = APIRouter()

@router.get("/version")
def get_version():
    return {"version": "1.0.0"}

@router.post("/users/{user_id}")
async def create_user(
    user_id: str = Path(..., description="The unique identifier for the user"),
    graph_ops: GraphOps = Depends(get_graph_ops),
    response: Response = None
):
    try:
        result = await UserService.create_user(user_id, graph_ops)
        
        # Set appropriate status code based on whether user was created or already existed
        if result["status"] == "exists":
            response.status_code = 200  # OK - user already exists
        else:
            response.status_code = 201  # Created - new user
            
        return result
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/users/{user_id}", status_code=200, description="Delete an existing user from the system")
async def delete_user(
    user_id: str = Path(..., description="The unique identifier for the user"),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        # Check if user exists first
        if not await graph_ops.user_exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
            
        await UserService.delete_user(user_id, graph_ops)
        return {"message": f"User {user_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users/{user_id}/ingest", status_code=201)
async def ingest_data(
    user_id: str = Path(..., description="The unique identifier for the user"),
    data: UnstructuredData = Body(...),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        await IngestService.ingest_data(user_id, data, graph_ops)
        return {"message": "Data ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/users/{user_id}/rag/query", response_model=RAGResponse)
async def rag_query(
    user_id: str = Path(..., description="The unique identifier for the user"),
    query: RAGQuery = None,
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        if not query or not query.query:
            raise HTTPException(status_code=400, detail="Query is required")
            
        print(f"Processing RAG query for user {user_id}: {query.query}")
        result = await RAGService.query(user_id, query.query, graph_ops)
        return RAGResponse(answer=result)
    except Exception as e:
        print(f"Error in RAG query: {str(e)}")
        if "Neo4j" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Database connection error. Please ensure Neo4j is running and accessible."
            )
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/users/{user_id}/rag/query-vector", status_code=status.HTTP_200_OK)
async def rag_query_vector(
    user_id: str = Path(..., description="The unique identifier for the user"),
    query: RAGQuery = None,
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        if not query or not query.query:
            raise HTTPException(status_code=400, detail="Query is required")
            
        rag = RAGInterface(user_id)
        rag.graph_ops = graph_ops
        rag.graph_context_retriever = GraphContextRetriever(graph_ops)
        
        response = await rag.query_vector_only(query.query)
        return {"query": query.query, "response": response}
    except Exception as e:
        print(f"Error during vector-only RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users/{user_id}/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_insights(
    user_id: str = Path(..., description="The unique identifier for the user"),
    ask_request: AskRequest = None,
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        if not ask_request:
            raise HTTPException(status_code=400, detail="Request body is required")
        
        response = await AskService.ask_insights(user_id, ask_request, graph_ops)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users/{user_id}/custom-data", status_code=status.HTTP_200_OK)
async def update_custom_data(
    user_id: str = Path(..., description="The unique identifier for the user"),
    update: CustomGraphUpdate = None,
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    """
    Update or create custom structured data in the graph
    """
    try:
        if not update:
            raise HTTPException(status_code=400, detail="Request body is required")
        
        custom_service = CustomDataService(graph_ops)
        result = await custom_service.update_custom_data(user_id, update)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))