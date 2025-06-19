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
from server.logging_config import get_logger

logger = get_logger(__name__)


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
        logger.info(f"Creating user: {user_id}")
        result = await UserService.create_user(user_id, graph_ops)
        
        # Set appropriate status code based on whether user was created or already existed
        if result["status"] == "exists":
            response.status_code = 200  # OK - user already exists
            logger.debug(f"User {user_id} already exists")
        else:
            response.status_code = 201  # Created - new user
            logger.info(f"User {user_id} created successfully")
            
        return result
            
    except ValueError as e:
        logger.warning(f"Invalid user ID format: {user_id} - {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid user ID: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error occurred while creating user")

@router.delete("/users/{user_id}", status_code=200, description="Delete an existing user from the system")
async def delete_user(
    user_id: str = Path(..., description="The unique identifier for the user"),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        logger.info(f"Deleting user: {user_id}")
        
        # Check if user exists first
        if not await graph_ops.user_exists(user_id):
            logger.warning(f"Attempted to delete non-existent user: {user_id}")
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
            
        await UserService.delete_user(user_id, graph_ops)
        logger.info(f"User {user_id} deleted successfully")
        return {"message": f"User {user_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error occurred while deleting user")

@router.post("/users/{user_id}/ingest", status_code=201)
async def ingest_data(
    user_id: str = Path(..., description="The unique identifier for the user"),
    data: UnstructuredData = Body(...),
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        logger.info(f"Ingesting data for user: {user_id}")
        
        # Validate user exists
        if not await graph_ops.user_exists(user_id):
            logger.warning(f"Attempted to ingest data for non-existent user: {user_id}")
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
            
        # Validate data content
        if not data.content or len(data.content.strip()) == 0:
            logger.warning(f"Empty content provided for user {user_id}")
            raise HTTPException(status_code=400, detail="Content cannot be empty")
            
        await IngestService.ingest_data(user_id, data, graph_ops)
        logger.info(f"Data ingested successfully for user {user_id}")
        return {"message": "Data ingested successfully"}
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid data format for user {user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to ingest data for user {user_id}: {str(e)}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(status_code=503, detail="Database connection error. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while ingesting data")

@router.post("/users/{user_id}/rag/query", response_model=RAGResponse)
async def rag_query(
    user_id: str = Path(..., description="The unique identifier for the user"),
    query: RAGQuery = None,
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        if not query or not query.query:
            logger.warning(f"Empty query received for user {user_id}")
            raise HTTPException(status_code=400, detail="Query is required")
            
        # Validate user exists
        if not await graph_ops.user_exists(user_id):
            logger.warning(f"RAG query attempted for non-existent user: {user_id}")
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
            
        # Validate query length
        if len(query.query.strip()) > 1000:
            logger.warning(f"Query too long for user {user_id}: {len(query.query)} characters")
            raise HTTPException(status_code=400, detail="Query is too long (max 1000 characters)")
            
        logger.info(f"Processing RAG query for user {user_id}: {query.query[:100]}...")
        result = await RAGService.query(user_id, query.query, graph_ops)
        logger.info(f"RAG query completed successfully for user {user_id}")
        return RAGResponse(answer=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in RAG query for user {user_id}: {str(e)}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="Database connection error. Please ensure Neo4j is running and accessible."
            )
        if "openai" in str(e).lower() or "api" in str(e).lower():
            raise HTTPException(status_code=502, detail="External service error. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while processing query")

@router.post("/users/{user_id}/rag/query-vector", status_code=status.HTTP_200_OK)
async def rag_query_vector(
    user_id: str = Path(..., description="The unique identifier for the user"),
    query: RAGQuery = None,
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        if not query or not query.query:
            logger.warning(f"Empty vector query received for user {user_id}")
            raise HTTPException(status_code=400, detail="Query is required")
            
        # Validate user exists
        if not await graph_ops.user_exists(user_id):
            logger.warning(f"Vector RAG query attempted for non-existent user: {user_id}")
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
            
        logger.info(f"Processing vector RAG query for user {user_id}: {query.query[:100]}...")
        rag = RAGInterface(user_id)
        rag.graph_ops = graph_ops
        rag.graph_context_retriever = GraphContextRetriever(graph_ops)
        
        response = await rag.query_vector_only(query.query)
        logger.info(f"Vector RAG query completed successfully for user {user_id}")
        return {"query": query.query, "response": response}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during vector-only RAG query for user {user_id}: {e}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(status_code=503, detail="Database connection error. Please try again later.")
        if "openai" in str(e).lower() or "api" in str(e).lower():
            raise HTTPException(status_code=502, detail="External service error. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while processing vector query")

@router.post("/users/{user_id}/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_insights(
    user_id: str = Path(..., description="The unique identifier for the user"),
    ask_request: AskRequest = None,
    graph_ops: GraphOps = Depends(get_graph_ops)
):
    try:
        if not ask_request:
            logger.warning(f"Empty ask request received for user {user_id}")
            raise HTTPException(status_code=400, detail="Request body is required")
            
        if not ask_request.query or len(ask_request.query.strip()) == 0:
            logger.warning(f"Empty query in ask request for user {user_id}")
            raise HTTPException(status_code=400, detail="Query is required")
            
        # Validate user exists
        if not await graph_ops.user_exists(user_id):
            logger.warning(f"Ask insights attempted for non-existent user: {user_id}")
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
            
        logger.info(f"Processing ask insights for user {user_id}: {ask_request.query[:100]}...")
        response = await AskService.ask_insights(user_id, ask_request, graph_ops)
        logger.info(f"Ask insights completed successfully for user {user_id}")
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid ask request format for user {user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")
    except Exception as e:
        logger.error(f"Error in ask insights for user {user_id}: {str(e)}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(status_code=503, detail="Database connection error. Please try again later.")
        if "openai" in str(e).lower() or "api" in str(e).lower():
            raise HTTPException(status_code=502, detail="External service error. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while processing insights request")

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
            logger.warning(f"Empty custom data update received for user {user_id}")
            raise HTTPException(status_code=400, detail="Request body is required")
            
        # Validate user exists
        if not await graph_ops.user_exists(user_id):
            logger.warning(f"Custom data update attempted for non-existent user: {user_id}")
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
            
        # Validate update content
        if not update.nodes and not update.relationships:
            logger.warning(f"Empty custom data update for user {user_id}")
            raise HTTPException(status_code=400, detail="At least one node or relationship must be provided")
            
        logger.info(f"Processing custom data update for user {user_id}: {len(update.nodes or [])} nodes, {len(update.relationships or [])} relationships")
        custom_service = CustomDataService(graph_ops)
        result = await custom_service.update_custom_data(user_id, update)
        logger.info(f"Custom data update completed successfully for user {user_id}")
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid custom data format for user {user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    except Exception as e:
        logger.error(f"Error in custom data update for user {user_id}: {str(e)}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(status_code=503, detail="Database connection error. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while updating custom data")