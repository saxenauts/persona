from fastapi import APIRouter, HTTPException, status
from persona.core.graph_ops import GraphOps
from persona.models.schema import NodeModel, RelationshipModel, GraphUpdateModel
from persona.core.constructor import GraphConstructor
from persona.llm.prompts import sample_statements, ASTRONAUT_PROMPT, SPACE_SCHOOL_CHAT
from persona.models.schema import UnstructuredData
from persona.core.constructor import GraphContextRetriever
from persona.core.rag_interface import RAGInterface
from persona.models.schema import UserCreate, IngestData, RAGQuery, RAGResponse
from persona.services.user_service import UserService
from persona.services.ingest_service import IngestService
from persona.services.rag_service import RAGService
from persona.services.ask_service import AskService
from persona.services.custom_data_service import CustomDataService
from persona.models.schema import LearnRequest, LearnResponse, AskRequest, AskResponse, GraphSchema, CustomGraphUpdate, CustomNodeData, CustomRelationshipData


router = APIRouter()

@router.get("/version")
def get_version():
    return {"version": "1.0.0"}

@router.post("/user/create", status_code=201, description="Create a new user in the system")
async def create_user(user: UserCreate):
    try:
        await UserService.create_user(user.user_id)
        return {"message": f"User {user.user_id} created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/user/delete", status_code=200, description="Delete an existing user from the system")
async def delete_user(user: UserCreate):
    try:
        await UserService.delete_user(user.user_id)
        return {"message": f"User {user.user_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/ingest", status_code=201)
async def ingest_data(data: IngestData):
    try:
        await IngestService.ingest_data(data.user_id, data.content)
        return {"message": "Data ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/rag/query", response_model=RAGResponse)
async def rag_query(query: RAGQuery):
    try:
        print(f"Processing RAG query for user {query.user_id}: {query.query}")
        result = await RAGService.query(query.user_id, query.query)
        return RAGResponse(answer=result)
    except Exception as e:
        print(f"Error in RAG query: {str(e)}")
        if "Neo4j" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Database connection error. Please ensure Neo4j is running and accessible."
            )
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/rag-query-vector", status_code=status.HTTP_200_OK)
async def rag_query_vector(query: RAGQuery):
    try:
        async with RAGInterface(query.user_id) as rag:
            response = await rag.query_vector_only(query.query)
            return {"query": query.query, "response": response}
    except Exception as e:
        print(f"Error during vector-only RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_insights(ask_request: AskRequest):
    try:
        response = await AskService.ask_insights(ask_request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/custom-data", status_code=status.HTTP_200_OK)
async def update_custom_data(update: CustomGraphUpdate):
    """
    Update or create custom structured data in the graph
    """
    try:
        graph_ops = await GraphOps().__aenter__()
        custom_service = CustomDataService(graph_ops)
        result = await custom_service.update_custom_data(update.user_id, update)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await graph_ops.__aexit__(None, None, None)