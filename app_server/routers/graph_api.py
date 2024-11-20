from fastapi import APIRouter, HTTPException, status
from persona_graph.core.graph_ops import GraphOps
from persona_graph.models.schema import NodeModel, RelationshipModel, GraphUpdateModel
from persona_graph.core.constructor import GraphConstructor
from persona_graph.llm.prompts import sample_statements, ASTRONAUT_PROMPT, SPACE_SCHOOL_CHAT
from persona_graph.models.schema import UnstructuredData
from persona_graph.core.constructor import GraphContextRetriever
from persona_graph.core.rag_interface import RAGInterface
from persona_graph.models.schema import UserCreate, IngestData, RAGQuery, RAGResponse
from persona_graph.services.user_service import UserService
from persona_graph.services.ingest_service import IngestService
from persona_graph.services.rag_service import RAGService
import random

router = APIRouter()


@router.post("/users", status_code=201)
async def create_user(user: UserCreate):
    try:
        await UserService.create_user(user.user_id)
        return {"message": f"User {user.user_id} created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/users/{user_id}")
async def delete_user(user_id: str):
    try:
        await UserService.delete_user(user_id)
        return {"message": f"User {user_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/ingest/{user_id}")
async def ingest_data(user_id: str, data: IngestData):
    try:
        await IngestService.ingest_data(user_id, data.content)
        return {"message": "Data ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/rag/{user_id}/query", response_model=RAGResponse)
async def rag_query(user_id: str, query: RAGQuery):
    try:
        result = await RAGService.query(user_id, query.query)
        return RAGResponse(answer=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/version")
def get_version():
    return {"version": "1.0.0"}  # Replace with your actual version number


@router.post("/rag-query", status_code=status.HTTP_200_OK)
async def rag_query(query: str, user_id: str):
    try:
        rag = RAGInterface(user_id)
        response = await rag.query(query)
        return {"query": query, "response": response}
    except Exception as e:
        print(f"Error during RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await rag.close()

@router.post("/rag-query-vector", status_code=status.HTTP_200_OK)
async def rag_query_vector(query: str, user_id: str):
    try:
        rag = RAGInterface(user_id)
        response = await rag.query_vector_only(query)
        return {"query": query, "response": response}
    except Exception as e:
        print(f"Error during vector-only RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await rag.close()


# Test the constructor flow

@router.post("/test-constructor-flow", status_code=status.HTTP_200_OK)
async def test_constructor_flow():
    graph_constructor = None
    try:
        user_id = "test_user"
        graph_constructor = await GraphConstructor(user_id=user_id).__aenter__()
            
        # Clean up the graph first
        print("Cleaning graph for user:", user_id)
        await graph_constructor.clean_graph()
        

         # First ensure user exists
        try:
            await UserService.create_user(user_id)
        except Exception as e:
            if "already exists" not in str(e):
                raise e
            

        # Ensure vector index exists (only once)
        try:
            await graph_constructor.graph_ops.neo4j_manager.ensure_vector_index()
        except Exception as index_error:
            if "EquivalentSchemaRuleAlreadyExists" not in str(index_error):
                raise index_error
            print("Vector index already exists, continuing...")

        print(f"Processing text")
        data = UnstructuredData(title="Sample Statement", content=SPACE_SCHOOL_CHAT)
        
        # Process the unstructured data
        await graph_constructor.ingest_unstructured_data_to_graph(data)
        
        # Retrieve and print the updated graph context
        print("Retrieving updated graph context...")
        context = await graph_constructor.graph_context_retriever.get_rich_context(query="Technology", user_id=user_id)
        print("Updated graph context:", context)

        return {"status": "Graph updated successfully", "context": context}

    except Exception as e:
        print(f"Error during constructor test flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if graph_constructor:
            print("Closing graph constructor...")
            await graph_constructor.__aexit__(None, None, None)