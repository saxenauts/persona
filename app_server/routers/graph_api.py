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
from persona_graph.services.learn_service import LearnService
from persona_graph.services.ask_service import AskService
from persona_graph.services.byoa import PersonalizeServiceBYOA
from persona_graph.models.schema import LearnRequest, LearnResponse, AskRequest, AskResponse, PersonalizeRequest, PersonalizeResponse, GraphSchema


router = APIRouter()

@router.get("/version")
def get_version():
    return {"version": "1.0.0"}

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

@router.post("/learn", response_model=LearnResponse, status_code=status.HTTP_200_OK)
async def learn_user(learn_request: LearnRequest):
    try:
        response = await LearnService.learn_user(learn_request)
        return response
    except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_insights(ask_request: AskRequest):
    try:
        response = await AskService.ask_insights(ask_request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/byoa", response_model=PersonalizeResponse, status_code=status.HTTP_200_OK)
async def personalize_user(personalize_request: PersonalizeRequest):
    try:
        response = await PersonalizeServiceBYOA.personalize_user(personalize_request)
        return response
    except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))


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


@router.post("/test-learn-flow", status_code=status.HTTP_200_OK)
async def test_learn_flow():
    graph_ops = None
    try:
        # Initialize GraphOps
        graph_ops = await GraphOps().__aenter__()
        learn_service = LearnService(graph_ops)

        # Create a test schema
        test_schema = GraphSchema(
            name="Food Preferences",
            description="Learn about user's food preferences and eating patterns",
            attributes=[
                'FAVORITE_CUISINE',
                'DIETARY_RESTRICTION',
                'MEAL_TIMING',
                'FLAVOR_PREFERENCE'
            ],
            relationships=[
                'PAIRS_WELL_WITH',
                'AVOIDS_WITH',
                'PREFERS_BEFORE',
                'PREFERS_AFTER'
            ]
        )

        # Create learn request
        learn_request = LearnRequest(
            user_id="test_user",
            schema=test_schema,
            description="Understanding user's food preferences and eating patterns"
        )

        # Test the learn service
        response = await learn_service.learn_user(learn_request)
        
        # Verify schema was stored by trying to retrieve it
        all_schemas = await graph_ops.get_all_schemas()
        stored_schema = next((s for s in all_schemas if s.name == test_schema.name), None)

        if not stored_schema:
            raise ValueError("Schema was not stored successfully")

        return {
            "status": "Success",
            "schema_id": response.schema_id,
            "stored_schema": stored_schema.model_dump()
        }

    except Exception as e:
        print(f"Error during learn test flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if graph_ops:
            print("Closing graph ops...")
            await graph_ops.__aexit__(None, None, None)



@router.post("/test-ask-flow", status_code=status.HTTP_200_OK)
async def test_ask_flow():
    graph_ops = None
    try:
        # Initialize GraphOps
        graph_ops = await GraphOps().__aenter__()
        ask_service = AskService(graph_ops)

        # Create a test ask request for food preferences
        test_request = AskRequest(
            user_id="test_user",
            query="What are this user's top 3 favorite topics in space?",
            output_schema={
                "favorite_topics": [
                    {
                        "topic": "Boosters",  # example
                        "evidence": ["Talks about boosters a lot", "Watches SpaceX launches"]  # example
                    }
                ],
                "analysis": {
                    "primary_topic": "Boosters",  # example
                    "frequency": "3 times per week",  # example
                    "evidence": ["Talks about boosters a lot", "Watches SpaceX launches"]  # example
                }
            }
        )

        # Test the ask service
        response = await ask_service.ask_insights(test_request)
        
        return {
            "status": "Success",
            "query": test_request.query,
            "result": response.result
        }

    except Exception as e:
        print(f"Error during ask test flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if graph_ops:
            print("Closing graph ops...")
            await graph_ops.__aexit__(None, None, None)