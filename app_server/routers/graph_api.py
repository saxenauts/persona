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
from persona.services.learn_service import LearnService
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
            graph_schema=test_schema,
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


@router.post("/test-custom-data-flow", status_code=status.HTTP_200_OK)
async def test_custom_data_flow():
    """
    Test the custom data flow with gaming preferences
    """
    graph_ops = None
    try:
        # Initialize GraphOps
        graph_ops = await GraphOps().__aenter__()
        custom_service = CustomDataService(graph_ops)

        # Create test data
        test_update = CustomGraphUpdate(
            nodes=[
                CustomNodeData(
                    name="current_gaming_preference",
                    properties={
                        "favorite_genre": "RPG",
                        "current_game": "Baldur's Gate 3",
                        "hours_played": 120,
                        "last_played": "2024-03-15T14:30:00Z"
                    }
                )
            ],
            relationships=[
                RelationshipModel(
                    source="current_gaming_preference",
                    target="test_user",
                    relation="PREFERENCE_OF"
                )
            ]
        )

        # Test the custom data service
        result = await custom_service.update_custom_data("test_user", test_update)
        
        # Verify data was stored by retrieving the node
        node_data = await graph_ops.get_node_data("current_gaming_preference", "test_user")
        
        if not node_data:
            raise ValueError("Custom data was not stored successfully")

        return {
            "status": "Success",
            "update_result": result,
            "stored_data": node_data
        }

    except Exception as e:
        print(f"Error during custom data test flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if graph_ops:
            print("Closing graph ops...")
            await graph_ops.__aexit__(None, None, None)