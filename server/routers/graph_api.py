from fastapi import APIRouter, HTTPException, status, Path, Depends, Body, Response
from persona.core.graph_ops import GraphOps
from persona.core.rag_interface import RAGInterface
from persona.models.schema import UserCreate, RAGQuery, RAGResponse
from persona.models.schema import AskRequest, AskResponse
from persona.services.user_service import UserService
from persona.services.rag_service import RAGService
from persona.services.ask_service import AskService
from persona.adapters import PersonaAdapter
from server.dependencies import get_graph_ops
from server.logging_config import get_logger
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import os
import re


logger = get_logger(__name__)


# --- Request Models (replacing legacy UnstructuredData) ---
class IngestRequest(BaseModel):
    """Request body for ingesting content."""

    content: str = Field(..., description="Raw text content to ingest.")
    source_type: str = Field(
        default="conversation",
        description="Type of content (conversation, notes, etc.)",
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None, description="Optional metadata."
    )


class IngestBatchRequest(BaseModel):
    """Request body for batch ingestion."""

    items: List[IngestRequest] = Field(..., description="List of items to ingest.")


router = APIRouter()

# Regex for validating user IDs. Allows alphanumeric chars, hyphens, and underscores.
# This provides a basic level of sanitization to prevent injection or invalid characters.
USER_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]+$")


def is_valid_user_id(user_id: str) -> bool:
    """Check if the user ID matches the allowed pattern."""
    return bool(USER_ID_REGEX.match(user_id))


@router.get("/version")
def get_version():
    return {"version": "1.0.0"}


@router.post("/users/{user_id}")
async def create_user(
    user_id: str = Path(..., description="The unique identifier for the user"),
    graph_ops: GraphOps = Depends(get_graph_ops),
    response: Response = None,
):
    try:
        if not is_valid_user_id(user_id):
            raise ValueError("Invalid user ID format.")

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
        raise HTTPException(status_code=422, detail=f"Invalid user ID format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error occurred while creating user"
        )


@router.delete(
    "/users/{user_id}",
    status_code=200,
    description="Delete an existing user from the system",
)
async def delete_user(
    user_id: str = Path(..., description="The unique identifier for the user"),
    graph_ops: GraphOps = Depends(get_graph_ops),
):
    try:
        if not is_valid_user_id(user_id):
            raise ValueError("Invalid user ID format.")

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
    except ValueError as e:
        logger.warning(f"Invalid user ID provided for deletion: {user_id} - {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid user ID format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error occurred while deleting user"
        )


@router.post("/users/{user_id}/ingest", status_code=201)
async def ingest_data(
    user_id: str = Path(..., description="The unique identifier for the user"),
    data: IngestRequest = Body(...),
    graph_ops: GraphOps = Depends(get_graph_ops),
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

        # Use PersonaAdapter for ingestion
        adapter = PersonaAdapter(user_id, graph_ops)
        result = await adapter.ingest(
            content=data.content, source_type=data.source_type
        )

        if not result.success:
            raise HTTPException(
                status_code=500, detail=f"Ingestion failed: {result.error}"
            )

        type_counts = {}
        for memory in result.memories:
            mem_type = getattr(memory, "type", "unknown")
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

        links_created = len(result.links)
        logger.info(
            f"Data ingested successfully for user {user_id}: {len(result.memories)} memories"
        )
        return {
            "message": "Data ingested successfully",
            "memories_created": len(result.memories),
            "memories_created_by_type": type_counts,
            "links_created": links_created,
            "timings_ms": {
                "extract": result.extract_time_ms or 0.0,
                "embed": result.embed_time_ms or 0.0,
                "persist": result.persist_time_ms or 0.0,
                "total": result.total_time_ms or 0.0,
            },
        }

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid data format for user {user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to ingest data for user {user_id}: {str(e)}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="Database connection error. Please try again later.",
            )
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while ingesting data",
        )


@router.post("/users/{user_id}/ingest/batch", status_code=201)
async def ingest_batch_data(
    user_id: str = Path(..., description="The unique identifier for the user"),
    batch_data: IngestBatchRequest = Body(...),
    graph_ops: GraphOps = Depends(get_graph_ops),
):
    try:
        logger.info(
            f"Ingesting batch of {len(batch_data.items)} items for user: {user_id}"
        )

        # Validate user exists
        if not await graph_ops.user_exists(user_id):
            logger.warning(
                f"Attempted to batch ingest for non-existent user: {user_id}"
            )
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        if not batch_data.items:
            raise HTTPException(status_code=400, detail="Batch cannot be empty")

        # Use PersonaAdapter for batch ingestion
        adapter = PersonaAdapter(user_id, graph_ops)
        items_for_adapter = [
            {"content": item.content, "source_type": item.source_type}
            for item in batch_data.items
        ]
        results = await adapter.ingest_batch(items_for_adapter)

        total_memories = 0
        total_links = 0
        type_counts: Dict[str, int] = {}
        timing_totals = {"extract": 0.0, "embed": 0.0, "persist": 0.0, "total": 0.0}
        for r in results:
            if not r.success:
                continue
            total_memories += len(r.memories)
            total_links += len(r.links)
            timing_totals["extract"] += r.extract_time_ms or 0.0
            timing_totals["embed"] += r.embed_time_ms or 0.0
            timing_totals["persist"] += r.persist_time_ms or 0.0
            timing_totals["total"] += r.total_time_ms or 0.0
            for memory in r.memories:
                mem_type = getattr(memory, "type", "unknown")
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

        logger.info(
            f"Batch ingestion completed for user {user_id}: {total_memories} memories"
        )
        return {
            "message": f"Successfully ingested batch of {len(batch_data.items)} items",
            "memories_created": total_memories,
            "memories_created_by_type": type_counts,
            "links_created": total_links,
            "timings_ms": timing_totals,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to batch ingest for user {user_id}: {str(e)}")
        if "Neo4j" in str(e):
            raise HTTPException(status_code=503, detail="Database connection error.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/users/{user_id}/rag/query", response_model=RAGResponse)
async def rag_query(
    user_id: str = Path(..., description="The unique identifier for the user"),
    query: RAGQuery = None,
    graph_ops: GraphOps = Depends(get_graph_ops),
):
    try:
        if not query or not query.query:
            logger.warning(f"Empty query received for user {user_id}")
            raise HTTPException(status_code=400, detail="Query is required")

        # Validate user exists
        if not await graph_ops.user_exists(user_id):
            logger.warning(f"RAG query attempted for non-existent user: {user_id}")
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        # Validate query length (configurable)
        max_query_chars = int(os.getenv("RAG_QUERY_MAX_CHARS", "0"))
        if max_query_chars > 0 and len(query.query.strip()) > max_query_chars:
            logger.warning(
                f"Query too long for user {user_id}: {len(query.query)} characters"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Query is too long (max {max_query_chars} characters)",
            )

        logger.info(f"Processing RAG query for user {user_id}: {query.query[:100]}...")
        result = await RAGService.query(
            user_id,
            query.query,
            retrieval_query=query.retrieval_query,
            include_stats=query.include_stats,
        )
        logger.info(f"RAG query completed successfully for user {user_id}")
        if isinstance(result, dict):
            return RAGResponse(**result)
        return RAGResponse(answer=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in RAG query for user {user_id}: {str(e)}")
        if "Neo4j" in str(e) or "database" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="Database connection error. Please ensure Neo4j is running and accessible.",
            )
        if "openai" in str(e).lower() or "api" in str(e).lower():
            raise HTTPException(
                status_code=502,
                detail="External service error. Please try again later.",
            )
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing query",
        )


@router.post(
    "/users/{user_id}/ask", response_model=AskResponse, status_code=status.HTTP_200_OK
)
async def ask_insights(
    user_id: str = Path(..., description="The unique identifier for the user"),
    ask_request: AskRequest = None,
    graph_ops: GraphOps = Depends(get_graph_ops),
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

        logger.info(
            f"Processing ask insights for user {user_id}: {ask_request.query[:100]}..."
        )
        response = await AskService.ask_insights(user_id, ask_request)
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
            raise HTTPException(
                status_code=503,
                detail="Database connection error. Please try again later.",
            )
        if "openai" in str(e).lower() or "api" in str(e).lower():
            raise HTTPException(
                status_code=502,
                detail="External service error. Please try again later.",
            )
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing insights request",
        )
