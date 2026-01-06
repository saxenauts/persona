from fastapi import APIRouter, HTTPException, status, Path, Depends, Body, Response
from persona.core.graph_ops import GraphOps
from persona.models.schema import UserCreate, AskRequest, AskResponse
from persona.services.user_service import UserService
from persona.services.persona_service import PersonaService
from persona.services.integration_agent import run_integration_agent
from persona.adapters import PersonaAdapter
from persona.utils.session import get_session_id
from server.dependencies import get_graph_ops
from server.logging_config import get_logger
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import os
import re


logger = get_logger(__name__)


# --- Request Models (replacing legacy UnstructuredData) ---
class IngestRequest(BaseModel):
    content: str = Field(..., description="Raw text content to ingest.")
    source_type: str = Field(default="conversation")
    provider_session_id: Optional[str] = Field(default=None)
    store_transcript: bool = Field(default=False)
    finalize_session: bool = Field(
        default=False,
        description="If true, run integration after ingest (use for last message in session).",
    )
    metadata: Optional[Dict[str, str]] = Field(default=None)


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


class RateLimiterStats(BaseModel):
    name: str
    bucket_tokens: int
    bucket_requests: float
    total_requests: int
    total_tokens: int
    wait_time_ms: int
    retries_429: int


class StatsResponse(BaseModel):
    rate_limiters: List[RateLimiterStats]
    uptime_seconds: float


import time as _time

_server_start_time = _time.time()


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get real-time rate limiter statistics for monitoring bandwidth/quota utilization."""
    from persona.llm.rate_limiter import get_rate_limiter_registry

    registry = get_rate_limiter_registry()
    stats = registry.get_all_stats()

    return StatsResponse(
        rate_limiters=[RateLimiterStats(**s) for s in stats],
        uptime_seconds=_time.time() - _server_start_time,
    )


@router.post("/users/{user_id}")
async def create_user(
    response: Response,
    user_id: str = Path(..., description="The unique identifier for the user"),
    graph_ops: GraphOps = Depends(get_graph_ops),
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

        # Generate canonical session_id
        session_id = get_session_id(data.source_type, data.provider_session_id)

        adapter = PersonaAdapter(user_id, graph_ops)
        result = await adapter.ingest(
            content=data.content,
            source_type=data.source_type,
            session_id=session_id,
            store_transcript=data.store_transcript,
            finalize_session=data.finalize_session,
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


class ChatMessageInput(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessageInput] = Field(..., description="Conversation history")
    user_timezone: str = Field(default="UTC")
    session_id: Optional[str] = Field(
        default=None, description="Session ID for continuity"
    )
    max_turns: Optional[int] = Field(default=None, description="Max agent turns")
    timeout: Optional[float] = Field(default=None, description="Timeout in seconds")
    include_stats: bool = Field(default=False, description="Include execution stats")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response")
    status: str = Field(..., description="Execution status")
    session_id: str = Field(..., description="Session ID for continuity")
    stats: Optional[Dict[str, Any]] = Field(default=None, description="Execution stats")
    state: Optional[str] = Field(default=None, description="Resumable state")


@router.post(
    "/users/{user_id}/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def chat(
    user_id: str = Path(..., description="The unique identifier for the user"),
    request: ChatRequest = Body(...),
    graph_ops: GraphOps = Depends(get_graph_ops),
):
    try:
        if not await graph_ops.user_exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")

        session_id = get_session_id("persona", request.session_id)

        last_user_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )

        if not last_user_msg:
            raise HTTPException(status_code=400, detail="No user message found")

        logger.info(f"Chat for user {user_id}: {last_user_msg[:100]}...")

        service = PersonaService(graph_ops)
        result = await service.run_agent(
            user_id=user_id,
            query=last_user_msg,
            include_stats=request.include_stats,
            user_timezone=request.user_timezone,
            session_id=session_id,
            max_turns=request.max_turns,
            timeout=request.timeout,
        )

        logger.info(f"Chat completed for user {user_id}")
        return ChatResponse(
            response=result["answer"],
            status=result["status"],
            session_id=session_id,
            stats=result.get("stats"),
            state=result.get("state"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing chat",
        )


@router.post(
    "/users/{user_id}/persona/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
)
async def persona_ask(
    user_id: str = Path(..., description="The unique identifier for the user"),
    request: AskRequest = Body(...),
    graph_ops: GraphOps = Depends(get_graph_ops),
):
    try:
        if not await graph_ops.user_exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        logger.info(f"Persona ask for user {user_id}: {request.query[:100]}...")

        service = PersonaService(graph_ops)
        result = await service.ask(
            user_id=user_id,
            query=request.query,
            output_schema=request.output_schema,
        )

        logger.info(f"Persona ask completed for user {user_id}")
        return AskResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in persona ask for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing ask request",
        )


class IntegrateRequest(BaseModel):
    memory_ids: Optional[List[str]] = Field(default=None)


class IntegrateResponse(BaseModel):
    status: str
    links_created: int = 0
    merges_applied: int = 0
    derived_created: int = 0
    conflicts_flagged: int = 0
    errors: List[str] = Field(default_factory=list)


@router.post(
    "/users/{user_id}/integrate",
    response_model=IntegrateResponse,
    status_code=status.HTTP_200_OK,
)
async def trigger_integration(
    user_id: str = Path(..., description="The unique identifier for the user"),
    request: Optional[IntegrateRequest] = Body(default=None),
    graph_ops: GraphOps = Depends(get_graph_ops),
):
    try:
        if not await graph_ops.user_exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        trigger_ids = request.memory_ids if request and request.memory_ids else []

        logger.info(f"Triggering integration for user {user_id}")
        result = await run_integration_agent(
            user_id=user_id,
            trigger_ids=trigger_ids,
            graph_ops=graph_ops,
            session_id=None,  # Full user scope
        )

        return IntegrateResponse(
            status="success" if result.success else "failed",
            links_created=result.links_created,
            merges_applied=result.merges_performed,
            derived_created=0,
            conflicts_flagged=result.flags_raised,
            errors=result.errors,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in integration for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing integration",
        )


@router.post(
    "/users/{user_id}/sessions/{session_id}/close",
    response_model=IntegrateResponse,
    status_code=status.HTTP_200_OK,
)
async def close_session(
    user_id: str = Path(..., description="The unique identifier for the user"),
    session_id: str = Path(..., description="The session ID to close and integrate"),
    graph_ops: GraphOps = Depends(get_graph_ops),
):
    try:
        if not await graph_ops.user_exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        logger.info(f"Closing session {session_id} for user {user_id}")
        result = await run_integration_agent(
            user_id=user_id,
            trigger_ids=[],
            graph_ops=graph_ops,
            session_id=session_id,
        )

        return IntegrateResponse(
            status="success" if result.success else "failed",
            links_created=result.links_created,
            merges_applied=result.merges_performed,
            derived_created=0,
            conflicts_flagged=result.flags_raised,
            errors=result.errors,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing session {session_id} for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while closing session",
        )
