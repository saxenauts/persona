from fastapi import (
    APIRouter,
    HTTPException,
    status,
    Path,
    Depends,
    Body,
    Response,
    Query,
)
from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.models.schema import UserCreate, AskRequest, AskResponse
from persona.services.user_service import UserService
from persona.services.persona_service import PersonaService
from persona.services.integration_agent import run_integration_agent
from persona.adapters import PersonaAdapter
from persona.utils.session import get_session_id
from server.dependencies import get_graph_ops
from server.logging_config import get_logger
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List, Any
from datetime import datetime
import os
import re


logger = get_logger(__name__)


# --- Request Models (replacing legacy UnstructuredData) ---
class IngestRequest(BaseModel):
    content: str = Field(..., description="Raw text content to ingest.")
    source_type: str = Field(default="conversation")
    provider_session_id: Optional[str] = Field(default=None)
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Optional event timestamp for correct chronology (ISO 8601).",
    )
    store_transcript: bool = Field(default=False)
    finalize_session: bool = Field(
        default=False,
        description="If true, run integration after ingest (use for last message in session).",
    )
    metadata: Optional[Dict[str, str]] = Field(default=None)


class IngestBatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., description="Raw text content to ingest.")
    source_type: str = Field(default="conversation")
    provider_session_id: Optional[str] = Field(default=None)
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Optional event timestamp for correct chronology (ISO 8601).",
    )


class IngestBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[IngestBatchItem] = Field(..., description="List of items to ingest.")


class TimingsMs(BaseModel):
    extract: float = Field(default=0.0)
    embed: float = Field(default=0.0)
    persist: float = Field(default=0.0)
    total: float = Field(default=0.0)


class IngestResponse(BaseModel):
    message: str
    session_id: str
    memories_created: int
    memories_created_by_type: Dict[str, int]
    links_created: int
    timings_ms: TimingsMs


class IngestBatchResponse(BaseModel):
    message: str
    session_ids: List[str]
    memories_created: int
    memories_created_by_type: Dict[str, int]
    links_created: int
    timings_ms: TimingsMs


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


@router.post(
    "/users/{user_id}/ingest",
    response_model=IngestResponse,
    status_code=201,
)
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
            timestamp=data.timestamp,
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
            "session_id": session_id,
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


@router.post(
    "/users/{user_id}/ingest/batch",
    response_model=IngestBatchResponse,
    status_code=201,
)
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

        session_ids = [
            get_session_id(item.source_type, item.provider_session_id)
            for item in batch_data.items
        ]

        adapter = PersonaAdapter(user_id, graph_ops)
        items_for_adapter = [
            {
                "content": item.content,
                "source_type": item.source_type,
                "timestamp": item.timestamp,
                "session_id": session_id,
            }
            for item, session_id in zip(batch_data.items, session_ids)
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
            "session_ids": session_ids,
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


class IntegrateResponse(BaseModel):
    status: str
    links_created: int = 0
    merges_applied: int = 0
    derived_created: int = 0
    conflicts_flagged: int = 0
    errors: List[str] = Field(default_factory=list)


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


class MemoryItem(BaseModel):
    id: str
    type: str
    title: str
    snippet: str
    event_time: Optional[str] = None


class MemoriesResponse(BaseModel):
    user_id: str
    total: int
    by_type: Dict[str, int]
    memories: List[MemoryItem]


@router.get(
    "/users/{user_id}/memories",
    response_model=MemoriesResponse,
    status_code=status.HTTP_200_OK,
    tags=["debug"],
)
async def list_memories(
    user_id: str = Path(..., description="The unique identifier for the user"),
    memory_type: Optional[str] = Query(
        None, description="Filter by type: episode, psyche, entity, note"
    ),
    limit: int = Query(50, ge=1, le=500, description="Max memories to return"),
    graph_ops: GraphOps = Depends(get_graph_ops),
):
    """
    List all stored memories for a user.

    Debug endpoint for inspecting what was ingested.
    """
    try:
        if not await graph_ops.user_exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        store = MemoryStore(graph_ops.graph_db, graph_ops.vector_store)

        if memory_type:
            memories = await store.get_by_type(memory_type, user_id, limit=limit)
        else:
            memories = await store.get_recent(user_id, limit=limit)

        type_counts: Dict[str, int] = {}
        items = []
        for mem in memories:
            mem_type = mem.type
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

            if mem_type == "entity":
                parts = [
                    f"{getattr(mem, 'entity_type', 'entity')}: {getattr(mem, 'canonical_name', mem.title)}"
                ]
                desc = getattr(mem, "description", "")
                if desc:
                    parts.append(desc[:100])
                attrs = getattr(mem, "attributes", [])
                if attrs:
                    attr_strs = [f"{a.key}: {a.value}" for a in attrs[:5]]
                    parts.append("Facts: " + "; ".join(attr_strs))
                snippet = " | ".join(parts)
            else:
                content = getattr(mem, "content", "") or ""
                snippet = content[:200] + "..." if len(content) > 200 else content

            items.append(
                MemoryItem(
                    id=str(mem.id),
                    type=mem_type,
                    title=mem.title or "",
                    snippet=snippet,
                    event_time=mem.event_time.isoformat() if mem.event_time else None,
                )
            )

        return MemoriesResponse(
            user_id=user_id,
            total=len(items),
            by_type=type_counts,
            memories=items,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing memories for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class RecallRequest(BaseModel):
    query: str = Field(..., description="Search query")
    memory_types: Optional[List[str]] = Field(
        None, description="Filter by types: episode, psyche, entity, note"
    )
    limit: int = Field(10, ge=1, le=50, description="Max results")


class RecallHit(BaseModel):
    id: str
    type: str
    title: str
    snippet: str
    score: float
    event_time: Optional[str] = None


class RecallResponse(BaseModel):
    query: str
    count: int
    hits: List[RecallHit]


@router.post(
    "/users/{user_id}/recall",
    response_model=RecallResponse,
    status_code=status.HTTP_200_OK,
    tags=["debug"],
)
async def test_recall(
    user_id: str = Path(..., description="The unique identifier for the user"),
    request: RecallRequest = Body(...),
    graph_ops: GraphOps = Depends(get_graph_ops),
):
    """
    Test the recall tool directly.

    Debug endpoint for verifying what memories are retrieved for a query.
    """
    try:
        if not await graph_ops.user_exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        from uuid import UUID
        from persona.tools.memory import _memory_to_hit

        store = MemoryStore(graph_ops.graph_db, graph_ops.vector_store)

        results = await graph_ops.text_similarity_search(
            query=request.query,
            user_id=user_id,
            limit=request.limit * 2,
        )

        hits = []
        for r in results.get("results", []):
            node_id = r.get("nodeName")
            score = r.get("score", 0.0)

            try:
                mem = await store.get(UUID(node_id), user_id)
                if mem:
                    if request.memory_types and mem.type not in request.memory_types:
                        continue

                    hit = _memory_to_hit(mem, score=score)
                    hits.append(
                        RecallHit(
                            id=hit.id,
                            type=hit.type,
                            title=hit.title,
                            snippet=hit.snippet,
                            score=hit.score,
                            event_time=hit.event_time,
                        )
                    )
                    if len(hits) >= request.limit:
                        break
            except Exception as e:
                logger.debug(f"Could not retrieve memory {node_id}: {e}")

        return RecallResponse(
            query=request.query,
            count=len(hits),
            hits=hits,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recall for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class MemoryStatsResponse(BaseModel):
    user_id: str
    total_memories: int
    by_type: Dict[str, int]
    recent_titles: List[str]


@router.get(
    "/users/{user_id}/memories/stats",
    response_model=MemoryStatsResponse,
    status_code=status.HTTP_200_OK,
    tags=["debug"],
)
async def get_memory_stats(
    user_id: str = Path(..., description="The unique identifier for the user"),
    graph_ops: GraphOps = Depends(get_graph_ops),
):
    """
    Get quick stats about stored memories.

    Debug endpoint for verifying ingestion worked.
    """
    try:
        if not await graph_ops.user_exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        store = MemoryStore(graph_ops.graph_db, graph_ops.vector_store)
        all_memories = await store.get_recent(user_id, limit=500)

        type_counts: Dict[str, int] = {}
        for mem in all_memories:
            type_counts[mem.type] = type_counts.get(mem.type, 0) + 1

        recent_titles = [mem.title or "(untitled)" for mem in all_memories[:10]]

        return MemoryStatsResponse(
            user_id=user_id,
            total_memories=len(all_memories),
            by_type=type_counts,
            recent_titles=recent_titles,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class MemeplexResponse(BaseModel):
    user_id: str
    topics: List[str] = Field(default_factory=list)
    people: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    places: List[str] = Field(default_factory=list)
    concepts: List[str] = Field(default_factory=list)
    last_week_topics: List[str] = Field(default_factory=list)
    last_month_topics: List[str] = Field(default_factory=list)
    recent_focus: str = ""
    updated_at: Optional[str] = None


@router.get(
    "/users/{user_id}/memeplex",
    response_model=MemeplexResponse,
    status_code=status.HTTP_200_OK,
)
async def get_memeplex(
    user_id: str = Path(..., description="The unique identifier for the user"),
    graph_ops: GraphOps = Depends(get_graph_ops),
):
    try:
        if not await graph_ops.user_exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        store = MemoryStore(graph_ops.graph_db, graph_ops.vector_store)
        memeplex = await store.get_memeplex(user_id)

        if not memeplex:
            return MemeplexResponse(user_id=user_id)

        return MemeplexResponse(
            user_id=user_id,
            topics=memeplex.topics,
            people=memeplex.people,
            projects=memeplex.projects,
            places=memeplex.places,
            concepts=memeplex.concepts,
            last_week_topics=memeplex.last_week_topics,
            last_month_topics=memeplex.last_month_topics,
            recent_focus=memeplex.recent_focus,
            updated_at=memeplex.updated_at.isoformat() if memeplex.updated_at else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memeplex for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class MemeplexRefreshResponse(BaseModel):
    status: str
    topics_count: int = 0
    entities_count: int = 0
    updated_at: Optional[str] = None


@router.post(
    "/users/{user_id}/memeplex/refresh",
    response_model=MemeplexRefreshResponse,
    status_code=status.HTTP_200_OK,
)
async def refresh_memeplex(
    user_id: str = Path(..., description="The unique identifier for the user"),
    as_of: Optional[str] = Query(
        None, description="ISO 8601 timestamp to anchor memeplex refresh (for eval use)"
    ),
    graph_ops: GraphOps = Depends(get_graph_ops),
):
    from persona.services.consolidation_service import refresh_memeplex as do_refresh

    try:
        if not await graph_ops.user_exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        store = MemoryStore(graph_ops.graph_db, graph_ops.vector_store)
        as_of_dt = None
        if as_of:
            try:
                as_of_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid as_of timestamp: {as_of}"
                )
        memeplex = await do_refresh(user_id, graph_ops, store, as_of=as_of_dt)

        if not memeplex:
            return MemeplexRefreshResponse(status="no_data")

        entities_count = (
            len(memeplex.people)
            + len(memeplex.projects)
            + len(memeplex.places)
            + len(memeplex.concepts)
        )

        return MemeplexRefreshResponse(
            status="refreshed",
            topics_count=len(memeplex.topics),
            entities_count=entities_count,
            updated_at=memeplex.updated_at.isoformat() if memeplex.updated_at else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing memeplex for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
