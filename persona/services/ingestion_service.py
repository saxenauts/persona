"""
Memory Ingestion Service for Persona v2.

Ingests raw input and extracts memories:
- Episode (the narrative record)
- Psyche (traits, values, preferences)
- Notes (tasks, projects, todos, facts, reminders, etc.)

Uses LLM structured output for extraction, then generates embeddings.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Optional, List, Tuple
from uuid import uuid4
from pydantic import BaseModel, Field

from persona.models.memory import (
    Memory,
    MemoryLink,
    EpisodeOutput,
    PsycheOutput,
    NoteOutput,
    IngestionOutput,
)
from persona.llm.client_factory import get_chat_client, get_embedding_client
from persona.llm.providers.base import ChatMessage
from server.logging_config import get_logger

logger = get_logger(__name__)

EXTRACTION_TIMEOUT_SECONDS = 120
EXTRACTION_MAX_RETRIES = 3
EXTRACTION_RETRY_DELAY_SECONDS = 2

# Configuration
EXTRACTION_TIMEOUT_SECONDS = 120  # 2 minutes per extraction attempt
EXTRACTION_MAX_RETRIES = 3
EXTRACTION_RETRY_DELAY_SECONDS = 2


# ============================================================================
# Ingestion Prompt
# ============================================================================

INGESTION_SYSTEM_PROMPT = """You are a memory ingestion system for a personal knowledge assistant. Your job is to process raw input and extract meaningful memories.

Extract:
1. **Episode**: A narrative memory of what happened. Write as natural prose. Preserve emotional context.
2. **Psyche**: Any identity-related content (traits, preferences, values, beliefs) if present.
3. **Notes**: Any structured items (tasks, facts, reminders, contacts, ideas, lists) if present.

## Guidelines

**For Episodes:**
- Write as narrative prose, not lists
- Preserve emotional nuance and context
- Title should be 2-10 words

**For Psyche:**
- Extract when user reveals preferences, values, beliefs, or traits
- Be specific: "prefers remote work" not "has work preferences"
- Types: trait, preference, value, belief

**For Notes:**
- Extract concrete action items, facts, contacts, ideas, or lists
- Types: task, project, reminder, todo, fact, contact, idea, list
- Status is usually "active" for new items

## Output Format
Return valid JSON:
{
  "episode": {"title": "...", "content": "..."},
  "psyche": [{"type": "...", "content": "..."}],
  "notes": [{"type": "...", "title": "...", "content": "...", "status": "active"}]
}

Empty arrays for psyche/notes if none found."""


INGESTION_USER_TEMPLATE = """Process this input and extract memories:

**Timestamp:** {timestamp}
**Source:** {source_type}

**Input:**
{raw_content}"""


# ============================================================================
# Ingestion Result
# ============================================================================


class IngestionResult(BaseModel):
    """Result of memory ingestion."""

    memories: List[Memory] = Field(default_factory=list)
    links: List[MemoryLink] = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    extraction_attempts: int = 1
    extraction_timeout: bool = False
    extract_time_ms: Optional[float] = None
    embed_time_ms: Optional[float] = None
    persist_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None


# ============================================================================
# Ingestion Service
# ============================================================================


class MemoryIngestionService:
    """
    Ingests raw input and produces Memory objects with embeddings.
    """

    def __init__(self):
        self.chat_client = get_chat_client()
        self.embedding_client = get_embedding_client()

    async def ingest(
        self,
        raw_content: str,
        user_id: str,
        timestamp: Optional[datetime] = None,
        session_id: Optional[str] = None,
        source_type: str = "conversation",
        source_ref: Optional[str] = None,
    ) -> IngestionResult:
        """
        Ingest raw content and extract memories.

        Returns IngestionResult with list of Memory objects (episode, psyche, goals).
        """
        timestamp = timestamp or datetime.utcnow()
        day_id = timestamp.strftime("%Y-%m-%d")

        try:
            # Extract via LLM
            start_extract = time.time()
            extraction = await self._extract(raw_content, timestamp, source_type)
            extract_time_ms = (time.time() - start_extract) * 1000

            memories: List[Memory] = []
            links: List[MemoryLink] = []

            # Create episode memory
            episode_id = uuid4()
            from persona.models.memory import EpisodeMemory, PsycheMemory, NoteMemory

            episode = EpisodeMemory(
                id=episode_id,
                title=extraction.episode.title,
                content=extraction.episode.content,
                timestamp=timestamp,
                created_at=datetime.utcnow(),
                day_id=day_id,
                session_id=session_id,
                source_type=source_type,
                source_ref=source_ref,
                user_id=user_id,
            )
            memories.append(episode)

            # Create psyche memories
            for p in extraction.psyche:
                psyche = PsycheMemory(
                    id=uuid4(),
                    psyche_type=p.type,
                    title=p.type,
                    content=p.content,
                    timestamp=timestamp,
                    created_at=datetime.utcnow(),
                    day_id=day_id,
                    source_type=source_type,
                    user_id=user_id,
                )
                memories.append(psyche)
                # Link psyche to source episode
                links.append(
                    MemoryLink(
                        source_id=psyche.id,
                        target_id=episode_id,
                        relation="derived_from",
                    )
                )

            # Create note memories
            for n in extraction.notes:
                note = NoteMemory(
                    id=uuid4(),
                    note_type=n.type,
                    title=n.title,
                    content=n.content,
                    status=n.status,
                    timestamp=timestamp,
                    created_at=datetime.utcnow(),
                    day_id=day_id,
                    source_type=source_type,
                    user_id=user_id,
                )
                memories.append(note)
                # Link note to source episode
                links.append(
                    MemoryLink(
                        source_id=note.id, target_id=episode_id, relation="derived_from"
                    )
                )

            # Generate embeddings
            start_embed = time.time()
            memories = await self._add_embeddings(memories)
            embed_time_ms = (time.time() - start_embed) * 1000

            logger.info(
                f"Ingested {len(memories)} memories for user {user_id} | LLM: {extract_time_ms:.0f}ms | Embed: {embed_time_ms:.0f}ms"
            )

            return IngestionResult(
                memories=memories,
                links=links,
                success=True,
                extract_time_ms=extract_time_ms,
                embed_time_ms=embed_time_ms,
            )

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return IngestionResult(success=False, error=str(e))

    async def _extract(
        self, raw_content: str, timestamp: datetime, source_type: str
    ) -> IngestionOutput:
        """Extract memories via LLM with timeout and retry logic."""

        user_prompt = INGESTION_USER_TEMPLATE.format(
            timestamp=timestamp.strftime("%Y/%m/%d (%a) %H:%M"),
            source_type=source_type,
            raw_content=raw_content,
        )

        prompt_tokens_est = len(INGESTION_SYSTEM_PROMPT) // 4 + len(user_prompt) // 4
        content_chars = len(raw_content)

        last_error = None
        for attempt in range(1, EXTRACTION_MAX_RETRIES + 1):
            try:
                logger.debug(
                    f"Extraction attempt {attempt}/{EXTRACTION_MAX_RETRIES}: ~{prompt_tokens_est} tokens, {content_chars} chars"
                )

                response = await asyncio.wait_for(
                    self.chat_client.chat(
                        messages=[
                            ChatMessage(role="system", content=INGESTION_SYSTEM_PROMPT),
                            ChatMessage(role="user", content=user_prompt),
                        ],
                        response_format={"type": "json_object"},
                    ),
                    timeout=EXTRACTION_TIMEOUT_SECONDS,
                )

                data = json.loads(response.content)
                result = IngestionOutput(**data)

                memory_count = 1 + len(result.psyche) + len(result.notes)
                if attempt > 1:
                    logger.info(
                        f"Extraction succeeded on attempt {attempt}: {memory_count} memories"
                    )

                return result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {EXTRACTION_TIMEOUT_SECONDS}s"
                logger.warning(
                    f"Extraction attempt {attempt} timed out after {EXTRACTION_TIMEOUT_SECONDS}s (content: {content_chars} chars)"
                )

            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                logger.warning(f"Extraction attempt {attempt} JSON parse failed: {e}")

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    f"Extraction attempt {attempt} failed: {type(e).__name__}: {e}"
                )

            if attempt < EXTRACTION_MAX_RETRIES:
                await asyncio.sleep(EXTRACTION_RETRY_DELAY_SECONDS * attempt)

        logger.error(
            f"Extraction failed after {EXTRACTION_MAX_RETRIES} attempts. Last error: {last_error}. Content: {content_chars} chars"
        )
        return IngestionOutput(
            episode=EpisodeOutput(
                title=raw_content[:50] + "..."
                if len(raw_content) > 50
                else raw_content,
                content=raw_content,
            )
        )

        prompt_tokens_est = len(INGESTION_SYSTEM_PROMPT) // 4 + len(user_prompt) // 4
        logger.debug(
            f"LLM extraction: ~{prompt_tokens_est} tokens input, {len(raw_content)} chars content"
        )

        response = await self.chat_client.chat(
            messages=[
                ChatMessage(role="system", content=INGESTION_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_prompt),
            ],
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.content)
            return IngestionOutput(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return IngestionOutput(
                episode=EpisodeOutput(
                    title=raw_content[:50] + "..."
                    if len(raw_content) > 50
                    else raw_content,
                    content=raw_content,
                )
            )

    async def _add_embeddings(self, memories: List[Memory]) -> List[Memory]:
        """Generate embeddings for all memories."""

        texts = [f"{m.title} | {m.content}" for m in memories]

        try:
            embeddings = await self.embedding_client.embeddings(texts)
            for i, m in enumerate(memories):
                m.embedding = embeddings[i]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")

        return memories


# ============================================================================
# Convenience function
# ============================================================================


async def ingest_memory(
    raw_content: str,
    user_id: str,
    timestamp: Optional[datetime] = None,
    session_id: Optional[str] = None,
    source_type: str = "conversation",
    source_ref: Optional[str] = None,
) -> IngestionResult:
    """Convenience function for memory ingestion."""
    service = MemoryIngestionService()
    return await service.ingest(
        raw_content=raw_content,
        user_id=user_id,
        timestamp=timestamp,
        session_id=session_id,
        source_type=source_type,
        source_ref=source_ref,
    )
