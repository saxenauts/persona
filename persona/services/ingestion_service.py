"""
Memory Ingestion Service for Persona.

Ingests raw input and extracts memories:
- Episode (the narrative record)
- Psyche (traits, values, preferences)
- Notes (tasks, goals, reminders)

Uses LLM structured output for extraction, then generates embeddings.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Optional, List
from uuid import uuid4
from pydantic import BaseModel, Field

from persona.models.memory import (
    Memory,
    MemoryLink,
    EpisodeOutput,
    PsycheOutput,
    NoteOutput,
    EntityOutput,
    IngestionOutput,
)
from persona.llm.client_factory import get_chat_client, get_embedding_client
from persona.llm.providers.base import ChatMessage
from server.logging_config import get_logger
from server.config import config

logger = get_logger(__name__)

# Configuration
EXTRACTION_TIMEOUT_SECONDS = 120  # 2 minutes per extraction attempt
EXTRACTION_MAX_RETRIES = 3
EXTRACTION_RETRY_DELAY_SECONDS = 2


# ============================================================================
# Ingestion Prompt
# ============================================================================

INGESTION_SYSTEM_PROMPT = """You are a memory ingestion system for a personal knowledge assistant. Your job is to process raw input and extract meaningful memories into 4 pillars.

## The 4 Memory Pillars

1. **Episode**: What happened (narrative evidence). Always extract one.
2. **Psyche**: Who they are (traits, preferences, values, beliefs). Extract if present.
3. **Entity**: What/who exists (people, places, things, concepts). Extract referents mentioned.
4. **Note**: What to do (tasks, goals, reminders). ONLY when intention/trigger is present.

## Entity vs Note (CRITICAL DISTINCTION)

- **Entity** = Things that EXIST (nouns): "Sarah", "Paris", "Project Alpha"
- **Note** = Things to DO (intentions): "call Sarah", "book trip to Paris"
- **Facts** = Entity ATTRIBUTES, not Notes: "Sarah's birthday is June 5" → Entity attribute

Create Notes ONLY when you see intention signals:
- "remind me", "I need to", "I should", "don't forget"
- Due dates or deadlines
- Imperatives or action items

## Temporal Extraction (CRITICAL)

You MUST extract `event_time` - when the event ACTUALLY happened, not when it was recorded.
- Use the provided current time and timezone as reference
- Resolve relative references: "yesterday" → actual date, "last week" → actual date range
- If ambiguous, use the current timestamp as event_time
- Format: ISO 8601 (YYYY-MM-DDTHH:MM:SS)

Examples:
- "I had coffee with Sarah yesterday" (current: 2026-01-02) → event_time: "2026-01-01T12:00:00"
- "Back in 2020, I quit my job" → event_time: "2020-06-01T00:00:00" (approximate)
- "Just talked to mom" → event_time: same as current timestamp

## Guidelines

**Episodes:** Write as narrative prose, preserve emotional context. Title 2-10 words.

**Psyche:** Extract preferences, values, beliefs, traits. Types: trait, preference, value, belief.

**Entities:** Extract people, places, organizations, projects, tools, concepts.
- Include canonical_name and any aliases mentioned
- Include attributes (facts) about the entity
- Types: person, place, organization, project, tool, concept

**Notes:** ONLY for commitments/intentions. Types: task, goal, reminder, idea, list, project.
- Include entity_refs: names of entities the note relates to

## Output Format (JSON)
{
  "event_time": "YYYY-MM-DDTHH:MM:SS",
  "episode": {"title": "...", "content": "..."},
  "psyche": [{"type": "...", "content": "..."}],
  "entities": [{"entity_type": "...", "canonical_name": "...", "aliases": [], "description": "...", "attributes": [{"key": "...", "value": "..."}]}],
  "notes": [{"type": "...", "title": "...", "content": "...", "status": "active", "entity_refs": [...]}]
}

Respond with valid JSON only. Empty arrays if none found for psyche/entities/notes."""


INGESTION_USER_TEMPLATE = """Process this input and extract memories:

**Current Time:** {timestamp}
**Timezone:** {timezone}
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
        timezone: str = "UTC",
        session_id: Optional[str] = None,
        source_type: str = "conversation",
        source_ref: Optional[str] = None,
    ) -> IngestionResult:
        """
        Ingest raw content and extract memories.

        Args:
            raw_content: The raw text to process
            user_id: User ID for this memory
            timestamp: Current time (for context). Defaults to now.
            timezone: User's timezone (e.g., "America/Los_Angeles"). Used for temporal extraction.
            session_id: Optional session identifier
            source_type: Type of source content
            source_ref: Reference to source

        Returns IngestionResult with list of Memory objects (episode, psyche, notes, entities).
        """
        observed_at = datetime.utcnow()
        timestamp = timestamp or observed_at

        try:
            start_extract = time.time()
            extraction = await self._extract(
                raw_content, timestamp, timezone, source_type
            )
            extract_time_ms = (time.time() - start_extract) * 1000

            event_time = self._parse_event_time(extraction.event_time, timestamp)
            day_id = event_time.strftime("%Y-%m-%d")

            extraction_model = config.MACHINE_LEARNING.LLM_SERVICE or "unknown"

            memories: List[Memory] = []
            links: List[MemoryLink] = []

            episode_id = uuid4()
            from persona.models.memory import EpisodeMemory, PsycheMemory, NoteMemory

            episode = EpisodeMemory(
                id=episode_id,
                title=extraction.episode.title,
                content=extraction.episode.content,
                event_time=event_time,
                observed_at=observed_at,
                day_id=day_id,
                session_id=session_id,
                source_type=source_type,
                source_ref=source_ref,
                extraction_model=extraction_model,
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
                    event_time=event_time,
                    observed_at=observed_at,
                    day_id=day_id,
                    session_id=session_id,
                    source_type=source_type,
                    extraction_model=extraction_model,
                    user_id=user_id,
                )
                memories.append(psyche)
                links.append(
                    MemoryLink(
                        source_id=psyche.id,
                        target_id=episode_id,
                        relation="derived_from",
                    )
                )

            for n in extraction.notes:
                note = NoteMemory(
                    id=uuid4(),
                    note_type=n.type,
                    title=n.title,
                    content=n.content,
                    status=n.status,
                    event_time=event_time,
                    observed_at=observed_at,
                    day_id=day_id,
                    session_id=session_id,
                    source_type=source_type,
                    extraction_model=extraction_model,
                    user_id=user_id,
                )
                memories.append(note)
                links.append(
                    MemoryLink(
                        source_id=note.id, target_id=episode_id, relation="derived_from"
                    )
                )

            from persona.models.memory import EntityMemory, EntityAttribute

            for e in extraction.entities:
                entity = EntityMemory(
                    id=uuid4(),
                    entity_type=e.entity_type,
                    canonical_name=e.canonical_name,
                    aliases=e.aliases,
                    description=e.description,
                    title=e.canonical_name,
                    content=e.description or f"{e.entity_type}: {e.canonical_name}",
                    attributes=[
                        EntityAttribute(
                            key=attr.key,
                            value=attr.value,
                            evidence_id=episode_id,
                        )
                        for attr in e.attributes
                    ],
                    mentioned_in=[episode_id],
                    event_time=event_time,
                    observed_at=observed_at,
                    day_id=day_id,
                    session_id=session_id,
                    source_type=source_type,
                    extraction_model=extraction_model,
                    user_id=user_id,
                )
                memories.append(entity)
                links.append(
                    MemoryLink(
                        source_id=episode_id,
                        target_id=entity.id,
                        relation="MENTIONS",
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
        self, raw_content: str, timestamp: datetime, timezone: str, source_type: str
    ) -> IngestionOutput:
        """Extract memories via LLM with timeout and retry logic."""

        user_prompt = INGESTION_USER_TEMPLATE.format(
            timestamp=timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
            timezone=timezone,
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

                data = json.loads(response.content or "{}")
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

    def _parse_event_time(
        self, event_time_str: Optional[str], fallback: datetime
    ) -> datetime:
        """Parse LLM-extracted event_time with validation guardrails.

        Validates:
        - Rejects future dates (more than 1 day ahead) - LLM hallucination
        - Clips extreme past dates (older than 100 years) - unreasonable
        - Falls back to provided timestamp on any parse failure
        """
        if not event_time_str:
            return fallback

        try:
            parsed = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))

            # Make naive datetime timezone-aware for comparison
            now = datetime.now(parsed.tzinfo) if parsed.tzinfo else datetime.utcnow()

            # Reject future dates (more than 1 day ahead - accounts for timezone drift)
            max_future = now + timedelta(days=1)
            if parsed > max_future:
                logger.warning(
                    f"Rejected future event_time '{event_time_str}' (>{max_future.isoformat()}), using fallback"
                )
                return fallback

            # Clip extreme past dates (older than 100 years)
            min_past = now - timedelta(days=365 * 100)
            if parsed < min_past:
                logger.warning(
                    f"Rejected ancient event_time '{event_time_str}' (<{min_past.year}), using fallback"
                )
                return fallback

            return parsed

        except (ValueError, TypeError) as e:
            logger.warning(
                f"Failed to parse event_time '{event_time_str}': {e}, using fallback"
            )
            return fallback

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
    timezone: str = "UTC",
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
        timezone=timezone,
        session_id=session_id,
        source_type=source_type,
        source_ref=source_ref,
    )
