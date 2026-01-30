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
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any
from uuid import uuid4
from pydantic import BaseModel, Field

from persona.models.memory import (
    Memory,
    MemoryLink,
    EpisodeOutput,
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

1. **Episode**: What happened (narrative evidence). Always extract one. This is the primary memory.
2. **Psyche**: Who they are (identity-defining traits). RARELY extract - only for significant identity signals.
3. **Entity**: What/who exists (people, places, things). Extract referents with their attributes/facts.
4. **Note**: What to do (tasks, goals). ONLY when explicit intention/trigger is present.

## PSYCHE EXTRACTION (CAPTURE THE WHY)

Psyche represents what drives behavior - preferences, values, beliefs, and identity.
Extract Psyche when you see evaluative language revealing the person's inner landscape.

**EXTRACT psyche when you see:**
- Preferences: "I like/love/hate/prefer...", "I enjoy/dread..."
- Identity: "I'm the kind of person who...", "I always/never..."
- Values/beliefs: "I value...", "I believe...", "What matters to me is..."
- Reactions revealing preference: "That was amazing/terrible", "I had so much fun"
- Recurring patterns with sentiment: doing something repeatedly AND expressing feeling about it

**DO NOT extract psyche for:**
- Neutral activity descriptions: "I went to the store" (no sentiment = Episode only)
- Situational states: "I'm tired today" (temporary, not identity)
- Single mentions without evaluative language

**Psyche types:**
- trait: personality characteristics ("I'm introverted")
- preference: likes/dislikes ("I love hiking")
- value: what matters to them ("Family comes first")
- belief: worldview ("I believe in hard work")

**Guideline**: 1-2 Psyche per session is healthy if evaluative language is present.
Skip Psyche only when the session is purely factual narration.

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

## Guidelines

**Episodes:** Write as narrative prose, preserve emotional context and specifics. Title 2-10 words.
The episode IS the memory - make it rich and retrievable.

**Entities:** Extract people, places, organizations, projects, tools, concepts.
- Include canonical_name and any aliases mentioned
- Include attributes (facts) about the entity - this is where facts live!
- Types: person, place, organization, project, tool, concept
- CRITICAL: Only extract attributes EXPLICITLY stated in the input
- Do NOT infer emotions, outcomes, or experiences unless directly quoted
- Do NOT merge facts from different contexts about similar topics

**Notes:** ONLY for commitments/intentions. Types: task, goal, reminder, idea, list, project.

## Output Format (JSON)
{
  "event_time": "YYYY-MM-DDTHH:MM:SS",
  "episode": {"title": "...", "content": "..."},
  "psyche": [{"type": "...", "content": "..."}],
  "entities": [{"entity_type": "...", "canonical_name": "...", "aliases": [], "description": "...", "attributes": [{"key": "...", "value": "..."}]}],
  "notes": [{"type": "...", "title": "...", "content": "...", "status": "active", "entity_refs": [...]}]
}

Respond with valid JSON only. Empty arrays are expected for psyche in most cases."""


INGESTION_USER_TEMPLATE = """Process this input and extract memories:

**Current Time:** {timestamp}
**Timezone:** {timezone}
**Source:** {source_type}

**Input:**
{raw_content}"""


ENTITY_DEDUP_THRESHOLD = 0.9


def normalize_entity_name(name: str) -> str:
    """Lowercase, strip whitespace, normalize separators to spaces."""
    return name.lower().strip().replace("-", " ").replace("_", " ")


def fuzzy_match(str1: str, str2: str) -> float:
    """Return similarity score 0-1 using SequenceMatcher."""
    return SequenceMatcher(None, str1, str2).ratio()


def find_matching_entity(
    entity_name: str,
    existing_entities: Dict[str, Any],
) -> Optional[str]:
    """Find matching entity by normalized name or fuzzy match (>0.9 threshold)."""
    normalized = normalize_entity_name(entity_name)

    if normalized in existing_entities:
        return normalized

    for existing_norm in existing_entities:
        if fuzzy_match(normalized, existing_norm) >= ENTITY_DEDUP_THRESHOLD:
            return existing_norm

    return None


def merge_entity_attributes(
    existing: Dict[str, Any],
    new_entity: Dict[str, Any],
) -> None:
    """Merge aliases, attributes, and description from new_entity into existing."""
    existing_aliases = set(existing.get("aliases", []) or [])
    new_aliases = set(new_entity.get("aliases", []) or [])
    new_canonical = new_entity.get("canonical_name", "")
    if (
        new_canonical
        and new_canonical.lower() != existing.get("canonical_name", "").lower()
    ):
        new_aliases.add(new_canonical)
    existing["aliases"] = list(existing_aliases | new_aliases)

    existing_attrs = existing.get("attributes", []) or []
    new_attrs = new_entity.get("attributes", []) or []
    for attr in new_attrs:
        key_val = (attr.get("key"), attr.get("value"))
        if not any((a.get("key"), a.get("value")) == key_val for a in existing_attrs):
            existing_attrs.append(attr)
    existing["attributes"] = existing_attrs

    new_desc = new_entity.get("description", "")
    existing_desc = existing.get("description", "")
    if new_desc and len(new_desc) > len(existing_desc):
        existing["description"] = new_desc


def deduplicate_entities(
    entities: List[Any],
) -> tuple[List[Any], Dict[str, Any]]:
    """Deduplicate entities using fuzzy matching (>0.9 threshold), merging attributes."""
    if not entities:
        return [], {
            "entities_before": 0,
            "entities_after": 0,
            "merges_applied": 0,
            "dedup_rate": 0.0,
        }

    entities_before = len(entities)
    seen: Dict[str, Dict[str, Any]] = {}

    for entity in entities:
        if hasattr(entity, "model_dump"):
            entity_dict = entity.model_dump()
        elif hasattr(entity, "__dict__"):
            entity_dict = vars(entity).copy()
        else:
            entity_dict = dict(entity)

        canonical = entity_dict.get("canonical_name", "")
        if not canonical:
            continue

        match_key = find_matching_entity(canonical, seen)

        if match_key:
            merge_entity_attributes(seen[match_key], entity_dict)
        else:
            seen[normalize_entity_name(canonical)] = entity_dict

    deduplicated = []
    if entities and hasattr(entities[0], "model_validate"):
        model_cls = type(entities[0])
        for entity_dict in seen.values():
            try:
                deduplicated.append(model_cls.model_validate(entity_dict))
            except Exception:
                deduplicated.append(type(entities[0])(**entity_dict))
    else:
        deduplicated = list(seen.values())

    entities_after = len(deduplicated)
    merges_applied = entities_before - entities_after

    return deduplicated, {
        "entities_before": entities_before,
        "entities_after": entities_after,
        "merges_applied": merges_applied,
        "dedup_rate": round(merges_applied / entities_before, 3)
        if entities_before > 0
        else 0.0,
    }


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
    dedup_metrics: Optional[Dict[str, Any]] = None


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

            base_event_time = self._parse_event_time(extraction.event_time, timestamp)
            day_id = base_event_time.strftime("%Y-%m-%d")

            extraction_model = config.MACHINE_LEARNING.LLM_SERVICE or "unknown"

            memories: List[Memory] = []
            links: List[MemoryLink] = []

            # Sequence counter for unique event_time per memory (microsecond offsets)
            # This preserves extraction order for timeline queries
            # Seed with observed_at.microsecond * 1000 to avoid cross-ingest collisions
            # when multiple ingests happen in the same second
            memory_seq = observed_at.microsecond * 1000

            def next_event_time() -> datetime:
                nonlocal memory_seq
                et = base_event_time + timedelta(microseconds=memory_seq)
                memory_seq += 1
                return et

            episode_id = uuid4()
            from persona.models.memory import EpisodeMemory, PsycheMemory, NoteMemory

            episode = EpisodeMemory(
                id=episode_id,
                title=extraction.episode.title,
                content=extraction.episode.content,
                event_time=next_event_time(),
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
                    event_time=next_event_time(),
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
                    event_time=next_event_time(),
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

            deduped_entities, dedup_metrics = deduplicate_entities(extraction.entities)
            if dedup_metrics["merges_applied"] > 0:
                logger.info(
                    f"Entity dedup: {dedup_metrics['entities_before']} -> {dedup_metrics['entities_after']} "
                    f"({dedup_metrics['merges_applied']} merged, rate={dedup_metrics['dedup_rate']})"
                )

            for e in deduped_entities:
                # Build rich content that includes attributes for better vector search
                content_parts = [f"{e.entity_type}: {e.canonical_name}"]
                if e.aliases:
                    content_parts.append(f"(also known as: {', '.join(e.aliases)})")
                if e.description:
                    content_parts.append(e.description)
                if e.attributes:
                    attr_strs = [f"{a.key}: {a.value}" for a in e.attributes]
                    content_parts.append("Facts: " + "; ".join(attr_strs))
                entity_content = " | ".join(content_parts)

                entity = EntityMemory(
                    id=uuid4(),
                    entity_type=e.entity_type,
                    canonical_name=e.canonical_name,
                    aliases=e.aliases,
                    description=e.description,
                    title=e.canonical_name,
                    content=entity_content,
                    attributes=[
                        EntityAttribute(
                            key=attr.key,
                            value=attr.value,
                            evidence_id=episode_id,
                        )
                        for attr in e.attributes
                    ],
                    mentioned_in=[episode_id],
                    event_time=next_event_time(),
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
                f"Ingested {len(memories)} memories for user {user_id} | "
                f"LLM: {extract_time_ms:.0f}ms | Embed: {embed_time_ms:.0f}ms | "
                f"Entities: {dedup_metrics['entities_after']} (deduped from {dedup_metrics['entities_before']})"
            )

            return IngestionResult(
                memories=memories,
                links=links,
                success=True,
                extract_time_ms=extract_time_ms,
                embed_time_ms=embed_time_ms,
                dedup_metrics=dedup_metrics,
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
