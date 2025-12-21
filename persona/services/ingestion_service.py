"""
Unified Memory Ingestion Service for Persona v2.

Ingests raw input and extracts all three memory types:
- Episode - the narrative record
- Psyche - traits, values, preferences
- Goals - tasks, projects, todos

Uses OpenAI Structured Outputs with strict schema for reliable extraction.
"""

import json
from datetime import datetime
from typing import Optional, List, Tuple
from uuid import uuid4
from pydantic import BaseModel, Field

from persona.models.memory import Episode, Psyche, Goal, PsycheCreateInput, GoalCreateInput
from persona.llm.client_factory import get_chat_client, get_embedding_client
from server.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Structured Output Schema (for OpenAI's strict JSON mode)
# ============================================================================

class EpisodeOutput(BaseModel):
    """Episode extraction output."""
    title: str = Field(description="Concise title (2-10 words)")
    content: str = Field(description="Narrative content preserving context and nuance")


class PsycheOutput(BaseModel):
    """Psyche extraction output."""
    type: str = Field(description="Type: trait, preference, value, belief, arc")
    content: str = Field(description="The trait/preference/belief in natural language")


class GoalOutput(BaseModel):
    """Goal extraction output."""
    type: str = Field(description="Type: task, project, reminder, todo, routine")
    title: str = Field(description="Short display title")
    content: str = Field(description="Description or context")
    status: str = Field(default="active", description="Status: active, completed, blocked")


class IngestionOutput(BaseModel):
    """Complete ingestion output with all three memory types."""
    episode: EpisodeOutput
    psyche: List[PsycheOutput] = Field(default_factory=list)
    goals: List[GoalOutput] = Field(default_factory=list)


# ============================================================================
# Ingestion Prompt (using best practices for structured output)
# ============================================================================

INGESTION_SYSTEM_PROMPT = """You are a memory ingestion system for a personal knowledge assistant. Your job is to process raw input and extract meaningful memories.

You will ALWAYS extract:
1. **Episode**: A narrative memory of what happened. Write as natural prose, not bullet points. Preserve emotional context and implications.

You will ALSO extract when present:
2. **Psyche**: Any identity-related content (traits, preferences, values, beliefs). Only extract if the user reveals something about who they are.
3. **Goals**: Any action items, tasks, projects, or todos. Only extract if there's something actionable.

## Guidelines

**For Episodes:**
- Write as narrative prose, not lists
- Preserve emotional nuance and context
- Include implications for the user's life
- Title should be 2-10 words

**For Psyche (only if present):**
- Extract when user reveals preferences, values, beliefs, or traits
- Be specific: "prefers remote work" not "has work preferences"
- Types: trait, preference, value, belief, arc

**For Goals (only if present):**
- Extract concrete action items or commitments
- Include context in the content field
- Types: task, project, reminder, todo, routine
- Status is usually "active" for new items

## Output Format
Return valid JSON matching this exact structure:
{
  "episode": {"title": "...", "content": "..."},
  "psyche": [{"type": "...", "content": "..."}],
  "goals": [{"type": "...", "title": "...", "content": "...", "status": "active"}]
}

If no psyche or goals are found, return empty arrays: "psyche": [], "goals": []"""


INGESTION_USER_TEMPLATE = """Process this input and extract memories:

**Timestamp:** {timestamp}
**Source:** {source_type}

**Input:**
{raw_content}

Extract the episode (required), and any psyche or goals if present."""


# ============================================================================
# Ingestion Service
# ============================================================================

class IngestionResult(BaseModel):
    """Result of memory ingestion."""
    episode: Episode
    psyche: List[Psyche] = Field(default_factory=list)
    goals: List[Goal] = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class MemoryIngestionService:
    """
    Unified memory ingestion service.
    
    Processes raw input through LLM to extract Episode, Psyche, and Goals,
    then generates embeddings for all items.
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
        source_ref: Optional[str] = None
    ) -> IngestionResult:
        """
        Ingest raw content and extract all memory types.
        
        Args:
            raw_content: The text to ingest
            user_id: User this memory belongs to
            timestamp: When this happened (defaults to now)
            session_id: Optional session grouping
            source_type: Origin type (conversation, instagram, etc.)
            source_ref: Reference to original source
            
        Returns:
            IngestionResult with Episode (always) and optional Psyche/Goals
        """
        timestamp = timestamp or datetime.utcnow()
        
        try:
            # Step 1: Extract memories via LLM
            extraction = await self._extract_memories(
                raw_content, timestamp, source_type
            )
            
            # Step 2: Create Episode (always)
            episode = Episode(
                id=uuid4(),
                title=extraction.episode.title,
                content=extraction.episode.content,
                timestamp=timestamp,
                created_at=datetime.utcnow(),
                day_id=timestamp.strftime("%Y-%m-%d"),
                session_id=session_id,
                source_type=source_type,
                source_ref=source_ref,
                user_id=user_id
            )
            
            # Step 3: Create Psyche items (if any)
            psyche_items = []
            for p in extraction.psyche:
                psyche = Psyche(
                    id=uuid4(),
                    type=p.type,
                    content=p.content,
                    timestamp=timestamp,
                    created_at=datetime.utcnow(),
                    source_episode_id=episode.id,
                    user_id=user_id
                )
                psyche_items.append(psyche)
            
            # Step 4: Create Goal items (if any)
            goal_items = []
            for g in extraction.goals:
                goal = Goal(
                    id=uuid4(),
                    type=g.type,
                    title=g.title,
                    content=g.content,
                    status=g.status,
                    timestamp=timestamp,
                    created_at=datetime.utcnow(),
                    source_episode_id=episode.id,
                    user_id=user_id
                )
                goal_items.append(goal)
            
            # Step 5: Generate embeddings for all items
            episode, psyche_items, goal_items = await self._generate_embeddings(
                episode, psyche_items, goal_items
            )
            
            logger.info(
                f"Ingested for user {user_id}: "
                f"1 episode, {len(psyche_items)} psyche, {len(goal_items)} goals"
            )
            
            return IngestionResult(
                episode=episode,
                psyche=psyche_items,
                goals=goal_items,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return IngestionResult(
                episode=Episode(
                    title="Ingestion error",
                    content=raw_content[:500],
                    day_id=timestamp.strftime("%Y-%m-%d"),
                    user_id=user_id
                ),
                success=False,
                error=str(e)
            )
    
    async def _extract_memories(
        self,
        raw_content: str,
        timestamp: datetime,
        source_type: str
    ) -> IngestionOutput:
        """Extract memories using LLM with structured output."""
        
        user_prompt = INGESTION_USER_TEMPLATE.format(
            timestamp=timestamp.strftime("%Y/%m/%d (%a) %H:%M"),
            source_type=source_type,
            raw_content=raw_content
        )
        
        # Use structured output for reliable extraction
        response = await self.chat_client.chat(
            messages=[
                {"role": "system", "content": INGESTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        try:
            data = json.loads(response.content)
            return IngestionOutput(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback: create minimal episode
            return IngestionOutput(
                episode=EpisodeOutput(
                    title=raw_content[:50] + "..." if len(raw_content) > 50 else raw_content,
                    content=raw_content
                )
            )
    
    async def _generate_embeddings(
        self,
        episode: Episode,
        psyche_items: List[Psyche],
        goal_items: List[Goal]
    ) -> Tuple[Episode, List[Psyche], List[Goal]]:
        """Generate embeddings for all memory items."""
        
        # Collect all texts to embed
        texts = []
        
        # Episode: title + content
        episode_text = f"{episode.title} | {episode.content}"
        texts.append(episode_text)
        
        # Psyche: type + content
        for p in psyche_items:
            texts.append(f"{p.type}: {p.content}")
        
        # Goals: type + title + content
        for g in goal_items:
            texts.append(f"{g.type}: {g.title} | {g.content}")
        
        if not texts:
            return episode, psyche_items, goal_items
        
        # Generate all embeddings in one batch call
        try:
            embeddings = await self.embedding_client.embed(texts)
            
            # Assign embeddings back
            idx = 0
            episode.embedding = embeddings[idx]
            idx += 1
            
            for p in psyche_items:
                p.embedding = embeddings[idx]
                idx += 1
            
            for g in goal_items:
                g.embedding = embeddings[idx]
                idx += 1
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Continue without embeddings
        
        return episode, psyche_items, goal_items


# ============================================================================
# Convenience function
# ============================================================================

async def ingest_memory(
    raw_content: str,
    user_id: str,
    timestamp: Optional[datetime] = None,
    session_id: Optional[str] = None,
    source_type: str = "conversation",
    source_ref: Optional[str] = None
) -> IngestionResult:
    """
    Convenience function for memory ingestion.
    
    Returns IngestionResult with:
    - episode: Always present
    - psyche: List (may be empty)
    - goals: List (may be empty)
    - success: bool
    - error: str if failed
    """
    service = MemoryIngestionService()
    return await service.ingest(
        raw_content=raw_content,
        user_id=user_id,
        timestamp=timestamp,
        session_id=session_id,
        source_type=source_type,
        source_ref=source_ref
    )
