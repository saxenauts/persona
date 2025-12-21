"""
Memory Ingestion Service for Persona v2.

Ingests raw input and extracts memories:
- Episode (the narrative record)
- Psyche (traits, values, preferences)
- Goals (tasks, projects, todos)

Uses LLM structured output for extraction, then generates embeddings.
"""

import json
from datetime import datetime
from typing import Optional, List, Tuple
from uuid import uuid4
from pydantic import BaseModel, Field

from persona.models.memory import Memory, MemoryLink, EpisodeOutput, PsycheOutput, GoalOutput, IngestionOutput
from persona.llm.client_factory import get_chat_client, get_embedding_client
from server.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Ingestion Prompt
# ============================================================================

INGESTION_SYSTEM_PROMPT = """You are a memory ingestion system for a personal knowledge assistant. Your job is to process raw input and extract meaningful memories.

Extract:
1. **Episode**: A narrative memory of what happened. Write as natural prose. Preserve emotional context.
2. **Psyche**: Any identity-related content (traits, preferences, values, beliefs) if present.
3. **Goals**: Any action items, tasks, projects, or todos if present.

## Guidelines

**For Episodes:**
- Write as narrative prose, not lists
- Preserve emotional nuance and context
- Title should be 2-10 words

**For Psyche:**
- Extract when user reveals preferences, values, beliefs, or traits
- Be specific: "prefers remote work" not "has work preferences"
- Types: trait, preference, value, belief

**For Goals:**
- Extract concrete action items or commitments
- Types: task, project, reminder, todo
- Status is usually "active" for new items

## Output Format
Return valid JSON:
{
  "episode": {"title": "...", "content": "..."},
  "psyche": [{"type": "...", "content": "..."}],
  "goals": [{"type": "...", "title": "...", "content": "...", "status": "active"}]
}

Empty arrays for psyche/goals if none found."""


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
        source_ref: Optional[str] = None
    ) -> IngestionResult:
        """
        Ingest raw content and extract memories.
        
        Returns IngestionResult with list of Memory objects (episode, psyche, goals).
        """
        timestamp = timestamp or datetime.utcnow()
        day_id = timestamp.strftime("%Y-%m-%d")
        
        try:
            # Extract via LLM
            extraction = await self._extract(raw_content, timestamp, source_type)
            
            memories: List[Memory] = []
            links: List[MemoryLink] = []
            
            # Create episode memory
            episode_id = uuid4()
            episode = Memory(
                id=episode_id,
                type="episode",
                title=extraction.episode.title,
                content=extraction.episode.content,
                timestamp=timestamp,
                created_at=datetime.utcnow(),
                day_id=day_id,
                session_id=session_id,
                source_type=source_type,
                source_ref=source_ref,
                user_id=user_id
            )
            memories.append(episode)
            
            # Create psyche memories
            for p in extraction.psyche:
                psyche = Memory(
                    id=uuid4(),
                    type="psyche",
                    title=p.type,
                    content=p.content,
                    timestamp=timestamp,
                    created_at=datetime.utcnow(),
                    day_id=day_id,
                    source_type=source_type,
                    user_id=user_id
                )
                memories.append(psyche)
                # Link psyche to source episode
                links.append(MemoryLink(
                    source_id=psyche.id,
                    target_id=episode_id,
                    relation="derived_from"
                ))
            
            # Create goal memories
            for g in extraction.goals:
                goal = Memory(
                    id=uuid4(),
                    type="goal",
                    title=g.title,
                    content=g.content,
                    status=g.status,
                    timestamp=timestamp,
                    created_at=datetime.utcnow(),
                    day_id=day_id,
                    source_type=source_type,
                    user_id=user_id,
                    properties={"goal_type": g.type}
                )
                memories.append(goal)
                # Link goal to source episode
                links.append(MemoryLink(
                    source_id=goal.id,
                    target_id=episode_id,
                    relation="derived_from"
                ))
            
            # Generate embeddings
            memories = await self._add_embeddings(memories)
            
            logger.info(f"Ingested {len(memories)} memories for user {user_id}")
            
            return IngestionResult(
                memories=memories,
                links=links,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return IngestionResult(success=False, error=str(e))
    
    async def _extract(
        self,
        raw_content: str,
        timestamp: datetime,
        source_type: str
    ) -> IngestionOutput:
        """Extract memories via LLM."""
        
        user_prompt = INGESTION_USER_TEMPLATE.format(
            timestamp=timestamp.strftime("%Y/%m/%d (%a) %H:%M"),
            source_type=source_type,
            raw_content=raw_content
        )
        
        response = await self.chat_client.chat(
            messages=[
                {"role": "system", "content": INGESTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            data = json.loads(response.content)
            return IngestionOutput(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return IngestionOutput(
                episode=EpisodeOutput(
                    title=raw_content[:50] + "..." if len(raw_content) > 50 else raw_content,
                    content=raw_content
                )
            )
    
    async def _add_embeddings(self, memories: List[Memory]) -> List[Memory]:
        """Generate embeddings for all memories."""
        
        texts = [f"{m.title} | {m.content}" for m in memories]
        
        try:
            embeddings = await self.embedding_client.embed(texts)
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
    source_ref: Optional[str] = None
) -> IngestionResult:
    """Convenience function for memory ingestion."""
    service = MemoryIngestionService()
    return await service.ingest(
        raw_content=raw_content,
        user_id=user_id,
        timestamp=timestamp,
        session_id=session_id,
        source_type=source_type,
        source_ref=source_ref
    )
