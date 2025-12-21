"""
Episode extraction service for converting raw input to narrative episodes.

Uses LLM to transform conversations and raw data into coherent narrative
memory units that preserve context and emotional nuance.
"""

import json
from datetime import datetime
from typing import Optional
from uuid import uuid4

from persona.models.episode import Episode, EpisodeCreateRequest
from persona.llm.prompts import (
    EPISODE_EXTRACTION_SYSTEM_PROMPT,
    EPISODE_EXTRACTION_USER_TEMPLATE
)
from persona.llm.client_factory import get_chat_client
from server.logging_config import get_logger

logger = get_logger(__name__)


class EpisodeExtractor:
    """
    Extracts narrative episodes from raw input using LLM.
    
    Replaces triplet extraction with contextual narrative generation
    that preserves emotional nuance and implications.
    """
    
    def __init__(self):
        self.client = get_chat_client()
    
    async def extract_episode(
        self,
        request: EpisodeCreateRequest,
        user_id: str
    ) -> Episode:
        """
        Convert raw input into a narrative episode.
        
        Args:
            request: The raw content and metadata
            user_id: User this episode belongs to
            
        Returns:
            A structured Episode with title, content, and temporal anchoring
        """
        timestamp = request.timestamp or datetime.utcnow()
        
        # Build the prompt
        user_prompt = EPISODE_EXTRACTION_USER_TEMPLATE.format(
            timestamp=timestamp.strftime("%Y/%m/%d (%a) %H:%M"),
            source_type=request.source_type,
            raw_content=request.raw_content
        )
        
        # Call the LLM
        try:
            response = await self.client.chat(
                messages=[
                    {"role": "system", "content": EPISODE_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.content)
            
            title = result.get("title", "Untitled episode")
            content = result.get("content", request.raw_content)
            
        except Exception as e:
            logger.error(f"Failed to extract episode via LLM: {e}")
            # Fallback: use raw content directly
            title = self._generate_fallback_title(request.raw_content)
            content = request.raw_content
        
        # Create the Episode
        episode = Episode(
            id=uuid4(),
            title=title,
            content=content,
            timestamp=timestamp,
            day_id=timestamp.strftime("%Y-%m-%d"),
            session_id=request.session_id,
            source_type=request.source_type,
            source_ref=request.source_ref,
            user_id=user_id
        )
        
        logger.info(f"Extracted episode: '{title}' for user {user_id}")
        return episode
    
    def _generate_fallback_title(self, content: str) -> str:
        """Generate a simple title from content when LLM fails."""
        # Take first 50 chars, find last word boundary
        if len(content) <= 50:
            return content
        
        truncated = content[:50]
        last_space = truncated.rfind(' ')
        if last_space > 20:
            return truncated[:last_space] + "..."
        return truncated + "..."


async def extract_episode(
    raw_content: str,
    user_id: str,
    timestamp: Optional[datetime] = None,
    session_id: Optional[str] = None,
    source_type: str = "conversation",
    source_ref: Optional[str] = None
) -> Episode:
    """
    Convenience function for extracting an episode from raw content.
    
    Args:
        raw_content: The text to convert into an episode
        user_id: User this episode belongs to
        timestamp: When this happened (defaults to now)
        session_id: Optional session grouping
        source_type: Origin type (conversation, instagram, etc.)
        source_ref: Reference to original source
        
    Returns:
        A structured Episode
    """
    request = EpisodeCreateRequest(
        raw_content=raw_content,
        timestamp=timestamp,
        session_id=session_id,
        source_type=source_type,
        source_ref=source_ref
    )
    
    extractor = EpisodeExtractor()
    return await extractor.extract_episode(request, user_id)
