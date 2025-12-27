"""
LLM-Enhanced Query Expansion for Retrieval.

Expands natural language queries into structured retrieval hints:
- Date ranges (temporal references like "last week")
- Entity references (people, places mentioned)
- Relationship threads (conversation topics to traverse)
"""

import json
from datetime import datetime, timedelta, date
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field

from persona.llm.client_factory import get_chat_client
from persona.llm.providers.base import ChatMessage
from server.logging_config import get_logger

logger = get_logger(__name__)


class DateRange(BaseModel):
    start: date
    end: date


class QueryExpansion(BaseModel):
    original_query: str
    date_range: Optional[DateRange] = None
    entities: List[str] = Field(default_factory=list)
    relationship_threads: List[str] = Field(default_factory=list)
    semantic_query: str = Field(default="")


QUERY_EXPANSION_PROMPT = """You are a query analyzer for a personal memory system. Analyze the user's query and extract structured retrieval hints.

Given a query and the current date, extract:
1. **date_range**: If the query mentions a time period (e.g., "last week", "yesterday", "in January"), compute the actual date range. Use null if no temporal reference.
2. **entities**: Extract any named entities (people, places, organizations, specific things like "my car", "the gym").
3. **relationship_threads**: Identify topic threads that might help find related memories (e.g., "fitness_journey", "work_projects", "family_events").
4. **semantic_query**: Clean the query for vector search - remove temporal qualifiers, keep semantic meaning.

Current date: {current_date}
User timezone: {timezone}

Return JSON:
{
  "date_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} or null,
  "entities": ["entity1", "entity2"],
  "relationship_threads": ["thread1", "thread2"],
  "semantic_query": "cleaned query for vector search"
}"""


async def expand_query(
    query: str, user_timezone: str = "UTC", current_date: Optional[date] = None
) -> QueryExpansion:
    """
    Expand a natural language query into structured retrieval hints.

    Uses LLM to parse temporal references, extract entities, and identify
    relationship threads that can help with graph traversal.
    """
    if current_date is None:
        current_date = date.today()

    try:
        chat_client = get_chat_client()

        system_prompt = QUERY_EXPANSION_PROMPT.format(
            current_date=current_date.isoformat(), timezone=user_timezone
        )

        response = await chat_client.chat(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=f"Query: {query}"),
            ],
            response_format={"type": "json_object"},
        )

        data = json.loads(response.content)

        date_range = None
        if data.get("date_range"):
            dr = data["date_range"]
            date_range = DateRange(
                start=date.fromisoformat(dr["start"]), end=date.fromisoformat(dr["end"])
            )

        return QueryExpansion(
            original_query=query,
            date_range=date_range,
            entities=data.get("entities", []),
            relationship_threads=data.get("relationship_threads", []),
            semantic_query=data.get("semantic_query", query),
        )

    except Exception as e:
        logger.warning(f"Query expansion failed, using fallback: {e}")
        return _fallback_expansion(query, current_date)


def _fallback_expansion(query: str, current_date: date) -> QueryExpansion:
    """
    Rule-based fallback when LLM expansion fails.

    Handles common temporal patterns without LLM.
    """
    query_lower = query.lower()
    date_range = None

    if "yesterday" in query_lower:
        yesterday = current_date - timedelta(days=1)
        date_range = DateRange(start=yesterday, end=yesterday)
    elif "last week" in query_lower or "past week" in query_lower:
        date_range = DateRange(start=current_date - timedelta(days=7), end=current_date)
    elif "last month" in query_lower or "past month" in query_lower:
        date_range = DateRange(
            start=current_date - timedelta(days=30), end=current_date
        )
    elif "today" in query_lower:
        date_range = DateRange(start=current_date, end=current_date)

    return QueryExpansion(
        original_query=query,
        date_range=date_range,
        entities=[],
        relationship_threads=[],
        semantic_query=query,
    )


def date_range_to_cypher_filter(
    date_range: DateRange, property_name: str = "timestamp"
) -> str:
    """
    Convert DateRange to Cypher WHERE clause fragment.

    Example: WHERE m.timestamp >= datetime('2025-12-19') AND m.timestamp <= datetime('2025-12-26')
    """
    start_dt = datetime.combine(date_range.start, datetime.min.time())
    end_dt = datetime.combine(date_range.end, datetime.max.time())

    return (
        f"m.{property_name} >= datetime('{start_dt.isoformat()}') AND "
        f"m.{property_name} <= datetime('{end_dt.isoformat()}')"
    )
