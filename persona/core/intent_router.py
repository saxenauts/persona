"""
Intent Router: Routes queries using UserCard fuzzy index.

FAST mode: Single-pass, deterministic routing using UserCard hints + embedding similarity.
SLOW mode: Agentic loop with LLM tool calls for complex queries.

The router takes query + UserCard and outputs RetrievalHints that guide downstream retrieval.
"""

import re
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from persona.models.memory import UserCard
from server.logging_config import get_logger

logger = get_logger(__name__)


class RetrievalMode(Enum):
    FAST = "fast"
    SLOW = "slow"


class RetrievalHints(BaseModel):
    """Output of IntentRouter - guides retrieval behavior."""

    mode: RetrievalMode = RetrievalMode.FAST

    search_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords extracted from query + UserCard aliases for spray-and-pray",
    )
    seed_memory_ids: List[str] = Field(
        default_factory=list,
        description="Memory IDs from UserCard hints to use as retrieval seeds",
    )

    memory_type_boost: List[str] = Field(
        default_factory=list,
        description="Memory types to prioritize: ['note', 'psyche', 'episode']",
    )
    link_type_boost: List[str] = Field(
        default_factory=list,
        description="Link types to prioritize in graph traversal",
    )

    date_range: Optional[Tuple[date, date]] = Field(
        default=None,
        description="Date filter resolved from temporal anchors or query",
    )

    top_k: int = Field(default=5, description="Number of results per search lane")
    hop_depth: int = Field(default=1, description="Graph traversal depth")

    resolved_query: str = Field(
        default="",
        description="Query with aliases resolved from UserCard",
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Router confidence in this plan",
    )


class IntentRouter:
    """Routes queries using UserCard fuzzy index for spray-and-pray retrieval."""

    ESCALATION_MARGIN = 0.15

    TEMPORAL_PATTERNS = {
        r"\byesterday\b": lambda d: (d - timedelta(days=1), d - timedelta(days=1)),
        r"\btoday\b": lambda d: (d, d),
        r"\blast week\b": lambda d: (d - timedelta(days=7), d),
        r"\bpast week\b": lambda d: (d - timedelta(days=7), d),
        r"\blast month\b": lambda d: (d - timedelta(days=30), d),
        r"\bthis week\b": lambda d: (d - timedelta(days=d.weekday()), d),
        r"\brecently\b": lambda d: (d - timedelta(days=7), d),
    }

    def __init__(self, embedding_client=None):
        self.embedding_client = embedding_client

    async def route(
        self,
        query: str,
        user_card: Optional[UserCard] = None,
        current_date: Optional[date] = None,
    ) -> RetrievalHints:
        """Route query using UserCard fuzzy index."""
        current_date = current_date or date.today()
        user_card = user_card or UserCard(user_id="unknown")

        resolved_query = self._resolve_aliases(query, user_card.entity_aliases)
        keywords = self._extract_keywords(resolved_query, user_card)
        seed_ids = self._find_seed_memories(keywords, user_card)
        date_range = self._resolve_date_range(query, user_card, current_date)
        memory_boosts = self._compute_memory_boosts(query, user_card)
        link_boosts = self._compute_link_boosts(user_card)
        mode, confidence = self._determine_mode(query, keywords, seed_ids, user_card)
        top_k, hop_depth = self._compute_retrieval_params(mode, confidence)

        hints = RetrievalHints(
            mode=mode,
            search_keywords=keywords,
            seed_memory_ids=seed_ids,
            memory_type_boost=memory_boosts,
            link_type_boost=link_boosts,
            date_range=date_range,
            top_k=top_k,
            hop_depth=hop_depth,
            resolved_query=resolved_query,
            confidence=confidence,
        )

        logger.debug(
            f"IntentRouter: mode={mode.value}, keywords={keywords[:5]}, "
            f"seeds={len(seed_ids)}, confidence={confidence:.2f}"
        )

        return hints

    def _resolve_aliases(self, query: str, aliases: Dict[str, str]) -> str:
        """Replace known aliases with canonical names."""
        resolved = query
        for alias, canonical in aliases.items():
            pattern = re.compile(re.escape(alias), re.IGNORECASE)
            resolved = pattern.sub(canonical, resolved)
        return resolved

    def _extract_keywords(self, query: str, user_card: UserCard) -> List[str]:
        """Extract keywords from query that match UserCard hints."""
        keywords = []
        query_lower = query.lower()
        query_words = set(re.findall(r"\b\w+\b", query_lower))

        for keyword in user_card.keyword_hints.keys():
            if keyword.lower() in query_lower:
                keywords.append(keyword)

        for tag in user_card.pinned_memories.keys():
            tag_words = set(tag.lower().split("_"))
            if tag_words & query_words:
                keywords.append(tag)

        for rel in user_card.key_relationships:
            rel_words = rel.lower().split()
            if any(w in query_lower for w in rel_words if len(w) > 2):
                keywords.append(rel)

        for focus in user_card.current_focus:
            focus_words = set(focus.lower().split())
            if focus_words & query_words:
                keywords.append(focus)

        significant_words = [
            w for w in query_words if len(w) > 3 and w not in self._stop_words()
        ]
        keywords.extend(significant_words[:5])

        return list(dict.fromkeys(keywords))[:10]

    def _find_seed_memories(
        self, keywords: List[str], user_card: UserCard
    ) -> List[str]:
        """Find seed memory IDs from UserCard hints matching keywords."""
        seeds = []

        for keyword in keywords:
            if keyword in user_card.keyword_hints:
                seeds.extend(user_card.keyword_hints[keyword])

            if keyword in user_card.pinned_memories:
                seeds.extend(user_card.pinned_memories[keyword])

        return list(dict.fromkeys(seeds))[:20]

    def _resolve_date_range(
        self,
        query: str,
        user_card: UserCard,
        current_date: date,
    ) -> Optional[Tuple[date, date]]:
        """Resolve date range from query or UserCard temporal anchors."""
        query_lower = query.lower()

        for pattern, resolver in self.TEMPORAL_PATTERNS.items():
            if re.search(pattern, query_lower):
                return resolver(current_date)

        for anchor_name, anchor_date_str in user_card.temporal_anchors.items():
            if anchor_name.lower() in query_lower:
                try:
                    if ":" in anchor_date_str:
                        start_str, end_str = anchor_date_str.split(":")
                        start = datetime.strptime(start_str, "%Y-%m-%d").date()
                        end = datetime.strptime(end_str, "%Y-%m-%d").date()
                        return (start, end)
                    else:
                        anchor_date = datetime.strptime(
                            anchor_date_str, "%Y-%m-%d"
                        ).date()
                        return (
                            anchor_date - timedelta(days=7),
                            anchor_date + timedelta(days=7),
                        )
                except ValueError:
                    continue

        return None

    def _compute_memory_boosts(self, query: str, user_card: UserCard) -> List[str]:
        """Determine which memory types to boost based on query and user distribution."""
        boosts = []
        query_lower = query.lower()

        task_signals = ["task", "todo", "should i", "what's next", "priorities"]
        if any(s in query_lower for s in task_signals):
            boosts.append("note")

        identity_signals = ["who am i", "my values", "what do i like", "preference"]
        if any(s in query_lower for s in identity_signals):
            boosts.append("psyche")

        event_signals = ["happened", "when did", "remember when", "that time"]
        if any(s in query_lower for s in event_signals):
            boosts.append("episode")

        if not boosts and user_card.dominant_memory_types:
            sorted_types = sorted(
                user_card.dominant_memory_types.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            boosts = [t for t, _ in sorted_types[:2]]

        return boosts

    def _compute_link_boosts(self, user_card: UserCard) -> List[str]:
        """Determine which link types to prioritize in graph traversal."""
        if not user_card.dominant_link_types:
            return []

        sorted_links = sorted(
            user_card.dominant_link_types.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [lt for lt, _ in sorted_links[:3]]

    def _determine_mode(
        self,
        query: str,
        keywords: List[str],
        seed_ids: List[str],
        user_card: UserCard,
    ) -> Tuple[RetrievalMode, float]:
        """Determine FAST vs SLOW mode based on query complexity and available hints."""
        confidence = 0.5

        if seed_ids:
            confidence += 0.2
        if keywords:
            confidence += min(0.15, len(keywords) * 0.03)

        complex_signals = [
            "compare",
            "difference between",
            "how did",
            "why did",
            "changed",
            "over time",
            "summarize",
            "all the",
        ]
        query_lower = query.lower()
        if any(s in query_lower for s in complex_signals):
            confidence -= 0.25

        if len(query.split()) > 20:
            confidence -= 0.15

        if "?" in query and query.count("?") > 1:
            confidence -= 0.1

        confidence = max(0.0, min(1.0, confidence))

        if confidence < (0.5 - self.ESCALATION_MARGIN):
            return RetrievalMode.SLOW, confidence
        return RetrievalMode.FAST, confidence

    def _compute_retrieval_params(
        self, mode: RetrievalMode, confidence: float
    ) -> Tuple[int, int]:
        """Compute top_k and hop_depth based on mode and confidence."""
        if mode == RetrievalMode.FAST:
            top_k = 5 if confidence > 0.6 else 7
            hop_depth = 1
        else:
            top_k = 10
            hop_depth = 2

        return top_k, hop_depth

    def _stop_words(self) -> set:
        return {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "it",
            "its",
            "they",
            "them",
            "their",
            "about",
            "tell",
            "know",
        }
