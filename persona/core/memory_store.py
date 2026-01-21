"""
Memory Store for Persona.

Unified storage for all memory types (episode, psyche, note).
Handles temporal linking, retrieval, and graph operations.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID

from persona.core.interfaces import GraphDatabase, VectorStore
from persona.models.memory import (
    Memory,
    MemoryLink,
    EntityMemory,
    EntityAttribute,
    Memeplex,
    MemoryStats,
    TemporalContext,
)
from server.logging_config import get_logger

logger = get_logger(__name__)


class MemoryStore:
    """
    Unified store for all memory types.

    All memories are stored as nodes in the graph.
    Links between memories are edges.
    """

    def __init__(
        self, graph_db: GraphDatabase, vector_store: Optional[VectorStore] = None
    ):
        self.graph_db = graph_db
        self.vector_store = vector_store

    def _memory_to_node_data(self, memory: Memory) -> Dict[str, Any]:
        if not memory.day_id:
            memory.day_id = memory.event_time.strftime("%Y-%m-%d")

        node_data = memory.model_dump(mode="json", exclude={"properties"})
        node_data["name"] = str(memory.id)

        if hasattr(memory, "properties") and memory.properties:
            for k, v in memory.properties.items():
                if isinstance(v, UUID):
                    node_data[k] = str(v)
                elif isinstance(v, datetime):
                    node_data[k] = v.isoformat()
                else:
                    node_data[k] = v

        return node_data

    async def create(
        self, memory: Memory, links: Optional[List[MemoryLink]] = None
    ) -> Memory:
        """
        Create a memory and optionally link it to other memories.

        Args:
            memory: The Memory to create
            links: Optional list of links to create

        Returns:
            The created Memory
        """
        # Create the memory node with FLAT properties (not nested JSON)
        # This is backend-agnostic: each field becomes a native property
        node_data = self._memory_to_node_data(memory)

        await self.graph_db.create_nodes([node_data], memory.user_id)

        if self.vector_store and memory.embedding:
            try:
                await self.vector_store.add_embedding(
                    node_name=node_data["name"],
                    embedding=memory.embedding,
                    user_id=memory.user_id,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to persist embedding for memory {node_data['name']}: {e}"
                )

        # Create links if provided
        if links:
            for link in links:
                await self.create_link(link, memory.user_id)

        logger.info(
            f"Created {memory.type} memory '{memory.title}' for user {memory.user_id}"
        )
        return memory

    async def create_many(
        self, memories: List[Memory], links: Optional[List[MemoryLink]], user_id: str
    ) -> None:
        if not memories:
            return

        node_data = [self._memory_to_node_data(memory) for memory in memories]
        await self.graph_db.create_nodes(node_data, user_id)

        if self.vector_store:
            rows = [
                {"node_name": str(memory.id), "embedding": memory.embedding}
                for memory in memories
                if memory.embedding
            ]
            if rows:
                try:
                    await self.vector_store.add_embeddings(rows, user_id)
                except Exception as e:
                    logger.warning(f"Failed to persist embeddings for batch: {e}")

        if links:
            relationships = [
                {
                    "source": str(link.source_id),
                    "target": str(link.target_id),
                    "relation": link.relation,
                }
                for link in links
            ]
            await self.graph_db.create_relationships(relationships, user_id)

    async def create_link(self, link: MemoryLink, user_id: str) -> None:
        """Create a link between two memories."""
        relationship = {
            "source": str(link.source_id),
            "target": str(link.target_id),
            "relation": link.relation,
            **link.properties,
        }
        await self.graph_db.create_relationships([relationship], user_id)

    async def get(self, memory_id: UUID, user_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        node_data = await self.graph_db.get_node(str(memory_id), user_id)

        if not node_data:
            return None

        return self._node_to_memory(node_data, user_id)

    async def get_memories_by_ids(
        self, memory_ids: List[UUID], user_id: str
    ) -> List[Memory]:
        """Batch retrieve multiple memories by ID."""
        if not memory_ids:
            return []

        id_strs = [str(mid) for mid in memory_ids]
        nodes = await self.graph_db.get_nodes_by_ids(id_strs, user_id)
        return [self._node_to_memory(n, user_id) for n in nodes]

    async def get_by_type(
        self, memory_type: str, user_id: str, limit: int = 50
    ) -> List[Memory]:
        """Get all memories of a specific type."""
        all_nodes = await self.graph_db.get_all_nodes(user_id)

        memories = [
            self._node_to_memory(n, user_id)
            for n in all_nodes
            if n.get("type") == memory_type
        ]

        # Sort by timestamp descending (most recent first)
        memories.sort(key=lambda m: m.event_time, reverse=True)
        return memories[:limit]

    async def get_by_day(self, day_id: str, user_id: str) -> List[Memory]:
        """Get all memories for a specific day."""
        all_nodes = await self.graph_db.get_all_nodes(user_id)

        memories = []
        for node in all_nodes:
            props = node.get("properties", {})
            if props.get("day_id") == day_id:
                memories.append(self._node_to_memory(node, user_id))

        memories.sort(key=lambda m: m.event_time)
        return memories

    async def get_recent(
        self, user_id: str, memory_type: Optional[str] = None, limit: int = 20
    ) -> List[Memory]:
        """Get recent memories, optionally filtered by type."""
        all_nodes = await self.graph_db.get_all_nodes(user_id)

        memories = []
        for node in all_nodes:
            if memory_type and node.get("type") != memory_type:
                continue
            memories.append(self._node_to_memory(node, user_id))

        memories.sort(key=lambda m: m.event_time, reverse=True)
        return memories[:limit]

    async def get_most_recent_episode(self, user_id: str) -> Optional[Memory]:
        episodes = await self.get_by_type("episode", user_id, limit=1)
        return episodes[0] if episodes else None

    async def get_temporal_predecessor(
        self, user_id: str, target_time: datetime
    ) -> Optional[Memory]:
        """Find episode with event_time closest to but before target_time."""
        all_nodes = await self.graph_db.get_all_nodes(user_id)

        candidates = []
        for node in all_nodes:
            if node.get("type") != "episode":
                continue
            mem = self._node_to_memory(node, user_id)
            if mem.event_time and mem.event_time < target_time:
                candidates.append(mem)

        if not candidates:
            return None

        candidates.sort(key=lambda m: m.event_time, reverse=True)
        return candidates[0]

    async def link_temporal_chain(
        self, new_memory: Memory, previous_memory: Memory
    ) -> None:
        """Create PREVIOUS/NEXT links between episodes."""
        # New → Previous
        await self.create_link(
            MemoryLink(
                source_id=new_memory.id,
                target_id=previous_memory.id,
                relation="PREVIOUS",
            ),
            new_memory.user_id,
        )
        # Previous → New
        await self.create_link(
            MemoryLink(
                source_id=previous_memory.id, target_id=new_memory.id, relation="NEXT"
            ),
            previous_memory.user_id,
        )

    def _node_to_memory(self, node: Dict[str, Any], user_id: str) -> Memory:
        """Convert a graph node to the correct polymorphic Memory model."""
        from pydantic import TypeAdapter, ValidationError
        import json

        props = node.get("properties", {})
        # Handle flat properties (new format) vs nested properties (old format)
        if not props and "title" in node:
            props = node

        # Deserialization: Parse JSON strings if any
        # This handles nested dicts/lists that were JSON-serialized
        processed_props = {}
        for k, v in props.items():
            if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                try:
                    processed_props[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    processed_props[k] = v
            else:
                processed_props[k] = v

        # Ensure 'id' is a valid UUID string
        if "id" not in processed_props and "name" in node:
            processed_props["id"] = node["name"]

        # Ensure user_id is set
        processed_props["user_id"] = user_id

        try:
            # Pydantic's TypeAdapter handles the discriminated union automatically!
            # It looks at the 'type' field and instantiates Episode/Psyche/Goal accordingly.
            return TypeAdapter(Memory).validate_python(processed_props)
        except ValidationError as e:
            logger.error(
                f"Failed to reconstruct memory {processed_props.get('id')}: {e}"
            )
            # Fallback to generic Memory-like structure if validation fails
            # This prevents crashing on bad data
            from persona.models.memory import EpisodeMemory

            return EpisodeMemory(
                **{
                    k: v
                    for k, v in processed_props.items()
                    if k in EpisodeMemory.model_fields
                },
                type="episode",  # Force valid type
                user_id=user_id,
            )

    # ========== Search Methods ==========

    async def search_text(
        self,
        user_id: str,
        query: str,
        types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """
        Keyword-based text search on title and content.

        Args:
            user_id: User ID
            query: Search query string
            types: Filter by memory types (episode, psyche, note)
            limit: Maximum results
        """
        all_nodes = await self.graph_db.get_all_nodes(user_id)
        query_lower = query.lower()

        matches = []
        for node in all_nodes:
            if types and node.get("type") not in types:
                continue

            # Check title and content for query match
            props = node.get("properties", node)
            title = str(props.get("title", "")).lower()
            content = str(props.get("content", "")).lower()

            if query_lower in title or query_lower in content:
                matches.append(self._node_to_memory(node, user_id))

        return matches[:limit]

    async def search_vector(
        self,
        user_id: str,
        query: str,
        types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Memory]:
        if not self.vector_store:
            logger.warning("No vector_store configured for search_vector")
            return []

        from persona.llm.embeddings import generate_embeddings_async

        query_embeddings = await generate_embeddings_async([query])
        if not query_embeddings or not query_embeddings[0]:
            logger.warning(f"Failed to generate embedding for query: {query}")
            return []

        results = await self.vector_store.search_similar(
            embedding=query_embeddings[0],
            user_id=user_id,
            limit=limit * 2,
        )

        memories = []
        for result in results:
            node = await self.graph_db.get_node(result["node_name"], user_id)
            if node:
                if types and node.get("type") not in types:
                    continue
                memories.append(self._node_to_memory(node, user_id))
                if len(memories) >= limit:
                    break

        return memories

    async def get_connected_batch(
        self, memory_ids: List[UUID], user_id: str, relation: Optional[str] = None
    ) -> Dict[UUID, List[tuple]]:
        """
        Get all relationships for multiple memories in a single query.

        Returns a dict mapping source memory ID to list of (target_id, relation) tuples.
        """
        if not memory_ids:
            return {}

        id_strs = [str(mid) for mid in memory_ids]
        id_set = set(id_strs)
        relationships = await self.graph_db.get_relationships_for_nodes(
            id_strs, user_id
        )

        result: Dict[UUID, List[tuple]] = {mid: [] for mid in memory_ids}
        for rel in relationships:
            source_node = rel.get("source_node")
            if source_node not in id_set:
                continue

            rel_type = rel.get("relation", "related_to")
            if relation and rel_type != relation:
                continue

            target = (
                rel.get("target")
                if rel.get("source") == source_node
                else rel.get("source")
            )

            source_uuid = UUID(source_node)
            if target:
                result[source_uuid].append((UUID(target), rel_type))

        return result

    async def get_note_hierarchy(
        self, user_id: str, root_id: Optional[UUID] = None
    ) -> List[Memory]:
        """
        Get notes and their subtasks in hierarchy.

        If root_id is provided, return that note and its children.
        Otherwise return all notes.
        """
        notes = await self.get_by_type("note", user_id, limit=100)

        if root_id:
            connections = await self.get_connected_batch(
                [root_id], user_id, relation="PARENT_OF"
            )
            child_ids = {target_id for target_id, _ in connections.get(root_id, [])}
            return [n for n in notes if n.id == root_id or n.id in child_ids]

        return notes

    # ========== Mutation Methods ==========

    async def update(
        self, memory_id: UUID, user_id: str, updates: Dict[str, Any]
    ) -> Optional[Memory]:
        """
        Update mutable fields of a memory.

        Args:
            memory_id: Memory ID to update
            user_id: User ID
            updates: Dict of field:value pairs to update

        Allowed fields: title, content, status, properties
        """
        # Get existing memory
        existing = await self.get(memory_id, user_id)
        if not existing:
            logger.warning(f"Memory {memory_id} not found for update")
            return None

        # Build update node
        node_data = {
            "name": str(memory_id),
            "type": existing.type,
        }

        # Add allowed updates
        allowed_fields = {"title", "content", "status", "due_date"}
        for field, value in updates.items():
            if field in allowed_fields:
                node_data[field] = value

        # Apply update
        await self.graph_db.create_nodes([node_data], user_id)

        logger.info(f"Updated memory {memory_id}: {list(updates.keys())}")
        return await self.get(memory_id, user_id)

    # ========== Entity-specific Methods ==========

    async def create_entity(self, entity: EntityMemory) -> EntityMemory:
        node_data = self._memory_to_node_data(entity)

        import json

        if entity.attributes:
            node_data["attributes"] = json.dumps(
                [attr.model_dump(mode="json") for attr in entity.attributes]
            )
        if entity.relationships:
            node_data["relationships"] = json.dumps(
                [rel.model_dump(mode="json") for rel in entity.relationships]
            )
        if entity.aliases:
            node_data["aliases"] = json.dumps(entity.aliases)
        if entity.mentioned_in:
            node_data["mentioned_in"] = json.dumps(
                [str(mid) for mid in entity.mentioned_in]
            )

        await self.graph_db.create_nodes([node_data], entity.user_id)

        if self.vector_store and entity.embedding:
            try:
                await self.vector_store.add_embedding(
                    node_name=node_data["name"],
                    embedding=entity.embedding,
                    user_id=entity.user_id,
                )
            except Exception as e:
                logger.warning(f"Failed to persist entity embedding: {e}")

        logger.info(
            f"Created entity '{entity.canonical_name}' ({entity.entity_type}) "
            f"for user {entity.user_id}"
        )
        return entity

    async def get_entity_by_name(
        self, name: str, user_id: str, include_aliases: bool = True
    ) -> Optional[EntityMemory]:
        all_nodes = await self.graph_db.get_all_nodes(user_id)
        name_lower = name.lower()

        for node in all_nodes:
            if node.get("type") != "entity":
                continue

            canonical = str(node.get("canonical_name", "")).lower()
            if canonical == name_lower:
                return self._node_to_memory(node, user_id)  # type: ignore

            if include_aliases:
                import json

                aliases_raw = node.get("aliases", "[]")
                if isinstance(aliases_raw, str):
                    try:
                        aliases = json.loads(aliases_raw)
                    except json.JSONDecodeError:
                        aliases = []
                else:
                    aliases = aliases_raw

                if any(alias.lower() == name_lower for alias in aliases):
                    return self._node_to_memory(node, user_id)  # type: ignore

        return None

    async def upsert_entity_attribute(
        self,
        entity_id: UUID,
        user_id: str,
        key: str,
        value: str,
        evidence_id: Optional[UUID] = None,
        confidence: float = 1.0,
        event_time: Optional[datetime] = None,
    ) -> Optional[EntityMemory]:
        """Append versioned attribute (preserves history for temporal queries)."""
        entity = await self.get(entity_id, user_id)
        if not entity or entity.type != "entity":
            logger.warning(f"Entity {entity_id} not found for attribute upsert")
            return None

        entity_mem: EntityMemory = entity  # type: ignore

        new_attr = EntityAttribute(
            key=key,
            value=value,
            confidence=confidence,
            evidence_id=evidence_id,
            updated_at=event_time or datetime.utcnow(),
        )

        entity_mem.attributes.append(new_attr)

        import json

        await self.graph_db.create_nodes(
            [
                {
                    "name": str(entity_id),
                    "type": "entity",
                    "attributes": json.dumps(
                        [attr.model_dump(mode="json") for attr in entity_mem.attributes]
                    ),
                }
            ],
            user_id,
        )

        logger.info(f"Added versioned attribute '{key}' on entity {entity_id}")
        return await self.get(entity_id, user_id)  # type: ignore

    def get_current_attribute(
        self, entity: EntityMemory, key: str
    ) -> Optional[EntityAttribute]:
        """Get most recent value for an attribute key."""
        matching = [a for a in entity.attributes if a.key == key]
        if not matching:
            return None
        return max(matching, key=lambda a: a.updated_at)

    def get_attribute_as_of(
        self, entity: EntityMemory, key: str, as_of: datetime
    ) -> Optional[EntityAttribute]:
        """Get attribute value as of a specific time."""
        matching = [
            a for a in entity.attributes if a.key == key and a.updated_at <= as_of
        ]
        if not matching:
            return None
        return max(matching, key=lambda a: a.updated_at)

    async def link_memory_to_entity(
        self, memory_id: UUID, entity_id: UUID, user_id: str
    ) -> None:
        entity = await self.get(entity_id, user_id)
        if not entity or entity.type != "entity":
            logger.warning(f"Entity {entity_id} not found for linking")
            return

        entity_mem: EntityMemory = entity  # type: ignore
        if memory_id not in entity_mem.mentioned_in:
            entity_mem.mentioned_in.append(memory_id)

            import json

            await self.graph_db.create_nodes(
                [
                    {
                        "name": str(entity_id),
                        "type": "entity",
                        "mentioned_in": json.dumps(
                            [str(mid) for mid in entity_mem.mentioned_in]
                        ),
                    }
                ],
                user_id,
            )

        await self.create_link(
            MemoryLink(source_id=memory_id, target_id=entity_id, relation="MENTIONS"),
            user_id,
        )
        logger.debug(f"Linked memory {memory_id} to entity {entity_id}")

    async def get_entities_by_type(
        self, entity_type: str, user_id: str, limit: int = 50
    ) -> List[EntityMemory]:
        all_nodes = await self.graph_db.get_all_nodes(user_id)

        entities = []
        for node in all_nodes:
            if node.get("type") != "entity":
                continue
            if node.get("entity_type") == entity_type:
                entities.append(self._node_to_memory(node, user_id))

        entities.sort(key=lambda e: e.event_time, reverse=True)
        return entities[:limit]  # type: ignore

    # ========== Memeplex Methods ==========

    async def get_memeplex(self, user_id: str) -> Optional[Memeplex]:
        all_nodes = await self.graph_db.get_all_nodes(user_id)

        memeplex_node = None
        temporal_node = None

        for node in all_nodes:
            if node.get("type") == "memeplex":
                memeplex_node = node
            elif node.get("type") == "temporal_context":
                temporal_node = node

        if memeplex_node:
            memeplex = self._node_to_memeplex(memeplex_node, user_id)
            if temporal_node:
                memeplex.temporal_context = self._node_to_temporal_context(
                    temporal_node
                )
            return memeplex

        return None

    async def save_memeplex(self, memeplex: Memeplex) -> Memeplex:
        import json

        memeplex.updated_at = datetime.utcnow()

        node_data = {
            "name": f"memeplex_{memeplex.user_id}",
            "type": "memeplex",
            "user_id": memeplex.user_id,
            "updated_at": memeplex.updated_at.isoformat(),
            "topics": json.dumps(memeplex.topics),
            "people": json.dumps(memeplex.people),
            "projects": json.dumps(memeplex.projects),
            "places": json.dumps(memeplex.places),
            "concepts": json.dumps(memeplex.concepts),
            "last_week_topics": json.dumps(memeplex.last_week_topics),
            "last_month_topics": json.dumps(memeplex.last_month_topics),
            "recent_focus": memeplex.recent_focus,
            "memory_stats": json.dumps(memeplex.memory_stats.model_dump()),
            "timeline_summary": memeplex.timeline_summary,
        }

        await self.graph_db.create_nodes([node_data], memeplex.user_id)
        logger.info(f"Saved memeplex for user {memeplex.user_id}")
        return memeplex

    async def get_or_create_memeplex(self, user_id: str) -> Memeplex:
        existing = await self.get_memeplex(user_id)
        if existing:
            return existing

        memeplex = Memeplex(user_id=user_id)
        return await self.save_memeplex(memeplex)

    async def compute_memory_stats(self, user_id: str) -> MemoryStats:
        all_nodes = await self.graph_db.get_all_nodes(user_id)

        stats = MemoryStats()
        timestamps: List[datetime] = []
        session_ids: set = set()

        for node in all_nodes:
            node_type = node.get("type")
            if node_type == "memeplex":
                continue

            stats.total_memories += 1

            if node_type == "episode":
                stats.total_episodes += 1
            elif node_type == "psyche":
                stats.total_psyche += 1
            elif node_type == "note":
                stats.total_notes += 1
                if node.get("status") == "active":
                    stats.active_notes += 1
            elif node_type == "entity":
                stats.total_entities += 1

            ts_str = node.get("event_time")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    timestamps.append(ts)
                except (ValueError, TypeError):
                    pass

            session_id = node.get("session_id")
            if session_id:
                session_ids.add(session_id)

        if timestamps:
            stats.earliest_memory = min(timestamps)
            stats.latest_memory = max(timestamps)

        stats.session_count = len(session_ids)
        return stats

    def _node_to_memeplex(self, node: Dict[str, Any], user_id: str) -> Memeplex:
        import json

        props = node.get("properties", {})
        if not props and "type" in node:
            props = node

        def parse_json_list(key: str) -> List[str]:
            raw = props.get(key, "[]")
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return []
            return raw if raw else []

        memory_stats_raw = props.get("memory_stats", "{}")
        if isinstance(memory_stats_raw, str):
            try:
                memory_stats_data = json.loads(memory_stats_raw)
            except json.JSONDecodeError:
                memory_stats_data = {}
        else:
            memory_stats_data = memory_stats_raw

        updated_at_str = props.get("updated_at")
        if updated_at_str:
            try:
                updated_at = datetime.fromisoformat(
                    updated_at_str.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                updated_at = datetime.utcnow()
        else:
            updated_at = datetime.utcnow()

        return Memeplex(
            user_id=user_id,
            updated_at=updated_at,
            topics=parse_json_list("topics"),
            people=parse_json_list("people"),
            projects=parse_json_list("projects"),
            places=parse_json_list("places"),
            concepts=parse_json_list("concepts"),
            last_week_topics=parse_json_list("last_week_topics"),
            last_month_topics=parse_json_list("last_month_topics"),
            recent_focus=props.get("recent_focus", ""),
            memory_stats=MemoryStats(**memory_stats_data),
            timeline_summary=props.get("timeline_summary", ""),
        )

    def _node_to_temporal_context(self, node: Dict[str, Any]) -> TemporalContext:
        import json

        props = node.get("properties", {})
        if not props and "type" in node:
            props = node

        upcoming_raw = props.get("upcoming", "[]")
        if isinstance(upcoming_raw, str):
            try:
                upcoming = json.loads(upcoming_raw)
            except json.JSONDecodeError:
                upcoming = []
        else:
            upcoming = upcoming_raw if upcoming_raw else []

        return TemporalContext(
            current_date=props.get("current_date", ""),
            week_summary=props.get("week_summary", ""),
            week_start=props.get("week_start", ""),
            month_summary=props.get("month_summary", ""),
            month_name=props.get("month_name", ""),
            upcoming=upcoming,
        )
