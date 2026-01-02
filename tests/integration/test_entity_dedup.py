import pytest
from datetime import datetime
from uuid import uuid4

from persona.core.memory_store import MemoryStore
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.models.memory import EntityMemory, EntityAttribute


@pytest.mark.asyncio
async def test_upsert_entity_creates_new():
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    store = MemoryStore(graph_db)

    user_id = f"test-entity-new-{uuid4()}"

    try:
        await graph_db.create_user(user_id)

        entity = EntityMemory(
            id=uuid4(),
            entity_type="person",
            canonical_name="Sarah",
            title="Sarah",
            content="User's sister",
            aliases=["Sarah Chen"],
            attributes=[
                EntityAttribute(key="relationship", value="sister"),
            ],
            user_id=user_id,
            timestamp=datetime.utcnow(),
        )

        result, is_new = await store.upsert_entity(entity)

        assert is_new is True
        assert result.id == entity.id
        assert result.canonical_name == "Sarah"

        retrieved = await store.get_entity_by_name("Sarah", user_id)
        assert retrieved is not None
        assert retrieved.canonical_name == "Sarah"

        print(f"\n✅ upsert_entity creates new entity correctly")

    finally:
        await graph_db.delete_user(user_id)
        await graph_db.close()


@pytest.mark.asyncio
async def test_upsert_entity_merges_existing():
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    store = MemoryStore(graph_db)

    user_id = f"test-entity-merge-{uuid4()}"
    episode_1 = uuid4()
    episode_2 = uuid4()

    try:
        await graph_db.create_user(user_id)

        original = EntityMemory(
            id=uuid4(),
            entity_type="person",
            canonical_name="Sarah",
            title="Sarah",
            content="User's sister",
            aliases=["Sarah Chen"],
            attributes=[
                EntityAttribute(key="relationship", value="sister"),
                EntityAttribute(key="city", value="NYC"),
            ],
            mentioned_in=[episode_1],
            user_id=user_id,
            timestamp=datetime.utcnow(),
        )
        await store.create_entity(original)

        duplicate = EntityMemory(
            id=uuid4(),
            entity_type="person",
            canonical_name="Sarah",
            title="Sarah",
            content="User's sister getting married",
            aliases=["Sarah C"],
            attributes=[
                EntityAttribute(key="relationship", value="sibling"),
                EntityAttribute(key="wedding_date", value="June 2025"),
            ],
            mentioned_in=[],
            user_id=user_id,
            timestamp=datetime.utcnow(),
        )

        result, is_new = await store.upsert_entity(duplicate, episode_id=episode_2)

        assert is_new is False
        assert result.id == original.id

        merged = await store.get_entity_by_name("Sarah", user_id)
        assert merged is not None
        assert merged.id == original.id

        alias_set = set(merged.aliases)
        assert "Sarah Chen" in alias_set
        assert "Sarah C" in alias_set

        attr_keys = {a.key for a in merged.attributes}
        assert "relationship" in attr_keys
        assert "city" in attr_keys
        assert "wedding_date" in attr_keys

        attr_map = {a.key: a.value for a in merged.attributes}
        assert attr_map["relationship"] == "sister"
        assert attr_map["city"] == "NYC"
        assert attr_map["wedding_date"] == "June 2025"

        assert episode_1 in merged.mentioned_in
        assert episode_2 in merged.mentioned_in

        print(f"\n✅ upsert_entity merges duplicates correctly")
        print(f"   Aliases: {merged.aliases}")
        print(f"   Attributes: {attr_map}")
        print(f"   Mentioned in: {len(merged.mentioned_in)} episodes")

    finally:
        await graph_db.delete_user(user_id)
        await graph_db.close()


@pytest.mark.asyncio
async def test_upsert_entity_finds_by_alias():
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    store = MemoryStore(graph_db)

    user_id = f"test-entity-alias-{uuid4()}"

    try:
        await graph_db.create_user(user_id)

        original = EntityMemory(
            id=uuid4(),
            entity_type="person",
            canonical_name="Sarah Chen",
            title="Sarah Chen",
            content="User's sister",
            aliases=["Sarah", "Sar"],
            attributes=[],
            user_id=user_id,
            timestamp=datetime.utcnow(),
        )
        await store.create_entity(original)

        by_alias = EntityMemory(
            id=uuid4(),
            entity_type="person",
            canonical_name="Sarah",
            title="Sarah",
            content="Sister mentioned again",
            aliases=[],
            attributes=[EntityAttribute(key="hobby", value="photography")],
            user_id=user_id,
            timestamp=datetime.utcnow(),
        )

        result, is_new = await store.upsert_entity(by_alias)

        assert is_new is False
        assert result.id == original.id

        merged = await store.get_entity_by_name("Sarah Chen", user_id)
        assert merged is not None
        attr_keys = {a.key for a in merged.attributes}
        assert "hobby" in attr_keys

        print(f"\n✅ upsert_entity finds existing by alias")

    finally:
        await graph_db.delete_user(user_id)
        await graph_db.close()
