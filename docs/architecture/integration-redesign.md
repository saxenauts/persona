# Integration Architecture Redesign

## Problem Statement

Current integration agent is **too expensive**: ~10 turns, ~50 tool calls, multiple LLM calls per run. Agent-based tool use workflow puts all the work on the LLM.

**Goal**: Learn from competitors (Graphiti, Honcho) to make integration fast AND high-quality.

---

## Competitor Research

### Graphiti: MinHash + LSH Fast-Path

**Architecture**: Two-tier deduplication before LLM involvement.

**Tier 1 - MinHash Similarity** (deterministic, no LLM):
- Compute MinHash signatures for entity names
- Use LSH (Locality Sensitive Hashing) for O(1) candidate lookup
- If similarity > 0.7, auto-merge without LLM
- Catches 70-80% of duplicates

**Tier 2 - LLM Resolution** (only for ambiguous cases):
- Low-similarity but semantically related entities
- Conflicting attributes
- ~20-30% of cases need LLM

**Key Insight**: "Most dedup is obvious - don't waste LLM tokens on it."

```python
# Graphiti pseudocode
async def resolve_entity(new_entity, existing_entities):
    # Fast path: MinHash similarity
    for existing in existing_entities:
        if minhash_similarity(new_entity.name, existing.name) > 0.7:
            return merge_entities(new_entity, existing)  # No LLM
    
    # Slow path: LLM for semantic resolution
    if semantic_candidates := find_semantic_similar(new_entity):
        return await llm_resolve(new_entity, semantic_candidates)
    
    return new_entity  # New entity, no merge needed
```

### Honcho: Dialectic + Deriver Separation

**Architecture**: Complete hot/cold path separation with PostgreSQL as queue.

| Component | Purpose | Execution | LLM Usage | Latency |
|-----------|---------|-----------|-----------|---------|
| **Dialectic** | Query interface | Sync API | Single call with cached context | 100-500ms |
| **Deriver** | Memory consolidation | Async worker | Heavy reasoning, batched | Minutes |

**Hot Path (Dialectic)** - `src/dialectic/chat.py`:
- Reads from **pre-materialized vector collections** (no LLM during read)
- Combines: working representation + recent conversation + peer cards + semantic search
- Returns in ~100-300ms (mostly DB/vector search time)
- **Never blocks on background processing**

**Cold Path (Deriver)** - `src/deriver/queue_manager.py`:
- PostgreSQL as both database AND queue (no Redis needed)
- Work units partition by: workspace, session, observer, observed, task_type
- Batch processing by token budget (8K tokens default)
- Two observation types: **explicit** (direct facts) + **deductive** (inferred with premises)

**Queue Pattern** (work unit isolation):
```python
# Work units prevent race conditions
work_unit_key = f"{workspace}:{session}:{observer}->{observed}:{task_type}"

# Atomic claim using INSERT...ON CONFLICT DO NOTHING
async def claim_work_units(db, work_unit_keys):
    stmt = insert(ActiveQueueSession).values([...]).on_conflict_do_nothing()
    return await db.execute(stmt)
```

**Eventual Consistency UX Mitigation**:
1. **Recency-weighted retrieval**: Documents have `message_created_at` for recency sorting
2. **Hybrid retrieval**: Vector similarity + recency + session filtering
3. **Status polling**: Apps can check `get_deriver_status()` before reads
4. **Session history fallback**: Always include recent raw messages alongside stale representation

**Key Insight**: "Embrace eventual consistency - mitigate with UX, not blocking."

---

## Current Persona Architecture (Problems)

```
User Message → Ingestion → [BLOCKING: Integration Agent] → Done
                                    ↓
                          10 turns, 50 tool calls
                          recall() + expand_neighbors() + commit_patch()
                          ~20-40s per batch
```

**Issues**:
1. **All work on LLM**: Agent decides everything (expensive)
2. **Blocking**: Can't return until integration complete
3. **No fast-path**: Every link requires LLM reasoning
4. **Too many psyche**: Creates noise in retrieval

---

## Proposed Redesign

### Phase 1: Reduce LLM Work (Graphiti-inspired)

**Add deterministic linking before LLM**:

```python
async def integrate_memory(memory: Memory):
    # 1. Fast path: Deterministic entity linking
    entities = extract_entities_from_content(memory.content)  # Regex/NER
    for entity in entities:
        if existing := await find_by_exact_name(entity.name):
            await create_link(memory.id, existing.id, "MENTIONS")
            continue
        if similar := await find_by_minhash(entity.name, threshold=0.7):
            await create_link(memory.id, similar.id, "MENTIONS")
            continue
        # Only create new entity if no match
        await create_entity(entity)
    
    # 2. Slow path: LLM only for semantic consolidation
    if needs_consolidation(memory):  # e.g., psyche type
        await queue_for_llm_consolidation(memory)
```

### Phase 2: Hot/Cold Separation (Honcho-inspired)

**Separate read path from consolidation**:

```python
# Hot path: Ingestion (fast)
async def ingest(content):
    memory = await store.create_memory(content)
    await fast_link(memory)  # Deterministic only
    return memory.id  # Return immediately

# Cold path: Background worker
async def consolidation_worker():
    while True:
        batch = await queue.get_batch(size=10)
        for memory in batch:
            await llm_consolidate(memory)  # Expensive, but background
```

### Phase 3: Reduce Psyche Noise

**Current**: Extract psyche for every session → retrieval noise
**Proposed**: 
- Episodes carry detail (self-contained)
- Psyche only for consolidated traits (rare)
- Psyche extraction threshold: Only if strongly indicative

```python
# In extraction prompt
PSYCHE_GUIDANCE = """
Create Psyche entries ONLY for:
- Strongly stated preferences ("I hate X", "I always do Y")
- Repeated patterns across multiple sessions
- Core identity traits

Do NOT create Psyche for:
- One-time mentions
- Contextual preferences
- Things better captured as Episode detail
"""
```

---

## Implementation Plan

| Phase | Change | Effort | Impact |
|-------|--------|--------|--------|
| 1a | Add MinHash similarity check before LLM | 2 days | -50% LLM calls |
| 1b | Deterministic entity linking (exact name match) | 1 day | -30% integration time |
| 2a | Queue-based background consolidation | 3 days | Non-blocking ingestion |
| 2b | Separate read/write paths | 2 days | Better latency |
| 3 | Stricter psyche extraction (prompt change) | Done | Less retrieval noise |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Integration time per batch | 20-40s | <5s (fast path) |
| LLM calls for linking | ~50 | <15 |
| Psyche entries per session | 3-5 | 0-1 |
| Ingestion latency (user-facing) | 30s | <3s |

---

## Open Questions

1. **MinHash implementation**: Use existing library (datasketch) or custom?
2. **Queue system**: PostgreSQL (like Honcho), Redis, or in-memory for MVP?
3. **Consolidation frequency**: Per-memory, per-session, or time-based batches?
4. **Entity extraction**: LLM-based or NER-based for fast path?
5. **Eventual consistency**: Status polling API or just rely on recency fallback?

---

## Key Learnings Summary

| Pattern | Graphiti | Honcho | Persona (Current) | Persona (Target) |
|---------|----------|--------|-------------------|------------------|
| Dedup fast-path | MinHash + LSH | N/A | None | MinHash |
| LLM usage | 20-30% of cases | Batched background | 100% (agent loop) | <30% |
| Queue | In-memory | PostgreSQL | None (blocking) | PostgreSQL |
| Read latency | ~1s | 100-500ms | 20-40s | <3s |
| Consistency | Strong | Eventual (mitigated) | Strong (expensive) | Eventual |

---

*Created: Jan 8, 2025*
*Updated: Jan 8, 2025 - Added Honcho research*
*Status: Research complete, implementation pending*
