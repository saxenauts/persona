# Honcho vs Persona: Speed Analysis

## Architecture Comparison

| Aspect | Honcho | Persona |
|--------|--------|---------|
| **Storage** | PostgreSQL + pgvector | Neo4j + vector index |
| **Processing** | Async queue (deriver worker) | Sync per-request |
| **API Response** | Immediate (persist only) | Waits for extraction |
| **Chunking** | 25K chars/message, 100 msg/batch | 16K chars/chunk (eval-side) |
| **Token Batching** | 4K-16K token threshold | None (process immediately) |
| **Parallelism** | Multi-worker queue + 5 concurrent questions | Semaphore(5) for extraction |

---

## Honcho Speed Secrets

### 1. Two-Phase Architecture (KEY INSIGHT)
```
API Request → DB Write (fast) → Return immediately
                    ↓
            Background Queue → Deriver Worker → LLM Extraction
```

**Persona does**:
```
API Request → LLM Extraction (slow) → DB Write → Return
```

**Impact**: Honcho's API is ~100x faster because extraction is async.

### 2. Token-Threshold Batching
Honcho accumulates messages until 4K-16K tokens, then makes ONE LLM call.

```python
# Honcho: Wait for threshold before processing
if total_tokens >= batch_max_tokens:
    process_batch()
```

**Persona**: Makes one LLM call per chunk immediately.

**Impact**: Honcho makes fewer LLM calls, each more efficient.

### 3. Multi-Instance Load Balancing (BEAM Eval)
```python
# Round-robin across multiple Honcho instances
instance_id = conversation_index % pool_size
port = base_api_port + instance_id
```

**Impact**: 4 instances = ~4x throughput for independent conversations.

---

## Current Persona Bottlenecks

### Measured Times (from BEAM eval logs)
| Phase | Time | % of Total |
|-------|------|------------|
| Ingestion (45 chunks × LLM) | ~140-260s | **90%** |
| Query (recall + response) | ~15-20s | ~10% |
| **Total per question** | ~160-280s | 100% |

### Breakdown of Ingestion
1. **LLM Extraction**: ~3s per chunk × 45 chunks = ~135s
2. **Embedding Generation**: ~0.5s per memory × 230 memories = ~115s (batched?)
3. **Neo4j Writes**: ~0.1s per memory = ~23s
4. **Temporal Linking**: ~5s

### Primary Bottleneck: Sequential Chunk Processing
Each 16K chunk makes a separate LLM call. With 45 chunks, that's 45 LLM calls.

---

## Optimization Opportunities

### 1. Parallel Chunk Extraction (QUICK WIN)
**Current**: Chunks processed sequentially within a session
**Fix**: Process chunks in parallel with semaphore

```python
# Current (sequential)
for chunk in chunks:
    result = await ingest(chunk)

# Proposed (parallel)
sem = asyncio.Semaphore(10)
async def process_chunk(chunk):
    async with sem:
        return await ingest(chunk)

results = await asyncio.gather(*[process_chunk(c) for c in chunks])
```

**Expected Speedup**: 5-10x on ingestion (45 sequential → 45/10 = 4.5 batches)
**Time Saved**: 140s → ~20-30s

### 2. Batch Embedding Generation
**Current**: Unknown if batched
**Fix**: Batch all embeddings in single API call

```python
# Instead of
for mem in memories:
    mem.embedding = await embed(mem.content)

# Do
texts = [m.content for m in memories]
embeddings = await embed_batch(texts)  # Single API call
```

**Expected Speedup**: ~5x on embedding phase

### 3. Async API Response (Honcho Pattern)
**Current**: API waits for full extraction
**Fix**: Persist content immediately, extract in background

```python
async def ingest(content):
    # Phase 1: Quick persist
    raw_id = await store.save_raw(content)
    
    # Phase 2: Queue for background extraction
    await queue.enqueue({"raw_id": raw_id, "content": content})
    
    # Return immediately
    return {"status": "queued", "raw_id": raw_id}
```

**Expected Speedup**: API response ~100x faster
**Tradeoff**: Memories not immediately available

### 4. Token-Threshold Batching
**Current**: Process each chunk separately
**Fix**: Accumulate chunks until token threshold, then single LLM call

```python
# Accumulate until 8K tokens
batch_tokens = 0
batch_content = []

for chunk in chunks:
    chunk_tokens = len(chunk) // 4
    if batch_tokens + chunk_tokens > 8192:
        # Process batch
        await process_batch(batch_content)
        batch_content = [chunk]
        batch_tokens = chunk_tokens
    else:
        batch_content.append(chunk)
        batch_tokens += chunk_tokens
```

**Expected Speedup**: ~4x fewer LLM calls

---

## Recommended Implementation Order

### Phase 1: Parallel Chunk Ingestion (1-2 hours)
Modify eval adapter to process chunks in parallel.

**Location**: `memory-evals/mem_eval/adapters/persona_adapter.py`
**Change**: Add parallel chunk submission with semaphore

**Expected Result**: Ingestion 140s → 30s

### Phase 2: Server-Side Parallel Extraction (2-4 hours)
Modify PersonaAdapter.ingest_batch to parallelize chunk extraction.

**Location**: `persona/adapters/persona_adapter.py`
**Change**: Already has semaphore - ensure it's used for chunks, not just sessions

### Phase 3: Batch Embeddings (1 hour)
Ensure embedding generation is batched.

**Location**: `persona/services/ingestion_service.py`
**Check**: `_add_embeddings()` method

### Phase 4: Async Queue (1-2 days) - OPTIONAL
Full Honcho-style async processing. Only if Phases 1-3 insufficient.

---

## Quick Speed Test Commands

```bash
# Current speed per chunk
time curl -X POST http://localhost:8000/api/v1/users/test/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "test content 16K chars..."}'

# Batch ingestion speed
time curl -X POST http://localhost:8000/api/v1/users/test/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"content": "chunk1"}, {"content": "chunk2"}, ...]}'
```

---

## Summary: Honcho vs Persona Speed

| Metric | Honcho | Persona (Current) | Persona (Optimized) |
|--------|--------|-------------------|---------------------|
| API Response | ~10ms | ~3000ms | ~3000ms (or ~10ms with queue) |
| Ingestion (100K) | ~30s | ~150s | ~30s |
| Query Latency | ~2s | ~15s | ~15s |
| BEAM 100K Time | ~5 hours* | ~7 hours | ~3 hours |

*Honcho runs 10 conversations in parallel across 4 instances
