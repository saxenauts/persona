# Ingestion Architecture Analysis: BEAM, PersonaMem, and Real-Life Usage

**Date**: 2026-01-19
**Purpose**: Reference document for understanding ingestion patterns across benchmarks and designing smart batch ingestion

---

## 1. Data Structure Comparison

### BEAM Benchmark
```
Conversation (400K-800K chars total)
├── chat_batch[0] (March 15, 2024) ← natural session boundary
│   ├── message[0]: User: ...
│   ├── message[1]: Assistant: ...
│   └── ... (100-300K chars)
├── chat_batch[1] (March 20, 2024) ← natural session boundary
│   └── ... (100-300K chars)
└── probing_questions: {temporal_reasoning: [...], ...}
```

**Key insight**: BEAM `chat_batches` ARE natural sessions with time anchors. We should ingest each batch as a separate session.

### PersonaMem Benchmark
```
Conversation (32K chars total)
├── Session 1 (system message boundary)
│   └── ~20 turns
├── Session 2
│   └── ~20 turns
└── ...12-14 sessions total
└── MCQ question with 4 options
```

**Key insight**: Sessions defined by system message boundaries. ~20 turns per session.

### LongMemEval Benchmark
```
Conversation (~115K chars)
├── Session 1 (explicit boundaries)
│   └── ~10-12 turns
├── Session 2
│   └── ~10-12 turns
└── ...sessions with clear boundaries
└── Free-form question
```

### Real-Life Usage Pattern
```
User session (~25 turns over 3 hours)
├── Turn 1-25 (continuous conversation)
└── User leaves for 2-3 hours

Session closes → Integration runs → Memeplex refreshes

User returns → New session starts
└── Working memory includes:
    ├── UserCard (identity)
    ├── Last 2 days of episodes
    ├── Week/month summaries from Memeplex
    └── Active notes
```

---

## 2. Architecture Comparison

### Honcho's Winning Strategy

| Aspect | Honcho Approach |
|--------|-----------------|
| **Ingestion unit** | Token-threshold batching (4096 tokens = ~8-10 messages) |
| **Processing** | Async background workers, API returns immediately |
| **Summaries** | Every 20 messages (short), every 60 messages (long) |
| **Session handling** | Work-unit isolation per session+observer+observed |
| **Speed** | Fast - single LLM call per 4096-token batch |

**Honcho's key insight**: NOT turn-by-turn (too slow), NOT full-session (too lossy). Token-threshold batching balances speed and context quality.

### Our Current Persona Architecture

| Aspect | Current Implementation |
|--------|------------------------|
| **4 Pillars** | Episode, Psyche, Entity, Note |
| **Session flow** | ingest() → extraction → persist → temporal chaining |
| **Session close** | triggers integration agent |
| **Integration** | links memories, triggers consolidation |
| **Consolidation** | refreshes UserCard, TemporalContext, Memeplex |
| **Memeplex** | last_week_topics, last_month_topics, recent_focus |
| **Working memory** | UserCard + last 2 days episodes + active notes |

**Problem**: We send entire sessions (100-300K chars) as single extraction units. LLM summarizes and drops atomic facts.

---

## 3. Root Cause of BEAM Failure (0% Accuracy)

```
Current Flow (BROKEN for large contexts):
┌────────────────────────────────────────────────────────────────┐
│ BEAM Conversation (800K chars, 3 batches)                      │
└────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Single batch      │
                    │ ingest call       │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
        │ Session 1 │   │ Session 2 │   │ Session 3 │
        │ 200K chars│   │ 300K chars│   │ 300K chars│
        │ ONE LLM   │   │ ONE LLM   │   │ ONE LLM   │
        │ extraction│   │ extraction│   │ extraction│
        └───────────┘   └───────────┘   └───────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Extracted memories│
                    │ LOSSY - atomic    │
                    │ facts dropped     │
                    └───────────────────┘
```

**The problem**: Each 200-300K char session goes through a SINGLE LLM extraction call. The model can't fit all this in context, so it summarizes and loses specific facts that BEAM questions ask about (e.g., "21 days", "250ms").

---

## 4. Recommended Solution: Token-Threshold Chunking

### Design Principles (from Oracle)

1. **Treat each BEAM `chat_batch` as one Persona session**
2. **Sub-divide large sessions into 4096-token chunks**
3. **Keep same `session_id` across all chunks** (one session, many ingestion items)
4. **Run integration once per session** (after all chunks)
5. **Run consolidation/memeplex refresh once at end** (not per session)

### Target Flow

```
BEAM Conversation (800K chars, 3 batches)
│
├── chat_batch[0] (March 15) = Session 1
│   ├── Chunk 1 (4096 tokens) → extraction → persist
│   ├── Chunk 2 (4096 tokens) → extraction → persist
│   ├── ... (same session_id)
│   └── Session close → Integration agent runs
│
├── chat_batch[1] (March 20) = Session 2
│   ├── Chunk 1 (4096 tokens) → extraction → persist
│   ├── ... (same session_id)
│   └── Session close → Integration agent runs
│
└── Final: Memeplex refresh (once at end)
```

### Chunking Heuristics

```python
TOKEN_ESTIMATE = len(text) // 4  # ~4 chars per token
CHUNK_TARGET = 4096              # tokens (Honcho's default)
CHUNK_MAX_CHARS = 16384          # ~4096 tokens

# Prefer splitting on message boundaries
# Fallback: char windows with 500-char overlap
# Offset timestamps: chunk_timestamp = anchor + timedelta(seconds=chunk_idx)
```

---

## 5. Implementation Plan

### Phase 1: Eval Adapter Fix (1-4 hours)
Modify `memory-evals/mem_eval/adapters/persona_adapter.py`:

```python
def add_sessions(self, user_id: str, sessions: list[dict]) -> None:
    # NEW: Chunk large sessions before sending
    items = []
    for idx, s in enumerate(sessions):
        content = s.get("content", "")
        if len(content) > CHUNK_MAX_CHARS:
            # Split into chunks, all with same session_id
            chunks = self._chunk_content(content, CHUNK_MAX_CHARS)
            for chunk_idx, chunk in enumerate(chunks):
                items.append({
                    "content": chunk,
                    "provider_session_id": f"eval_{idx}",  # Same session
                    "timestamp": base_time + timedelta(seconds=idx*1000 + chunk_idx),
                })
        else:
            items.append({"content": content, "provider_session_id": f"eval_{idx}"})
    
    # Send to API
    self._post(batch_ingest_endpoint, {"items": items})
```

### Phase 2: Server-Side Chunking (1-2 days)
Add chunking to `persona/adapters/persona_adapter.py` or `ingestion_service.py`:

1. Add `max_chunk_tokens` parameter to `ingest()`
2. Pre-split content before LLM extraction
3. Keep same `session_id` across chunks
4. Add `run_consolidation` flag to `close_session()`

### Phase 3: Integration Optimization
1. Dedupe session_ids in batch response
2. Add `run_consolidation=False` option for bulk backfills
3. Single memeplex refresh at end

---

## 6. Key Files Reference

| File | Purpose |
|------|---------|
| `memory-evals/mem_eval/benchmarks/beam.py` | BEAM data loader, builds sessions from chat_batches |
| `memory-evals/mem_eval/adapters/persona_adapter.py` | Eval adapter, calls Persona API |
| `persona/adapters/persona_adapter.py` | Server-side ingestion adapter |
| `persona/services/ingestion_service.py` | LLM extraction logic |
| `persona/services/integration_agent.py` | Post-session memory linking |
| `persona/services/consolidation_service.py` | UserCard + Memeplex refresh |
| `persona/models/memory.py` | Memory types, Memeplex model |

---

## 7. Success Criteria

| Benchmark | Before | Target |
|-----------|--------|--------|
| PersonaMem (32K) | 65.7% | 65%+ (maintain) |
| BEAM (100K) | 0% | 30%+ (measurable improvement) |
| BEAM (500K) | 0% | 20%+ |

**Key metrics to track**:
- Ingestion time per question
- Memories created per session
- Recall precision (do we find the specific fact?)
- Integration links created

---

## 8. Notes for Future Reference

### What We Learned

1. **BEAM tests extreme-scale memory** - it's a fundamentally different challenge than PersonaMem
2. **Honcho wins because of smart batching** - 4096 token threshold, not turn-by-turn
3. **Our architecture is correct** - 4 pillars, integration, memeplex are all good designs
4. **The bug is ingestion granularity** - we need to chunk large sessions

### What NOT to Do

- ❌ Turn-by-turn ingestion (too slow, 2000+ messages)
- ❌ Single extraction on 300K chars (too lossy)
- ❌ Store 300K char transcripts with embeddings (cost explosion)
- ❌ Run consolidation after every chunk (wasteful)

### What TO Do

- ✅ Token-threshold chunking (4096 tokens = ~16K chars)
- ✅ Same session_id across chunks
- ✅ Integration after session close
- ✅ Consolidation once at end
- ✅ Treat BEAM chat_batches as natural sessions
