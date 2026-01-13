# Persona Architecture

*A digital mind that syncs with human life*

---

## The Premise

Every AI conversation today starts from zero. You explain who you are, what you're working on, what you care about—again and again. The AI has no continuity, no growth, no relationship that deepens over time.

Persona exists to change this.

We're not building a database with LLM wrappers. We're building a **memetic organism**—a living graph of memes that represents who you are, what you've experienced, and what matters to you. Like genes for the body, memes are for the mind. The combination of personal stories, ideologies, aesthetics, and symbols we associate ourselves with forms our **memeplex**.

**The shift we represent:**
- SoTA 8 months ago: Infrastructure (storage, vectors, graphs, deduplication)
- SoTA now: Language intelligence that understands a human's world made of words

Other AIs have unique abilities. Persona has context. That's the edge.

---

## The Four Pillars of Personal Memory

Human memory isn't a filing cabinet. Cognitive science distinguishes several memory systems, each serving different functions. Persona implements four:

| Pillar | Cognitive Function | Key Question | Update Semantics |
|--------|-------------------|--------------|------------------|
| **Episode** | Episodic evidence | What happened? | Append-only |
| **Psyche** | Self-schema | Who am I? | Consolidate/evolve |
| **Entity** | Semantic referents | What/who is X? | Upsert with conflict |
| **Note** | Agent commitments | What should I do? | State machine |

### Episode: What Happened

Episodic memory is autobiographical—the record of lived experience. When you recall "the time I presented at the conference" or "yesterday's conversation with Sarah," you're accessing episodic memory.

Episodes form the backbone of "what do you remember about X?" queries. They carry the texture of experience—not just facts, but feelings, context, narrative.

### Psyche: Who I Am

Semantic self-memory is different from episodic. It's not about events but about identity—the stable patterns that define who you are. Your preferences, values, beliefs, personality traits.

The Psyche is what makes the AI "know" you across sessions. It's not extracted from a single conversation—it's consolidated from patterns across many. Most sessions don't reveal new psyche. Only significant identity signals warrant extraction.

### Entity: What Exists

Semantic memory about the world—knowledge about people, places, things, and concepts in your life. When you know "Sarah's birthday is June 5th" or "Project Alpha uses React," you're accessing entity knowledge.

**Critical**: Facts about entities live ON the entity as attributes, not scattered as separate memories. "Sarah works at Google" is an Entity attribute, not a Note.

### Note: What I Intend

Prospective memory—remembering to do things in the future. Notes are the system's commitment to the user. Created ONLY when explicit intention signals present: "remind me", "I need to", due dates, imperatives.

**Entity vs Note distinction**: Entity = nouns that EXIST. Note = intentions to DO.

---

## The Memeplex: World Model Index

A **memeplex** (from memetics, Dawkins/Blackmore) is a group of memes that reinforce and propagate together. In Persona, the Memeplex is the LLM's "table of contents" for a user's world.

### The Problem It Solves

Without Memeplex, the LLM only sees what's retrieved via `recall()`. It has no awareness of what *exists* for the user—can't proactively explore or make connections.

With Memeplex, the LLM has a lightweight index of what exists for the user. The index is free-form text maintained by the LLM and anchored with short IDs for precise references.

### Schema

```python
Memeplex(
    user_id: str,
    updated_at: datetime,
    index: str = "",  # Free-form, LLM-maintained hippocampal index
    memory_stats: MemoryStats,
    temporal_context: Optional[TemporalContext] = None,
)
```

### Key Design Decision

**Free-form index**—the LLM can organize topics, people, projects, and places however it wants. Short IDs (e.g., `ep:a3f2`, `e:b7c1`) keep references unambiguous and easy to trace.

### How It's Used

The Memeplex is refreshed during consolidation (after integration) and stored as a JSON blob in Neo4j. On each `/chat` request, PersonaService injects the index into the system prompt via `{world_model}`.

```markdown
## Memory Index

People: Sarah (wife) [e:b7c1], Max (colleague) [e:4a2f]
Projects: Persona v1 [e:91ab]
Episodes: "Dropped salsa class" [ep:3f2a]


**Last week**: Persona v1, fitness
**Current focus**: Building Memeplex for v1 release

*100 memories | 15 entities | 3 active notes*
```

---

## The Memory Pipeline

### Ingestion: Raw Content → Structured Memory

When content arrives via `/ingest`:

```
Raw Content (conversation, note, import)
    │
    ▼
PersonaAdapter.ingest()
    │
    ▼
MemoryIngestionService.ingest()
    ├── LLM Extraction (structured output)
    │   ├── Episode: What happened (always 1)
    │   ├── Entity: People, places, things mentioned
    │   ├── Psyche: Identity signals (RARE)
    │   └── Note: Intentions with triggers (RARE)
    │
    ├── Embedding Generation (parallel)
    │
    └── MemoryStore.create() → Neo4j
```

**Extraction Prompt** (`persona/services/ingestion_service.py`): The LLM is instructed to be **selective**—most sessions have 0 Psyche, Notes only with explicit intention signals. The Episode IS the memory; make it rich and retrievable.

**Temporal Extraction**: LLM resolves relative time ("yesterday", "last week") to actual dates using provided current time and timezone.

### Integration: Connecting the Graph

After ingestion, the **Batch Integration Agent** runs (fast 2-call mode):

```
Batch Integration
    │
    ├── Memeplex-first context (uses world model for grounding)
    ├── Chronological processing (sorted by event_time for causal integrity)
    ├── single LLM call (vs 5-9 turn agent loop)
    │
    ├── Connections: MENTIONS, LED_TO, CAUSED_BY, NEXT, SAME_AS, CONTRADICTS
    └── Consolidation triggered at end
```

**Connection Types**:
- `MENTIONS`: Episode references an Entity
- `LED_TO`: Causal forward (A caused B)
- `CAUSED_BY`: Causal backward
- `NEXT/PREVIOUS`: Temporal sequence
- `RELATES_TO`: Thematic association
- `CONTRADICTS`: Information conflicts
- `SAME_AS`: Entity deduplication

**Entity Resolution**: Integration agent detects when different names refer to same person/thing ("my wife Sarah" = "Sarah Chen"). Conservative—only merge when confident.

---

## Tool Protocol (Write Policy)

Persona implements a **CRITICAL WRITE POLICY** to prevent cluttering the graph with redundant or irrelevant information:

1. **Explicit Record**: `record()` should ONLY be called when the user explicitly asks to save or remember something ("Save this", "Remember that").
2. **Contextual Awareness**: Statements like "I went to X" are treated as **context for conversation**, not requests to record.
3. **Recall First**: The agent must use `recall()` first to find related memories and respond conversationally based on existing knowledge.

This policy ensures the memory graph remains high-signal and focused on what the user actually wants the AI to persist.

---

### Consolidation: Distilling Identity

After integration:

```
Consolidation Service
    │
    ├── UserCard Refresh
    │   └── LLM synthesizes identity prose from recent psyche + episodes
    │
    ├── TemporalContext Refresh
    │   └── Week/month summaries from recent episodes
    │
    └── Memeplex Refresh
        └── LLM extracts topics/entities, merges with existing
```

**UserCard**: Compact identity anchor placed first in context (primacy effect). Contains `identity_prose`—natural language summary of who this person IS right now.

---

## The Query Path

When a user sends a message to `/chat`:

```
/chat Request
    │
    ▼
PersonaService.run_agent()
    │
    ├── Load UserCard (cached or generate)
    ├── Load Memeplex
    │
    ├── Build System Prompt
    │   ├── {world_model} ← Memeplex.to_system_prompt()
    │   └── {user_context} ← UserCard.identity_prose
    │
    ├── Agent Loop (with tools)
    │   ├── recall(query, ...) → Semantic similarity search
    │   ├── browse(date_range, ...) → Chronological exploration
    │   ├── get_memory(id) → Full content retrieval
    │   ├── record(text) → Ingest new information
    │   ├── update_memory(id, updates) → Modify existing nodes
    │   ├── expand_neighbors(id) → Graph traversal
    │   └── follow_relationship(id, type) → Trace narrative chains
    │
    └── Response (grounded in memories)
```

**LLM-First Design**: No manual routers. No keyword matching. No heuristic gating. The LLM decides which tools to use based on context and prompt guidance. Modern LLMs handle 10-15 tools routinely; our 7 are trivial.

**Retrieval Strategy**: The system prompt guides triangulation:
1. **Semantic**: `recall` for "what did I say about X?"
2. **Temporal**: `browse` for "what happened last week?" or "show me my recent tasks"
3. **Structural**: `get_memory` after search to see full metadata before responding
4. **Graph**: `expand_neighbors` or `follow_relationship` to find context related to a specific entity or event
5. **Type filter**: Episodes for events, Psyche for preferences, Notes for tasks, Entities for world knowledge

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         /chat  /ask  /ingest                            │
├─────────────────────────────────────────────────────────────────────────┤
│                         PersonaService                                  │
│                   (unified orchestrator)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                         Tools Layer                                     │
│  recall | browse | get_memory | record | update_memory | expand | follow  │
├─────────────────────────────────────────────────────────────────────────┤
│                    Memory Index (Memeplex)                              │
│            Topics | People | Projects | Places | Concepts               │
├─────────────────────────────────────────────────────────────────────────┤
│                        Memory Store                                     │
│              create | get | search | link | save_memeplex               │
├─────────────────────────────────────────────────────────────────────────┤
│                      Integration Agent                                  │
│            Entity resolution | Causal chains | Consolidation            │
├─────────────────────────────────────────────────────────────────────────┤
│                       Graph Backend                                     │
│                  Neo4j (with vector index)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Files

| Component | File | Purpose |
|-----------|------|---------|
| **Entry Points** | `server/routers/graph_api.py` | API endpoints |
| **Orchestrator** | `persona/services/persona_service.py` | Query path, agent loop |
| **Ingestion** | `persona/services/ingestion_service.py` | Extraction prompts |
| **Adapter** | `persona/adapters/persona_adapter.py` | Unified ingest interface |
| **Integration** | `persona/services/integration_agent.py` | Graph linking |
| **Consolidation** | `persona/services/consolidation_service.py` | UserCard, Memeplex refresh |
| **Memory Model** | `persona/models/memory.py` | 4-pillar types, Memeplex |
| **Tools** | `persona/tools/memory.py` | recall, browse, get_memory, record, update_memory, expand, follow |
| **Store** | `persona/core/memory_store.py` | Neo4j operations |
| **Prompts** | `persona/llm/prompts.py` | PERSONAL_AI_SYSTEM_PROMPT |

---

## Design Principles

### LLM-First (CRITICAL)

**No manual routers. No keyword matching. No heuristic gating.**

All decisions made by LLMs through prompt engineering:
- Tool selection: LLM decides based on context
- Write vs defer: LLM decides based on prompt guidance
- Retrieval strategy: LLM chooses vector vs graph

**Anti-patterns (NEVER):**
- `if "remind" in message: enable_record()`
- `has_immediate_write_intent(message)`
- Manual preprocessing gates

### Recency Bias

Human memory weights the recent past heavily. What happened yesterday is more accessible than what happened a year ago. Persona mirrors this with time-windowed retrieval and `last_week_topics` / `last_month_topics` in Memeplex.

### Associative, Not Exhaustive

When you remember something, related memories come to mind unbidden. Persona's graph structure enables this: retrieving Memory A can surface Memory B through their relationship, even if B wasn't directly queried.

### Identity Continuity

The UserCard provides stability across sessions. While individual memories come and go from working memory, the identity summary persists—ensuring consistent understanding of who you are.

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/users/{user_id}/chat` | POST | Personal AI conversation |
| `/users/{user_id}/persona/ask` | POST | Structured JSON extraction |
| `/users/{user_id}/ingest` | POST | Ingest single content |
| `/users/{user_id}/ingest/batch` | POST | Batch ingest |
| `/users/{user_id}/sessions/{id}/close` | POST | Close session + integrate |
| `/users/{user_id}/memeplex` | GET | Read world model index |
| `/users/{user_id}/memeplex/refresh` | POST | Force memeplex refresh |

---

## What We're Building Toward

### Implemented
- 4-pillar memory model with cognitive semantics
- Memeplex world model index
- LLM-first tool selection
- Integration agent with entity resolution
- UserCard identity anchor
- Temporal context (week/month summaries)

### Deferred to v2
- **Intelligent Forgetting**: Decay mechanisms for irrelevant memories
- **Temporal Anchor Resolution**: "Year after my wedding" → date range
- **Link Reinforcement**: Fire-together-wire-together dynamics
- **Multi-pass Ingestion**: 2-day context window for richer extraction

---

*Persona: Memory that grows with you.*
