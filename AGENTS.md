# Repository Guidelines

## Project Structure

```
persona/           # Core library
├── adapters/      # High-level orchestrators (PersonaAdapter)
├── core/          # Database operations, retrieval, context formatting
├── llm/           # LLM clients, embeddings, prompts
├── models/        # Memory types (memory.py) and API schemas (schema.py)
└── services/      # Business logic (ingestion, persona)

server/            # FastAPI application
├── main.py        # App entry point with lifespan management
├── routers/       # API route definitions
├── dependencies.py # Dependency injection
└── config.py      # Environment configuration

tests/             # Test suite
├── unit/          # Unit tests (no external deps)
└── integration/   # Integration tests (requires Neo4j)
```

## Build & Development

```bash
# Install dependencies
poetry install

# Run API locally
poetry run uvicorn server.main:app --reload

# Run with Docker (recommended)
docker compose up -d

# Run tests
docker compose run --rm test        # Docker (preferred)
poetry run pytest tests/unit -v    # Local unit tests only
```

## Coding Conventions

- **Python 3.12**, PEP 8, 4-space indentation
- **Type hints** on all function signatures
- **Naming**: `snake_case` (modules/functions), `PascalCase` (classes), `UPPER_SNAKE` (constants)
- **Imports**: stdlib → third-party → local; no wildcards

## Testing Guidelines

- **Unit tests**: `tests/unit/test_*.py` - mock external dependencies
- **Integration tests**: `tests/integration/test_*.py` - require Neo4j
- Run integration tests via Docker: `docker compose run --rm test`

## Key Architecture Patterns

1. **4-Pillar Memory Model**: All data stored as `Episode`, `Psyche`, `Entity`, or `Note` types
   - **Episode**: What happened (narrative evidence, append-only)
   - **Psyche**: Who they are (traits, preferences, values, beliefs)
   - **Entity**: What/who exists (people, places, things, concepts with attributes)
   - **Note**: What to do (tasks, goals, reminders - only with intention triggers)
2. **PersonaAdapter**: Single entry point for ingestion (extracts, links, persists)
3. **PersonaService** (`persona/services/persona_service.py`): Unified orchestrator for queries with `query()` and `run_agent()` methods
4. **Retriever**: Time-windowed fetch + link expansion for working-memory context (vector similarity is used by `recall()`/search paths, not the base window)
5. **Tools Layer** (`persona/tools/`): `recall(query)` and `record(text)` dialectic tools with internal intelligence
6. **Dependency Injection**: `GraphOps` injected via FastAPI's `Depends()` — no duplicate connections
7. **Context Engineering**: Prose-based context formatting with UserCard identity anchor

## 4-Pillar Memory Model

| Pillar | Cognitive Function | Update Semantics | Key Question | What It Stores |
|--------|-------------------|------------------|--------------|----------------|
| Episode | Episodic evidence | Append-only | What happened? | Events, experiences, conversations |
| Psyche | Self-schema | Consolidate/evolve | Who am I? | Traits, preferences, values, beliefs |
| Entity | Semantic referents | Upsert with conflict handling | What/who is X? | People, places, orgs, projects, tools, concepts |
| Note | Agent commitments | State machine (active→done) | What should I do? | Tasks, goals, reminders, ideas, lists |

**Entity vs Note (CRITICAL):**
- Entity = Things that EXIST (nouns): "Sarah", "Paris", "Project Alpha"
- Note = Things to DO (intentions): "call Sarah", "book trip to Paris"
- Facts about entities (e.g., "Sarah's birthday is June 5") are **Entity attributes**, not Notes

## Services Layer

**PersonaService** (`persona/services/persona_service.py`): Primary entry point for memory-augmented dialogue.
- `run_agent()`: Single retrieval path using agent loop with recall/record/expand/follow tools. Optional `output_schema` for structured JSON output.
- `ask()`: Thin wrapper that calls `run_agent(output_schema=...)` for structured extraction use cases.

## Tools Layer

**Agent-Native Architecture**: Tools are treated as atomic primitives. Complex features are outcomes achieved by an agent operating in a dialectic loop. For example, answering "what happened after my wedding?" involves recalling "wedding" to find its date, then browsing for memories in the following time range. No complex date-parsing or multi-step routing logic exists in code; the LLM reasons with these primitives.

**ToolContext** (`persona/tools/context.py`): Per-request context injected at execution time. Contains `user_id`, `graph_ops`, `store`, `timezone`, and `user_card`. Injected automatically; not exposed to the LLM.

**Read Tools** (`persona/tools/memory.py`):
**recall(query, date_start?, date_end?, memory_types?, exclude_transcripts?, limit?)**: Semantic search with structured filters. Returns memory snippets ranked by similarity. `memory_types` filters by pillar (episode/psyche/note/entity). `exclude_transcripts` (default true) filters out raw conversation transcripts.
**browse(date_start?, date_end?, memory_types?, limit?, order?)**: Time-ordered listing of memories (order: "desc" default, "asc"). Used for chronological exploration and "what happened last week" queries.
**get_memory(memory_id)**: Fetches full memory content including all metadata, attributes, and status. Used after search to get complete details before updating.
**expand_neighbors(memory_id, relationship_types?)**: Graph-based expansion to find memories connected via any relationship.
**follow_relationship(source_id, relation_type, limit?)**: Traces specific relationship chains (e.g., LED_TO, CAUSED_BY) for narrative continuity.

**Write Tools** (`persona/tools/memory.py`):
**record(text)**: Ingests new information via the integration pipeline, automatically classifying it into the 4-pillar model.
**update_memory(memory_id, updates)**: Modifies existing memory fields. `updates` can include `title`, `content`, `status` (active/completed/cancelled), `due_date`, or `importance`. Used for marking tasks done or editing details.

**Static Registry** (`persona/tools/runner.py`): Global `REGISTRY` maps tool names to handlers. Execution is handled via `execute(tool_call, ctx)` with per-request context injection.

**Bounded Execution**: Handled by `execute_tools_bounded`, providing semaphore-limited concurrency, per-tool timeouts, and partial failure capture.

## Context Engineering Patterns

**UserCard** (`persona/models/memory.py`): Compact identity anchor placed first in context (primacy effect). Contains `user_id`, `timezone`, and `identity_prose` (natural language identity summary).

**Prose Format** (`persona/core/context.py`): `format_working_memory_prose()` renders memories as natural language with `<user>`, `<recent_context>`, `<active_context>` sections.

**Link Prose**: Memory links rendered inline (e.g., "led to X", "caused by Y") for narrative continuity.

## Memeplex: World Model Index (Implemented)

**Memeplex** (`persona/models/memory.py`): Per-user memory index providing the LLM with a "table of contents" for the user's world. Named after memetics (Dawkins) - a memeplex is a group of memes that reinforce and propagate together.

**Schema** (stored as JSON blob in Neo4j):
```python
Memeplex(
    user_id: str,
    topics: List[str],           # Universal themes: "fitness", "AI research"
    people: List[str],           # With context: "Sarah (wife)", "Max (colleague)"
    projects: List[str],         # "Persona", "Home Renovation"
    places: List[str],           # "SF (home)", "Tokyo (2024 trip)"
    concepts: List[str],         # "stoicism", "minimalism"
    last_week_topics: List[str], # Active in last 7 days
    last_month_topics: List[str],# Active in last 30 days
    recent_focus: str,           # "Building Memeplex for v1 release"
    memory_stats: MemoryStats,
)
```

**Key Design**: Topics are universal (everyone has them). Entities (people/projects/places) are optional - a "loner" might only have topics.

**Flow**:
```
Ingestion → Integration → Consolidation → refresh_memeplex()
                                                ↓
                                        Stored in Neo4j
                                                ↓
                         PersonaService.run_agent() fetches it
                                                ↓
                         Injected as {world_model} in system prompt
```

**API Endpoints**:
- `GET /users/{user_id}/memeplex` - Read current memeplex
- `POST /users/{user_id}/memeplex/refresh` - Force refresh from memories

**Key Files**:
- `persona/models/memory.py` - Memeplex model with `to_system_prompt()`
- `persona/services/consolidation_service.py` - `refresh_memeplex()` function
- `persona/core/memory_store.py` - `save_memeplex()` / `get_memeplex()`
- `persona/services/persona_service.py` - Injects `{world_model}` into prompt

## Environment Configuration

Required in `.env`:
```env
# LLM providers: openai or foundry
LLM_SERVICE=foundry/gpt-5.2
EMBEDDING_SERVICE=foundry/text-embedding-3-small

# For OpenAI provider
# OPENAI_API_KEY=sk-...

# For Foundry provider
AZURE_API_KEY=...
AZURE_API_BASE=https://...
AZURE_CHAT_DEPLOYMENT=gpt-5.2
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Graph Database
URI_NEO4J=bolt://neo4j:7687
USER_NEO4J=neo4j
PASSWORD_NEO4J=...
```

## Design Philosophy: LLM-First

**No manual routers. No keyword matching. No heuristic gating.**

All decisions are made by LLMs through prompt engineering:
- Tool selection: LLM decides which tool to call based on context
- Write vs defer: LLM decides when to `record()` based on prompt guidance
- Retrieval strategy: LLM chooses vector vs graph based on query semantics

**Anti-patterns (NEVER do these):**
- `if "remind" in message: enable_record_tool()` - NO keyword routing
- `has_immediate_write_intent(message)` - NO intent classifiers
- Manual routing logic before LLM calls - NO preprocessing gates

**Correct pattern:**
- Expose all tools to the LLM
- Provide clear guidance in system prompt with examples
- Trust the model to make the right decision
- Use prompt engineering, not code branching

This principle applies across ALL interfaces: `/chat`, `/ask`, `/ingest`, integration agents.

## Observability Requirements

This repo prioritizes **optimization work** - every layer must be transparent and measurable.

**Core Principles:**
1. **Always log metrics** - Rate monitors, timing, token counts must emit data even when throttling is disabled
2. **Never silent failures** - Errors bubble up with context; no empty catches
3. **Atomic state changes** - Checkpoints, configs use temp+rename pattern
4. **Resource cleanup** - Failed operations must release connections, reset adapters

**Eval Framework:** See https://github.com/saxenauts/memory-evals (separate repo)

**When adding new components:**
- Include timing instrumentation (`time.time()` around operations)
- Log to structured format (JSONL preferred)
- Expose metrics via env var or stats dict

## Current Eval-Focused Changes (WIP)

This section tracks in-flight changes aimed at improving PersonaMem eval performance.

**Persona Core Changes:**
1. `server/routers/graph_api.py` - Added optional `timestamp` field to `IngestRequest` and `IngestBatchItem` for correct chronology during eval ingestion.
2. `persona/services/persona_service.py` - Added `tool_results`, `user_card_present`, `memeplex_present`, `world_model_chars`, `user_context_chars` to stats output when `include_stats=True`.
3. `persona/tools/runner.py` - `AgentResult.tool_results` now includes each tool's `args` and `output` for deep logging.
4. `persona/tools/schemas.py` - Enhanced tool descriptions with WHEN TO USE / INTERPRETING RESULTS guidance; added `memory_types` and `exclude_transcripts` params to recall.
5. `persona/llm/prompts.py` - Restructured `PERSONAL_AI_SYSTEM_PROMPT` with explicit `<retrieval_policy>`, `<answering_rules>`, `<response_selection>`, and `<disambiguation>` sections to reduce generic responses and improve evidence-anchored answers.
6. `persona/core/backends/neo4j_vector.py` - Added missing `WITH n` clause in Cypher for `db.create.setNodeVectorProperty` calls.

**memory-evals Changes (separate repo):**
1. `PersonaAdapter` now sends `timestamp` on batch ingest, closes sessions via `/sessions/{id}/close`, and refreshes memeplex via `/memeplex/refresh`.
2. `IngestionLog` / `RetrievalLog` schemas extended with `session_ids`, `session_close`, `memeplex_refresh`, `tool_results`, `persona_context`.
3. Runner wires these new stats into deep logs for failure diagnosis.
4. PersonaMem queries now formatted as MCQ prompts with `(a)/(b)/(c)/(d)` options.

**Next Steps:**
- Run full PersonaMem eval with these changes
- Analyze deep logs for remaining failure modes (empty recall, wrong option, missing evolution)
- Iterate on prompt or tool behavior based on findings
