# Persona v1 Architecture Plan
> Synthesized from user vision + research on Honcho, SillyTavern, computational neuroscience, and current codebase analysis
> Created: Dec 26, 2025 | Branch: arch/v1-release

---

## Executive Summary

This plan addresses 7 architectural improvements to make Persona a best-in-class memory system. The work is sequenced to minimize risk and maximize learning from evals at each step.

**Immediate Priority**: Rename Goals → Notes (exhaustive commit before any other work)

---

## Phase 0: Terminology Rename (BLOCKING)

### Goals → Notes

**Rationale**: "Goals" is too narrow. The third memory type should hold:
- Goals, Projects, Tasks, Subtasks
- Facts, Favorites, Budget items
- Logs, Contacts, Research notes
- Ideas, Reminders, Lists

**New Name**: `Note` (or `NoteMemory`)
- Flexible enough for structured and unstructured content
- `note_type` field differentiates: `goal`, `task`, `fact`, `list`, `contact`, `reminder`, etc.

**Scope of Rename**:

| Location | Files Affected | Changes |
|----------|----------------|---------|
| **Models** | `persona/models/memory.py` | `GoalMemory` → `NoteMemory`, `GoalOutput` → `NoteOutput`, `goal` → `note` |
| **Core** | `context.py`, `retrieval.py`, `rag_interface.py`, `memory_store.py` | All `goal` references |
| **Prompts** | `persona/llm/prompts.py` | Extraction prompt terminology |
| **Services** | `ingestion_service.py` | Output handling |
| **Evals** | `runner.py`, `log_schema.py`, `deep_logger.py` | Stats and logging |
| **Tests** | 8+ test files | All goal-related tests |
| **Mintlify Docs** | `memory-model.mdx`, `memory-architecture.mdx`, `retrieval.mdx`, `architecture.mdx`, `example.mdx`, `product-design.mdx`, `introduction.mdx`, `quickstart.mdx` | All Goal references |

**Execution**: Single atomic commit with exhaustive find-replace + manual review.

---

## Phase 1: Smart Ingestion with Link Formation

### Current State (Level 1)
```
Raw Content → LLM Extraction → Memory Objects → Basic Links (temporal only)
```
- LLM sees only current input (no past context)
- Links: `PREVIOUS`/`NEXT` (temporal), `derived_from` (episode→psyche/note)
- No semantic relationship discovery

### Target State (Level 2)
```
Raw Content + Past Context → LLM Extraction → Memory Objects → Smart Links → Async Refinement
```

### Design: Multi-Pass Ingestion

**Pass 1: Extraction with Context (Sync)**
```python
def ingest_with_context(content: str, user_id: str):
    # Fetch recent context (last 2 days)
    past_memories = await memory_store.get_recent(user_id, days=2, limit=20)
    past_context = format_memories_for_extraction(past_memories)
    
    # LLM extraction with context awareness
    extraction = await extract_memories(
        content=content,
        past_context=past_context,  # NEW: LLM sees recent memories
        instructions="Identify connections to past memories when relevant"
    )
    
    # Persist memories with basic links
    memories = await persist_memories(extraction)
    
    # Return immediately (async refinement follows)
    return memories
```

**Pass 2: Smart Link Formation (Async, No LLM)**
```python
async def form_smart_links(new_memories: List[Memory], user_id: str):
    """Discover semantic relationships using embeddings, not LLM calls."""
    
    for memory in new_memories:
        # Find semantically similar existing memories
        similar = await vector_search(memory.embedding, top_k=10, threshold=0.7)
        
        for match in similar:
            # Create typed links based on memory types
            link_type = infer_link_type(memory, match)
            await create_link(memory.id, match.id, link_type)
```

**Link Types to Add**:
| Link Type | Meaning | Formation Logic |
|-----------|---------|-----------------|
| `related_to` | Semantic similarity | Embedding cosine > 0.75 |
| `elaborates` | New memory adds detail | Same topic, higher specificity |
| `contradicts` | Conflicting information | High similarity + opposing sentiment |
| `supports` | Reinforcing evidence | High similarity + same sentiment |
| `about_entity` | Shared entity reference | NER extraction match |
| `caused_by` | Causal chain | Temporal + semantic link |

**Pass 3: Link Refinement (Async, Optional LLM)**
```python
async def refine_links_with_llm(memory_id: UUID):
    """Use LLM to validate and enrich discovered links."""
    
    memory = await get_memory(memory_id)
    links = await get_links(memory_id)
    linked_memories = await get_memories([l.target_id for l in links])
    
    # LLM reviews links and suggests improvements
    refinement = await llm_refine_links(
        memory=memory,
        linked_memories=linked_memories,
        current_links=links
    )
    
    # Apply refinements (add/remove/retype links)
    await apply_link_refinements(refinement)
```

### Inspiration: Honcho's Deriver
- **Critical Analysis**: Single-pass reasoning extracts insights
- **Diff Representation**: Only save truly new observations
- **Explicit vs Deductive**: Categorize by "what was said" vs "what was meant"

### Inspiration: Hebbian Learning
- **Fire Together, Wire Together**: Temporal co-occurrence strengthens links
- **STDP**: Order matters - A→B strengthens if A precedes B
- **Replay Frequency**: Links strengthened during consolidation

---

## Phase 2: Intelligent Retrieval

### Current Failures (from eval analysis)
- **suggest_new_ideas**: 29.3% (should be higher for a graph-based system)
- **recall_user_shared_facts**: 45.3% (basic recall failing)

### Root Causes Identified
1. **Shallow traversal**: hop_depth=1 misses multi-hop relationships
2. **No temporal reasoning**: Date filtering is rudimentary
3. **Static context overload**: Always includes all goals+psyche
4. **Poor keyword matching**: No keyword index for fact recall

### Universal Retrieval Strategy

**IMPORTANT**: The memory engine is a universal `/chat` endpoint. Retrieval must work uniformly for all query types - no task-adaptive logic that changes behavior based on detected question type.

| Parameter | Default Value | Rationale |
|-----------|---------------|-----------|
| top_k | 10 | Balance between coverage and noise |
| hop_depth | 2 | Reach multi-hop relationships |
| Static Context | Always include | Psyche and active Notes provide grounding |

**Improvements (applied uniformly)**:
- Temporal filtering from natural language
- Keyword index for exact fact matching
- Multi-hop traversal for reasoning chains

### Temporal Query Enhancement
```python
TEMPORAL_PATTERNS = {
    r"recent(ly)?|lately": lambda d: (d - timedelta(days=7), d),
    r"this week": lambda d: (d - timedelta(days=d.weekday()), d),
    r"last week": lambda d: (d - timedelta(days=d.weekday()+7), d - timedelta(days=d.weekday())),
    r"this month": lambda d: (d.replace(day=1), d),
    r"yesterday": lambda d: (d - timedelta(days=1), d - timedelta(days=1)),
}

def extract_temporal_filter(query: str, anchor_date: datetime):
    for pattern, resolver in TEMPORAL_PATTERNS.items():
        if re.search(pattern, query, re.IGNORECASE):
            return resolver(anchor_date)
    return None
```

### Graph Traversal (Uniform, No Weighting Yet)

In Checkpoint 1, traversal is simple BFS without relationship weighting:

```python
async def graph_traversal(seeds: List[UUID], hop_depth: int = 2):
    """Traverse graph uniformly - no relationship weighting yet.
    
    Weighting will be added in Checkpoint 3 alongside link strength/decay.
    """
    visited = set()
    current_level = set(seeds)
    
    for depth in range(hop_depth):
        next_level = set()
        for node_id in current_level:
            if node_id in visited:
                continue
            visited.add(node_id)
            
            links = await get_links(node_id)
            for link in links:
                next_level.add(link.target_id)
        
        current_level = next_level
    
    return list(visited)
```

**Note**: Relationship weighting moves to Checkpoint 3 with Hebbian strengthening and decay - they belong together as part of the "intelligent link lifecycle" feature set.

---

## Phase 3: Consolidation / Dream System

### Concept
A background process that runs during user's "night" (opposite timezone) to:
1. **Cluster** related memories into themes
2. **Merge** redundant observations
3. **Strengthen** frequently-accessed links
4. **Prune** stale, low-value memories
5. **Update** the User Profile One-Pager

### Inspiration: Honcho's Dreamer
- **Consolidation Dreams**: Merge similar observations
- **Deduplication**: LLM-based reduction
- **times_derived**: Weight frequently reinforced facts

### Inspiration: Neuroscience
- **Sharp-Wave Ripples**: Replay sequences to strengthen paths
- **Pattern Separation**: Ensure similar events have distinct signatures
- **Semantization**: Strip episodic details, preserve semantic core

### Dream Task Types

```python
class DreamTask(Enum):
    CLUSTER = "cluster"           # Group related memories
    CONSOLIDATE = "consolidate"   # Merge redundant memories
    STRENGTHEN = "strengthen"     # Boost important links
    PRUNE = "prune"               # Remove stale memories
    UPDATE_PROFILE = "profile"    # Refresh user one-pager

async def run_dream_cycle(user_id: str, tasks: List[DreamTask] = None):
    """Run consolidation cycle for a user."""
    
    tasks = tasks or [DreamTask.CLUSTER, DreamTask.CONSOLIDATE, 
                      DreamTask.STRENGTHEN, DreamTask.PRUNE, DreamTask.UPDATE_PROFILE]
    
    for task in tasks:
        if task == DreamTask.CLUSTER:
            await cluster_memories(user_id)
        elif task == DreamTask.CONSOLIDATE:
            await consolidate_memories(user_id)
        elif task == DreamTask.STRENGTHEN:
            await strengthen_links(user_id)
        elif task == DreamTask.PRUNE:
            await prune_stale_memories(user_id)
        elif task == DreamTask.UPDATE_PROFILE:
            await update_user_profile(user_id)
```

### Clustering Algorithm
```python
async def cluster_memories(user_id: str):
    """Cluster memories into 2-5 major life themes."""
    
    # Get all memories with embeddings
    memories = await memory_store.get_all(user_id, with_embeddings=True)
    
    # HDBSCAN clustering (allows variable cluster count)
    embeddings = np.array([m.embedding for m in memories])
    clusterer = HDBSCAN(min_cluster_size=5, min_samples=3)
    labels = clusterer.fit_predict(embeddings)
    
    # Create/update cluster nodes
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Noise
        
        cluster_memories = [m for m, l in zip(memories, labels) if l == cluster_id]
        cluster_summary = await llm_summarize_cluster(cluster_memories)
        
        await create_or_update_cluster(
            user_id=user_id,
            cluster_id=cluster_id,
            summary=cluster_summary,
            member_ids=[m.id for m in cluster_memories]
        )
```

### Flexible Triggers
```python
class ConsolidationConfig(BaseModel):
    """Configuration for dream cycle triggers."""
    
    # Time-based
    schedule_cron: str = "0 3 * * *"  # 3 AM daily
    user_timezone: str = "UTC"
    
    # Event-based
    on_session_end: bool = False
    on_memory_count: int = 100  # Trigger after N new memories
    
    # Manual
    allow_manual_trigger: bool = True
```

---

## Phase 4: User Profile One-Pager

### Concept
A maintained, static-ish summary of the user that:
1. Contains core identity (who/what/where/how)
2. Tracks current life themes (2-3 active threads)
3. Acts as a retrieval index for smarter queries
4. Updated during consolidation, not every interaction

### Inspiration: SillyTavern Character Cards
- **V2/V3 Spec**: Structured JSON with markdown description
- **Keyword Triggers**: Automatic retrieval based on keywords
- **Token Budgeting**: Hard cap on profile size in context

### Profile Structure
```python
class UserProfile(BaseModel):
    """The User Profile One-Pager."""
    
    # Core Identity (rarely changes)
    identity: IdentitySection
    
    # Current Context (updated during dreams)
    current_themes: List[ThemeSection]
    
    # Retrieval Index (keyword → memory cluster mapping)
    retrieval_index: Dict[str, List[UUID]]
    
    # Metadata
    last_updated: datetime
    version: int

class IdentitySection(BaseModel):
    """Who is this person?"""
    name: Optional[str]
    age: Optional[int]
    location: Optional[str]
    occupation: Optional[str]
    core_values: List[str]  # Top 3-5
    personality_summary: str  # 2-3 sentences

class ThemeSection(BaseModel):
    """A current life thread."""
    title: str  # e.g., "Career Transition", "Health Journey"
    summary: str  # 2-3 sentences
    keywords: List[str]  # For retrieval triggering
    related_cluster_id: Optional[UUID]
    started: datetime
    last_activity: datetime
```

### Keyword-Based Retrieval Index
```python
async def build_retrieval_index(user_id: str) -> Dict[str, List[UUID]]:
    """Build keyword → memory mapping for fast retrieval."""
    
    index = defaultdict(list)
    
    # Get all notes (facts, goals, etc.)
    notes = await memory_store.get_by_type("note", user_id)
    
    for note in notes:
        # Extract keywords (can use LLM or simple NLP)
        keywords = extract_keywords(note.content)
        
        for keyword in keywords:
            index[keyword.lower()].append(note.id)
    
    return dict(index)

async def retrieve_by_keywords(query: str, user_profile: UserProfile) -> List[UUID]:
    """Use profile index for fast keyword-based retrieval."""
    
    query_keywords = extract_keywords(query)
    matched_ids = set()
    
    for keyword in query_keywords:
        if keyword.lower() in user_profile.retrieval_index:
            matched_ids.update(user_profile.retrieval_index[keyword.lower()])
    
    return list(matched_ids)
```

---

## Phase 5: Time & Chronology

### Natural Language Time Resolution
```python
class TemporalResolver:
    """Resolve natural language time references."""
    
    RELATIVE_PATTERNS = {
        "yesterday": lambda d: d - timedelta(days=1),
        "last week": lambda d: d - timedelta(weeks=1),
        "last month": lambda d: d - timedelta(days=30),
        "last year": lambda d: d - timedelta(days=365),
    }
    
    ANCHOR_PATTERNS = {
        r"after (?:my |the )?(.+)": "find_event_after",
        r"before (?:my |the )?(.+)": "find_event_before",
        r"during (?:my |the )?(.+)": "find_event_during",
        r"the (?:year|month|week) (?:of|after) (?:my |the )?(.+)": "find_relative_to_event",
    }
    
    async def resolve(self, query: str, user_id: str) -> Optional[DateRange]:
        # Check relative patterns first
        for pattern, resolver in self.RELATIVE_PATTERNS.items():
            if pattern in query.lower():
                return DateRange(start=resolver(datetime.now()))
        
        # Check anchor patterns (requires graph lookup)
        for pattern, method in self.ANCHOR_PATTERNS.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                event_desc = match.group(1)
                return await getattr(self, method)(event_desc, user_id)
        
        return None
    
    async def find_event_after(self, event_desc: str, user_id: str) -> DateRange:
        """Find time period after a described event."""
        # Search for the event
        events = await memory_store.search(user_id, event_desc, limit=1)
        if events:
            return DateRange(start=events[0].timestamp)
        return None
```

### Memory Versioning (TODO - Future)
```python
class MemoryVersion(BaseModel):
    """Track memory changes over time."""
    memory_id: UUID
    version: int
    content: str
    changed_at: datetime
    change_type: Literal["created", "updated", "merged", "split"]
    previous_version: Optional[int]
```

---

## Phase 6: Causal Chains & Intelligent Links

### Link Intelligence
Links should carry meaning beyond just "connected":

```python
class IntelligentLink(BaseModel):
    """A link with causal and semantic intelligence."""
    
    source_id: UUID
    target_id: UUID
    relation: str
    
    # Intelligence attributes
    strength: float = 1.0  # 0-1, updated by replay
    confidence: float = 1.0  # How certain is this link?
    causal_direction: Optional[Literal["forward", "backward", "bidirectional"]]
    
    # Temporal
    created_at: datetime
    last_activated: datetime
    activation_count: int = 0
    
    # Decay
    decay_rate: float = 0.01  # Per day
    
    def current_strength(self) -> float:
        """Calculate strength with temporal decay."""
        days_since = (datetime.now() - self.last_activated).days
        return self.strength * math.exp(-self.decay_rate * days_since)
```

### Hebbian Link Strengthening
```python
async def strengthen_on_co_retrieval(memory_ids: List[UUID]):
    """Strengthen links between memories retrieved together."""
    
    for i, id_a in enumerate(memory_ids):
        for id_b in memory_ids[i+1:]:
            link = await get_link(id_a, id_b)
            if link:
                # Existing link - strengthen
                link.strength = min(1.0, link.strength + 0.1)
                link.activation_count += 1
                link.last_activated = datetime.now()
                await update_link(link)
            else:
                # New co-occurrence - create weak link
                await create_link(
                    source_id=id_a,
                    target_id=id_b,
                    relation="co_retrieved",
                    strength=0.3,
                    confidence=0.5
                )
```

### Causal Chain Discovery
```python
async def discover_causal_chains(user_id: str):
    """During consolidation, discover causal patterns."""
    
    # Get temporally ordered episodes
    episodes = await memory_store.get_by_type("episode", user_id, order_by="timestamp")
    
    for i, current in enumerate(episodes[:-1]):
        next_episode = episodes[i + 1]
        
        # Check semantic continuity
        similarity = cosine_similarity(current.embedding, next_episode.embedding)
        
        if similarity > 0.6:
            # Potential causal link
            # Use LLM to verify causality
            is_causal = await llm_check_causality(current, next_episode)
            
            if is_causal:
                await create_link(
                    source_id=current.id,
                    target_id=next_episode.id,
                    relation="caused",
                    causal_direction="forward",
                    confidence=is_causal.confidence
                )
```

---

## Implementation Sequence (REVISED)

### Checkpoint 0: Foundation (No Eval)
1. **Goals → Notes rename** (exhaustive, single commit)

### Checkpoint 1: Smarter Ingestion + Retrieval (Full Eval)
*Features PersonaMem CAN measure*

2. **Context-aware extraction** (past 2 days context)
3. **Smart link formation** (sync/async hybrid, embedding-based)
4. **Keyword index** for Notes (direct fact matching)
5. **Temporal filtering** (natural language → date ranges)
6. **Multi-hop traversal** (depth 2-3, uniform - no weighting)

**Eval**: Full PersonaMem run
**Expected**: 51.7% → 60-65%

### Checkpoint 2: Memory Card + Consolidation (Mixed Eval)
*Partial PersonaMem signal + custom metrics*

7. **User Profile One-Pager** (static summary)
8. **Memory card as retrieval index** (keyword triggers)
9. **Async consolidation** (dream cycle infrastructure)
10. **Clustering** into life themes

**Eval**: PersonaMem + cluster quality + profile completeness
**Expected**: 60-65% → 68% (modest, features not fully measured)

### Checkpoint 3: Weighing + Forgetting (Custom Eval Only)
*PersonaMem NOT appropriate*

11. **Relationship weighting** in traversal
12. **Link strength** (Hebbian - fire together wire together)
13. **Temporal decay** on links and memories
14. **Intelligent forgetting** (prune low-value)
15. **Causal chain discovery**

**Eval**: Custom long-term simulation only
**Expected PersonaMem**: No change (not measured)

---

## Eval Appropriateness Summary

| Checkpoint | PersonaMem Signal | Notes |
|------------|-------------------|-------|
| 0 (Rename) | None | Terminology only |
| 1 (Ingestion+Retrieval) | Strong | Core features measurable |
| 2 (Card+Consolidation) | Weak | Single-session limitation |
| 3 (Weighing+Forgetting) | None | Requires time passage |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| PersonaMem Overall | 51.7% | 70%+ |
| suggest_new_ideas | 29.3% | 60%+ |
| recall_user_shared_facts | 45.3% | 70%+ |
| LongMemEval | 64.1% | 75%+ |

---

## Research References

- **Honcho**: Deriver/Dreamer architecture, Explicit vs Deductive categorization
- **SillyTavern**: Character Cards V2/V3, keyword-triggered lorebooks, token budgeting
- **Neuroscience**: CLS theory, Hebbian learning, Sharp-Wave Ripples, pattern separation
- **BEAM Benchmark**: 10 memory abilities, nugget-based evaluation

---

*Document created: Dec 26, 2025*
*Next action: Execute Goals → Notes rename*
