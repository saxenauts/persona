# Session & Provenance Design

## Why This Matters

When Persona stores a memory like "User prefers morning meetings", we need to answer:

- **Where did this come from?** (Which conversation? Which AI?)
- **When was this said?** (Original timestamp, not ingestion time)
- **Can I see the original?** (The actual conversation that led to this)
- **Is this still accurate?** (Was it contradicted later?)

This is **provenance** - the origin trail of every piece of knowledge.

---

## The Problem: Multiple Sources, One Memory Graph

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Persona   │  │   Claude    │  │   Slack     │  │  ChatGPT    │
│    /chat    │  │   Export    │  │   Thread    │  │   Export    │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     PERSONA MEMORY    │
                    │        GRAPH          │
                    │                       │
                    │  Episodes, Psyche,    │
                    │  Entities, Notes      │
                    └───────────────────────┘
```

**Questions we must answer:**

1. "This memory says I like coffee - where did I say that?"
2. "Show me all memories from my Claude conversation yesterday"
3. "I think the AI got something wrong - what was the original conversation?"
4. "Did I say this to Persona or to ChatGPT?"

---

## The Solution: Three Layers of Tracking

### Layer 1: Session Identity

Every conversation gets a **session_id** - a stable identifier that groups related messages.

```
session_id = "{provider}:{provider_session_id}"
```

**Examples:**

| Source | session_id | Meaning |
|--------|------------|---------|
| Persona /chat | `persona:abc123` | Chat session abc123 with Persona |
| Claude export | `claude:conv_xyz` | Conversation xyz from Claude |
| Slack thread | `slack:C123_1234567890` | Thread in channel C123 |
| ChatGPT | `chatgpt:chatcmpl-abc` | ChatGPT conversation |
| Voice memo | `voice:recording_2024-01-03` | Voice transcription |
| Manual note | `manual:note_123` | User-entered note |

**Why the prefix?**
- Claude might have session "abc123"
- ChatGPT might also have session "abc123"
- Prefix prevents collision: `claude:abc123` ≠ `chatgpt:abc123`

### Layer 2: Source Metadata (on every memory)

Every memory (Episode, Psyche, Entity, Note) carries provenance fields:

```python
class BaseMemory:
    # Identity
    memory_id: str              # Unique ID for this memory
    user_id: str                # Who this belongs to
    
    # PROVENANCE (the trail)
    session_id: str             # Which conversation: "claude:conv_xyz"
    source_type: str            # What kind of source: "claude", "slack", "persona_chat"
    source_ref: Optional[str]   # Provider's original ID (for linking back)
    
    # Timing
    observed_at: datetime       # When this HAPPENED (original timestamp)
    created_at: datetime        # When we STORED it (ingestion timestamp)
    
    # Extraction info
    extraction_model: str       # Which LLM extracted this: "gpt-4o"
    extraction_confidence: float # How confident (future use)
```

**Example memory with provenance:**

```python
PsycheMemory(
    memory_id="mem_abc123",
    user_id="user_456",
    
    # THE PROVENANCE TRAIL
    session_id="claude:conv_xyz",           # From Claude conversation xyz
    source_type="claude",                   # It was a Claude chat
    source_ref="msg_789",                   # Specifically message 789
    
    observed_at="2024-01-03T10:30:00Z",     # User said this at 10:30am
    created_at="2024-01-03T15:00:00Z",      # We ingested it at 3pm
    
    extraction_model="gpt-4o",              # GPT-4o extracted this
    
    # The actual content
    content="Prefers morning meetings over afternoon ones",
    psyche_type="preference",
)
```

**What this tells us:**
- This preference came from a Claude conversation
- The user said it at 10:30am on Jan 3
- We imported it later at 3pm
- GPT-4o did the extraction
- We can trace back to message 789 in conversation xyz

### Layer 3: Transcript Storage (the raw evidence)

Optionally store the **original conversation** as a special Episode.

```python
EpisodeMemory(
    memory_id="transcript_conv_xyz",
    user_id="user_456",
    
    session_id="claude:conv_xyz",
    source_type="transcript",        # Special marker: this is raw data
    
    content="""
[2024-01-03T10:25:00Z] user: Hey, can we schedule our weekly sync?
[2024-01-03T10:25:30Z] assistant: Sure! What time works best for you?
[2024-01-03T10:26:00Z] user: I prefer mornings, definitely before lunch.
[2024-01-03T10:26:30Z] assistant: Got it, I'll suggest morning slots.
    """,
    
    observed_at="2024-01-03T10:25:00Z",  # When conversation started
)
```

**Why store transcripts?**

| Use Case | How Transcript Helps |
|----------|---------------------|
| Debug | "The AI said I like coffee but I said tea" - check the transcript |
| Audit | Legal/compliance: prove what was actually said |
| Re-extraction | New model? Re-process old transcripts for better memories |
| Context | "What were we talking about when I said that?" |

**Why transcripts are separate from regular Episodes:**

Regular Episode: `"User prefers morning meetings"` (semantic, searchable)
Transcript: `"[user] I prefer mornings..."` (raw, not searchable by default)

Transcripts are **evidence**, not **knowledge**. We exclude them from normal recall.

---

## How The Pieces Connect

### Complete Flow: Claude Conversation → Persona Memories

```
STEP 1: User exports Claude conversation
        ┌─────────────────────────────────────┐
        │ Claude Export (JSON)                │
        │ conversation_id: "conv_xyz"         │
        │ messages: [                         │
        │   {role: "user", content: "..."},   │
        │   {role: "assistant", content: "..."}│
        │ ]                                   │
        └─────────────────────────────────────┘
                         │
                         ▼
STEP 2: POST /ingest with session context
        {
          "content": "<full conversation text>",
          "source_type": "claude",
          "provider_session_id": "conv_xyz",
          "store_transcript": true
        }
                         │
                         ▼
STEP 3: Generate canonical session_id
        session_id = "claude:conv_xyz"
                         │
                         ▼
STEP 4: Store transcript (if requested)
        Episode(
          session_id="claude:conv_xyz",
          source_type="transcript",
          content="<raw conversation>"
        )
                         │
                         ▼
STEP 5: Extract memories (LLM)
        ┌─────────────────────────────────────┐
        │ LLM analyzes conversation, extracts:│
        │ - Episode: "Discussed scheduling"   │
        │ - Psyche: "Prefers mornings"        │
        │ - Entity: "Weekly sync meeting"     │
        │ - Note: "Schedule sync for Tuesday" │
        └─────────────────────────────────────┘
                         │
                         ▼
STEP 6: All memories tagged with same session_id
        Every memory gets:
        - session_id: "claude:conv_xyz"
        - source_type: "claude"
        - observed_at: <original timestamps>
```

### Querying by Session

```python
# "Show me everything from my Claude chat yesterday"
memories = await get_memories_by_session("claude:conv_xyz")

# Returns:
# - Episode: "Discussed scheduling preferences"
# - Psyche: "Prefers morning meetings"
# - Entity: "Weekly sync - recurring meeting"
# - Note: "Schedule sync for Tuesday"
# - (optional) Transcript Episode if stored
```

### Tracing a Memory Back

```python
# User asks: "Why do you think I prefer mornings?"

memory = PsycheMemory(content="Prefers morning meetings", ...)

# We can answer:
f"""
This came from your Claude conversation on {memory.observed_at}.
Session: {memory.session_id}
Source: {memory.source_type}

Would you like to see the original conversation?
"""

# If they say yes, fetch the transcript:
transcript = await get_transcript(memory.session_id)
```

---

## The Three Implementation Steps

### Step 1: Standardize session_id Mapping

**Current problem:**
```python
# Inconsistent - sometimes timestamp, sometimes UUID, no provider prefix
session_id = session_id or f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}"
```

**After:**
```python
def get_session_id(provider: str, provider_session_id: Optional[str] = None) -> str:
    """
    Create a canonical session ID.
    
    Args:
        provider: Source system ("persona", "claude", "slack", etc.)
        provider_session_id: The ID from that system (optional)
    
    Returns:
        Canonical ID like "claude:conv_xyz" or "persona:550e8400-..."
    
    Examples:
        get_session_id("claude", "conv_xyz")     → "claude:conv_xyz"
        get_session_id("persona", None)           → "persona:550e8400-e29b-..."
        get_session_id("slack", "C123_12345")    → "slack:C123_12345"
    """
    if provider_session_id:
        return f"{provider}:{provider_session_id}"
    return f"{provider}:{str(uuid4())}"
```

**Where this is used:**

| Endpoint | How session_id is generated |
|----------|----------------------------|
| `/chat` | `get_session_id("persona", request.session_id)` |
| `/ingest` | `get_session_id(request.source_type, request.provider_session_id)` |

### Step 2: Transcript Episode Storage

**New optional parameter on /ingest:**

```python
class IngestRequest(BaseModel):
    content: str                           # The conversation text
    source_type: str = "conversation"      # Provider: "claude", "slack", etc.
    provider_session_id: Optional[str]     # Provider's ID for this conversation
    store_transcript: bool = False         # NEW: Save raw conversation?
```

**What happens when `store_transcript=True`:**

```python
async def ingest(request: IngestRequest, ...):
    session_id = get_session_id(request.source_type, request.provider_session_id)
    
    # Step 1: Store transcript if requested
    if request.store_transcript:
        transcript_episode = EpisodeMemory(
            user_id=user_id,
            session_id=session_id,
            source_type="transcript",      # Special marker
            content=request.content,        # Raw conversation
            observed_at=datetime.now(),
        )
        await store.save_memory(transcript_episode)
    
    # Step 2: Extract memories as normal
    memories = await extraction_service.extract(request.content, session_id=session_id)
    
    # Step 3: All memories get the same session_id
    for memory in memories:
        memory.session_id = session_id
        memory.source_type = request.source_type
        await store.save_memory(memory)
```

### Step 3: Exclude Transcripts from Recall

**The problem:**

Without filtering, `recall("meetings")` might return:
```
1. Episode: "User discussed scheduling preferences" ✓ Good
2. Transcript: "[user] can we schedule? [assistant] sure..." ✗ Raw noise
```

**The fix:**

```python
async def recall(
    query: str,
    user_id: str,
    exclude_transcripts: bool = True,  # Default: don't return transcripts
    ...
) -> List[Memory]:
    
    filters = {"user_id": user_id}
    
    if exclude_transcripts:
        filters["source_type"] = {"$ne": "transcript"}
    
    results = await vector_search(query, filters=filters)
    return results
```

**When you DO want transcripts:**

```python
# Debug API: Get the original conversation
@router.get("/users/{user_id}/sessions/{session_id}/transcript")
async def get_transcript(user_id: str, session_id: str):
    transcripts = await store.get_memories(
        user_id=user_id,
        session_id=session_id,
        source_type="transcript",
    )
    return transcripts
```

---

## Provenance Query Examples

### "Where did this memory come from?"

```python
memory = await get_memory("mem_abc123")

provenance = {
    "source": memory.source_type,           # "claude"
    "session": memory.session_id,           # "claude:conv_xyz"
    "when_said": memory.observed_at,        # "2024-01-03T10:30:00Z"
    "when_stored": memory.created_at,       # "2024-01-03T15:00:00Z"
    "extracted_by": memory.extraction_model # "gpt-4o"
}
```

### "Show me all memories from yesterday's Claude chat"

```python
memories = await get_memories(
    user_id=user_id,
    session_id="claude:conv_xyz",
)
# Returns all Episode, Psyche, Entity, Note from that session
```

### "What was the original conversation?"

```python
transcript = await get_memories(
    user_id=user_id,
    session_id="claude:conv_xyz",
    source_type="transcript",
)
# Returns the raw conversation text
```

### "Find memories from Slack vs Claude"

```python
# All from Slack
slack_memories = await get_memories(user_id=user_id, source_type="slack")

# All from Claude  
claude_memories = await get_memories(user_id=user_id, source_type="claude")
```

---

## Summary

| Layer | What It Tracks | Why It Matters |
|-------|---------------|----------------|
| **session_id** | Which conversation | Group related memories, query by session |
| **source_type** | Which provider | Filter by source, understand origin |
| **source_ref** | Provider's ID | Link back to original system |
| **observed_at** | When it happened | Temporal accuracy |
| **created_at** | When we stored it | Audit trail |
| **Transcript** | Raw conversation | Evidence, debug, re-extraction |

**The key insight:**

Memories are **extracted knowledge**. Transcripts are **raw evidence**. 

Both live in the same graph, but transcripts are hidden by default. When something seems wrong, you can always trace back to the source.

---

## Future: Integration Agent

After steps 1-3, we can add async processing:

```
Session closes
      ↓
Integration Agent (async):
  - Compare with existing memories
  - Resolve entities ("Sam" = "Samuel Chen")
  - Detect contradictions
  - Cross-reference sessions
  - Update Psyche consolidations
```

But that's a future phase. Steps 1-3 give us the foundation.
