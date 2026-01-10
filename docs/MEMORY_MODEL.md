# The Persona Memory Model

*A living architecture for personal memory*

---

## Why Memory Matters

Every conversation with an AI starts from zero. You explain who you are, what you're working on, what you prefer—again and again. The AI has no continuity, no growth, no relationship with you that deepens over time.

Persona exists to change this.

We're building a memory layer that doesn't just *store* information but *understands* it. Not a database of facts, but a living graph of experiences, identity, and intention that evolves with each interaction.

---

## The Four Pillars of Personal Memory

Human memory isn't a filing cabinet. Cognitive science distinguishes several memory systems, each serving different purposes. Persona implements four:

### Episodes: What Happened

Episodic memory is autobiographical—the record of lived experience. When you recall "the time I presented at the conference" or "yesterday's conversation with Sarah," you're accessing episodic memory. These memories are anchored in time and place, rich with context.

In Persona, an **Episode** captures a narrative unit: something that happened, was discussed, or occurred. Episodes have timestamps, summaries, and the raw content of what transpired. They form the backbone of "what do you remember about X?" queries.

```
Episode: "Had a difficult conversation with my manager about the promotion timeline.
         She said Q2 is more realistic but encouraged me to document my wins.
         I felt frustrated but understood the reasoning."

         Timestamp: 2025-01-15 14:30:00
         Source: conversation
```

### Psyche: Who I Am

Semantic self-memory is different from episodic. It's not about events but about identity—the stable patterns that define who you are. Your preferences, values, beliefs, personality traits. "I'm an introvert" isn't a memory of an event; it's knowledge about yourself.

In Persona, **Psyche** nodes capture these identity facets. They're extracted from conversations but represent enduring truths rather than momentary events. The psyche is what makes the AI "know" you across sessions.

```
Psyche: "Values deep work and dislikes interruptions during focused time"
        Type: preference

Psyche: "Tends to overthink decisions but commits fully once decided"
        Type: trait
```

### Notes: What I Intend

Prospective memory—remembering to do things in the future—is distinct from remembering the past. Humans use external aids (to-do lists, reminders, calendars) because prospective memory is notoriously unreliable.

**Notes** in Persona serve this function. They capture tasks, goals, projects, reminders, facts to remember, lists, and ideas. Unlike episodes (which happened) or psyche (which is), notes represent intention and outstanding commitments.

```
Note: "Prepare quarterly review presentation"
      Type: task
      Status: active
      Due: 2025-02-01

Note: \"Look into meditation apps - Sarah recommended Headspace\"
      Type: reminder
      Status: active
```

### Entities: What Exists

Semantic memory about the world—knowledge about people, places, things, and concepts. When you know \"Sarah's birthday is June 5th\" or \"Project Alpha uses React,\" you're accessing entity knowledge. This isn't about events (episodic) or about yourself (psyche)—it's about referents in your world.

In Persona, **Entity** nodes capture structured knowledge about named things. Each entity has a canonical name, optional aliases, and structured attributes. Facts about entities are stored as attributes on the entity, not as separate Notes.

```
Entity: "Sarah Smith"
        Type: person
        Aliases: ["Sarah", "my girlfriend"]
        Attributes:
          - birthday: "June 5"
          - works_at: "Google"
          - met_at: "College"

Entity: "Project Alpha"
        Type: project
        Attributes:
          - tech_stack: "React, Node.js"
          - deadline: "Q2 2025"
```

**Critical Distinction**: Entity vs Note
- **Entity** = Things that EXIST (nouns): "Sarah", "Paris", "Project Alpha"
- **Note** = Things to DO (intentions): "call Sarah", "book trip to Paris"
- **Facts** = Entity ATTRIBUTES: "Sarah's birthday is June 5" → attribute on the Sarah entity, NOT a Note

Notes are created only when there's an intention signal ("remind me", "I need to", due dates).

---

## Why Graph, Not Just Vectors

Modern AI memory systems typically use vector embeddings: convert text to numbers, store in a vector database, find similar content via cosine similarity. This works for "find things like X" but fails for deeper queries.

Consider: "What led to my decision to change jobs?"

A vector search might find memories containing the words "job" and "decision." But the causal chain—the sequence of events, frustrations, conversations, and realizations that *caused* the decision—requires understanding relationships between memories, not just similarity to a query.

Persona uses a **graph structure** because human memory is associative. One memory triggers another not because they contain similar words, but because they're connected—causally, temporally, thematically. The graph captures these connections explicitly.

### Relationship Types

| Relation | Meaning | Example |
|----------|---------|---------|
| `MENTIONS` | Entity reference | Episode mentions Sarah |
| `LED_TO` | Causal forward link | Argument with manager → decision to update resume |
| `CAUSED_BY` | Causal backward link | Burnout caused by three months of crunch |
| `NEXT` / `PREVIOUS` | Temporal sequence | Monday's meeting → Tuesday's follow-up |
| `RELATES_TO` | Thematic association | Both memories involve the project launch |

These relationships enable queries that vector search cannot answer:
- "What happened after the interview?"
- "What caused the change in my sleep patterns?"
- "Trace the sequence of events leading to the product launch"

---

## The Architecture

### Ingestion: From Conversation to Memory

When you talk to a Persona-powered assistant, the conversation flows through an ingestion pipeline:

1. **Extraction**: An LLM analyzes the conversation and extracts structured memories—episodes, psyche items, entities, and notes. This isn't keyword extraction; it's semantic understanding of what matters.

2. **Embedding**: Each memory is converted to a vector embedding for similarity search. We use OpenAI's `text-embedding-3-small` by default, but the embedding layer is pluggable.

3. **Linking**: The system identifies relationships between the new memories and existing ones. Did this event follow from a previous one? Does this preference connect to an existing trait?

4. **Persistence**: Memories and their relationships are written to the graph database (Neo4j). The vector embeddings enable similarity search; the graph structure enables associative traversal.

### Storage: The Memory Graph

Each user has an isolated subgraph. Memory nodes contain:

- **Content**: The natural language description
- **Type**: episode, psyche, entity, or note
- **Temporal anchors**: timestamp, created_at, day_id
- **Provenance**: source_type, session_id, extraction_model
- **Retrieval aids**: embedding vector, importance score
- **Metadata**: access_count, last_accessed, status (for notes)

Relationships (edges) connect memories:

```
(Episode: "Got the job offer") -[LED_TO]-> (Episode: "Celebrated with family")
                              -[CAUSED_BY]-> (Episode: "Final interview went well")

(Psyche: "Values stability") -[RELATES_TO]-> (Note: "Research 401k options")
```

### Retrieval: Building Working Memory

When the AI needs context for a response, it constructs a **working memory**—a subset of the full memory graph relevant to the current moment. This mirrors how human working memory holds a limited set of active information.

The retrieval process:

1. **Time-windowed fetch**: Get recent episodes (last N days), recent psyche updates, and active notes. Recency matters—what happened yesterday is more likely relevant than what happened a year ago.

2. **Link expansion**: For the retrieved memories, fetch their relationships. If we have Memory A, and A led to B, include that connection in context.

3. **Prose formatting**: Convert the graph subset into natural language. LLMs consume text, not database rows. The formatted context uses XML-style sections:

```xml
<user>
Alex is a software engineer in Austin focused on work-life balance.
Currently navigating a career transition while maintaining family connections.
</user>

<recent_context>
January 15: Had a difficult conversation with manager about promotion timeline.
  She said Q2 is more realistic (led to Updated resume draft).
January 14: Completed the system design for the new API.
</recent_context>

<active_context>
Preference: Values deep work and dislikes interruptions.
Tasks: Prepare quarterly review, Research 401k options.
</active_context>
```

### The UserCard: Identity Anchor

At the top of every context sits the **UserCard**—a compact prose summary of who this person is. It's placed first because of the primacy effect: LLMs weight early context heavily.

The UserCard isn't a static profile. It's regenerated periodically through consolidation, synthesizing recent memories into an updated identity summary. The person you are today reflects your recent experiences.

```python
class UserCard:
    user_id: str
    timezone: str
    identity_prose: str  # "Alex is a software engineer in Austin..."
    updated_at: datetime
```

---

## Tool Architecture: The Memory Protocol

When an AI agent needs to interact with memory, it uses tools. Persona provides a structured protocol—a clean interface between the LLM and the memory graph.

### The Philosophy: Structured Parameters, Not Parsing

Early designs had the LLM send natural language queries: "find memories about work from last week." The system then parsed these queries, extracting time references and topics.

This is backwards. LLMs are excellent at understanding time expressions. Why parse "last week" on our side when the LLM can directly provide `date_start: "2025-01-20"` and `date_end: "2025-01-27"`?

The current design inverts the responsibility:

| Old Pattern | New Pattern |
|-------------|-------------|
| LLM sends: "memories about Sarah from last month" | LLM sends: `{query: "Sarah", date_start: "2024-12-01", date_end: "2024-12-31"}` |
| System parses time expression | System executes structured query directly |
| Parsing errors possible | Clean, deterministic execution |

### Agent-Native Design

Persona follows an **agent-native design philosophy**. As described in *Every.to*: "Tools should be atomic primitives. Features are outcomes achieved by an agent operating in a loop."

No complex routing logic or date-parsing heuristics exist in the Persona codebase. Instead, the LLM reasons with these primitives to navigate the user's world.

**Example Workflow**: "What was I doing after my wedding?"
1. `recall("wedding")` → Finds the event and its date.
2. LLM reasons about the subsequent date range.
3. `browse(date_start="2023-07-01", ...)` → Explores memories chronologically.

### Available Tools

#### Read Tools

**recall(query, date_start?, date_end?, memory_types?, limit?)**: Semantic search with structured filters. Returns memory snippets ranked by similarity.

**browse(date_start?, date_end?, memory_types?, limit?, order?)**: Time-ordered listing of memories. Unlike `recall`, results are sorted by `event_time`, not similarity. Used for chronological exploration (e.g., "what happened last week", "show me June 2023").
- `order`: "desc" (newest first, default) or "asc" (oldest first).

**get_memory(memory_id)**: Fetches full memory content including all metadata, attributes, and status. Used after search to get complete details before updating.

**expand_neighbors(memory_id, relationship_types?)**: Given a memory, explore its graph connections. Returns neighbors linked by specified relationship types.

**follow_relationship(source_id, relation_type, limit?)**: Traces a specific relationship chain (e.g., `LED_TO`, `CAUSED_BY`) for narrative continuity.

#### Write Tools

**record(text)**: Ingests new information via the integration pipeline, automatically classifying it into the 4-pillar model.

**update_memory(memory_id, updates)**: Modifies existing memory fields. `updates` can include `title`, `content`, `status` (`active`/`completed`/`cancelled`), `due_date`, or `importance`.

### Tool Selection Strategy

| Goal | Primary Tool | Sorting |
|------|--------------|---------|
| Find specific information | `recall` | Similarity |
| Review recent history | `browse` | Recency |
| Mark task as done | `update_memory` | N/A |
| Deep dive into a memory | `get_memory` | N/A |
| Trace narrative cause/effect | `follow_relationship` | Temporal |

### Execution Model

Tools execute with bounded parallelism. If the LLM requests three recall operations, they run concurrently with:
- Semaphore-limited concurrency (default: 8)
- Per-tool timeout (default: 30s)
- Partial failure capture (one tool failing doesn't crash the batch)

The **ToolContext** carries per-request state (user_id, graph connection, timezone) and is injected at execution time—never exposed to the LLM.

---

## Design Principles

### Prose Over Structure

LLMs process natural language. Rather than feeding them structured JSON with keys like `memory_type` and `timestamp_utc`, we format memories as readable prose:

```
January 15: Had a difficult conversation with manager about promotion (led to resume update).
```

This isn't just aesthetic. Prose leverages the LLM's training on natural text, making comprehension more reliable than parsing structured formats.

### Recency Bias

Human memory is heavily weighted toward the recent past. What happened yesterday is more accessible than what happened a year ago. Persona mirrors this with time-windowed retrieval—recent memories are fetched by default, older ones require explicit queries.

### Associative, Not Exhaustive

When you remember something, related memories come to mind unbidden. Persona's graph structure enables this: retrieving Memory A can surface Memory B through their relationship, even if B wasn't directly queried.

### Identity Continuity

The UserCard provides stability across sessions. While individual memories come and go from working memory, the identity summary persists—ensuring the AI maintains a consistent understanding of who you are.

---

## Neuroscience References

The Persona architecture draws inspiration from cognitive science research on memory systems:

**Episodic-Semantic Distinction** (Tulving, 1972): The separation of Episode and Psyche memories reflects Tulving's foundational work distinguishing autobiographical event memory from general knowledge about the self.

**Working Memory** (Baddeley, 2000): The limited-capacity working memory constructed for each interaction mirrors Baddeley's model of a temporary workspace for active information processing.

**Spreading Activation** (Collins & Loftus, 1975): Graph-based retrieval where one memory activates connected memories parallels spreading activation models of semantic memory.

**Consolidation** (Squire, 1992): The UserCard regeneration process—synthesizing recent memories into identity—reflects memory consolidation, where experiences transform into stable knowledge.

**Context-Dependent Retrieval** (Godden & Baddeley, 1975): Including contextual information (time, source, session) in memories reflects research showing that context aids retrieval.

---

## What's Next

The current architecture handles storage, retrieval, and context formatting. Several capabilities are planned:

**Intelligent Forgetting**: Not all memories should persist forever. Decay mechanisms will reduce the salience of irrelevant memories over time, focusing retrieval on what matters.

**Contradiction Resolution**: When new information conflicts with existing memories (you changed jobs, moved cities, broke up), the system should update rather than accumulate contradictions.

**Entity Resolution**: When new information mentions "my wife," "Sarah," or "she," the system resolves these to the same entity. Entities enable queries like "everything about Sarah" and store facts as attributes rather than scattered notes.

**Temporal Reasoning**: Beyond simple date filtering—understanding "before the wedding," "during my time at Google," "last summer."

---

## Getting Started

To use Persona's memory system:

```python
from persona.services.persona_service import PersonaService
from persona.core.graph_ops import GraphOps

# Initialize with graph connection
async with GraphOps() as graph_ops:
    service = PersonaService(graph_ops)
    
    # Use the agent loop with tools
    result = await service.run_agent(
        user_id="user_123",
        query="What have I been working on lately?",
    )
    
    # Or for structured output
    result = await service.run_agent(
        user_id="user_123",
        query="Find everything about my career transition and summarize",
        include_stats=True,
    )
```

For detailed API documentation, see [API.md](./API.md).

---

*Persona: Memory that grows with you.*
