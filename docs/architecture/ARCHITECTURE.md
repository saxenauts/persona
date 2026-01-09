# Persona Architecture

*A living memory system that syncs with human life*

---

## Philosophy

Every AI conversation today starts from zero. You explain who you are, what you're working on, what you care about—again and again. The AI has no continuity, no growth, no relationship that deepens over time.

Persona exists to change this.

We're not building a database with LLM wrappers. We're building a **digital mind** that mirrors how humans actually think about their lives—in language, narrative, meaning. A system that develops understanding over time, not just stores facts.

**The shift we represent:**
- SoTA 8 months ago: Infrastructure (storage, vectors, graphs, deduplication)
- SoTA now: Language intelligence that understands a human's world made of words

This is personal computing becoming more human—designed for how people actually think.

---

## The Four Pillars of Personal Memory

Human memory isn't a filing cabinet. Cognitive science distinguishes several memory systems, each serving different functions. Persona implements four:

### Episode: What Happened

Episodic memory is autobiographical—the record of lived experience. When you recall "the time I presented at the conference" or "yesterday's conversation with Sarah," you're accessing episodic memory.

**Cognitive Function**: Narrative evidence of life events
**Update Semantics**: Append-only (history doesn't change)
**Key Question**: "What happened?"

```
Episode: "Had a difficult conversation with my manager about the promotion.
         She said Q2 is more realistic but encouraged me to document wins.
         I felt frustrated but understood the reasoning."
         
         Timestamp: 2025-01-15 14:30:00
```

Episodes form the backbone of "what do you remember about X?" queries. They carry the texture of experience—not just facts, but feelings, context, narrative.

### Psyche: Who I Am

Semantic self-memory is different from episodic. It's not about events but about identity—the stable patterns that define who you are. Your preferences, values, beliefs, personality traits.

**Cognitive Function**: Self-schema (identity model)
**Update Semantics**: Consolidate and evolve over time
**Key Question**: "Who am I?"

```
Psyche: "Values deep work and dislikes interruptions during focused time"
        Type: preference

Psyche: "Tends to overthink decisions but commits fully once decided"
        Type: trait
```

The Psyche is what makes the AI "know" you across sessions. It's not extracted from a single conversation—it's consolidated from patterns across many.

### Entity: What Exists

Semantic memory about the world—knowledge about people, places, things, and concepts in the user's life. When you know "Sarah's birthday is June 5th" or "Project Alpha uses React," you're accessing entity knowledge.

**Cognitive Function**: Semantic referents (the nouns of someone's life)
**Update Semantics**: Upsert with conflict resolution
**Key Question**: "What/who is X?"

```
Entity: "Sarah Smith"
        Type: person
        Aliases: ["Sarah", "my girlfriend"]
        Attributes:
          - birthday: "June 5"
          - works_at: "Google"
          - relationship: "girlfriend"
```

Entities carry structured attributes. Facts about entities live ON the entity, not scattered as separate memories.

### Note: What I Intend

Prospective memory—remembering to do things in the future. Humans use external aids because prospective memory is unreliable. Notes are the system's commitment to the user.

**Cognitive Function**: Agent commitments
**Update Semantics**: State machine (active → done/cancelled)
**Key Question**: "What should I do?"

```
Note: "Prepare quarterly review presentation"
      Type: task
      Status: active
      Due: 2025-02-01
```

**Critical Distinction: Entity vs Note**

| Entity                                       | Note                                       |
| -------------------------------------------- | ------------------------------------------ |
| Things that EXIST (nouns)                    | Things to DO (intentions)                  |
| "Sarah", "Paris", "Project Alpha"            | "call Sarah", "book trip", "finish report" |
| Facts are attributes: "Sarah's birthday..."  | Action items with state                    |
| Created when the world contains something    | Created when there's an intention signal   |

---

## Working Memory: The Active Mental Workspace

When humans think, they don't access their entire memory. They hold a limited set of information in **working memory**—the active mental workspace where current thinking happens.

Persona mirrors this. For each interaction, we construct a working memory—the subset of the full memory graph that's relevant right now.

### Composition

```
Working Memory (what the LLM sees)
│
├── UserCard: Identity Anchor
│   "Alex is a software engineer in Austin navigating a career transition.
│    They value work-life balance, prefer morning deep work, and are 
│    currently excited about their side project in AI tooling."
│
├── Recent Context: What's been happening
│   "January 15: Difficult conversation with manager about promotion (led to resume update)
│    January 14: Completed system design for new API
│    January 12: Started exploring new job opportunities"
│
├── Active Context: What's ongoing
│   "Current tasks: Prepare quarterly review, Update resume
│    Preferences: Morning meetings, Deep work blocks
│    Active project: AI tooling side project"
│
└── Memeplex Guidance: How to query effectively
    "Available entities: Sarah (girlfriend), Project Alpha, Manager...
     Time expressions: 'last week', 'before the promotion talk'...
     Relationship types: LED_TO, CAUSED_BY, MENTIONS..."
```

### The UserCard: Identity Anchor

The UserCard sits at primacy position—the first thing the LLM sees. It's a compact prose summary of who this person IS right now.

The UserCard isn't static. It's regenerated through consolidation, synthesizing recent experiences into an updated identity. The person you are today reflects your recent experiences.

```python
UserCard(
    user_id="alex_123",
    timezone="America/Los_Angeles",
    identity_prose="Alex is a software engineer in Austin who values..."
)
```

### Why Prose Over Structure

LLMs process natural language. Rather than feeding them JSON with keys like `memory_type` and `timestamp_utc`, we format memories as readable prose:

```
January 15: Had a difficult conversation with manager about promotion (led to resume update).
```

This isn't aesthetic. Prose leverages the LLM's training on natural text, making comprehension more reliable than structured formats.

---

## Memeplex: Ideas That Cluster Together

A **memeplex** (from memetics, Dawkins/Blackmore) is a group of memes that reinforce and propagate together. In Persona, the Memeplex is how memories cluster and relate.

### Beyond Indexing

The Memeplex isn't just a lookup table. It's:

1. **Navigation**: How do I find related memories from a starting point?
2. **Context**: What entities, relationships, and patterns exist in this person's world?
3. **Query Guidance**: How should the LLM formulate effective queries?

### Three Dimensions

| Dimension | Function | Example |
|-----------|----------|---------|
| **Entity Registry** | WHO/WHAT exists | Sarah, Project Alpha, Austin |
| **Temporal Timeline** | WHEN things happened | Sequences, durations, epochs |
| **Topic Cluster** | ABOUT what themes | Career, relationships, hobbies |

### LLM-Usable Context

The Memeplex provides context so the LLM knows HOW to query:

```
You have access to these entity types: person, project, place, organization
You can filter by time: date_start, date_end (ISO format)
You can follow relationships: LED_TO (what came after), CAUSED_BY (what triggered)

Key entities in this user's world:
- Sarah (person, girlfriend)
- Project Alpha (project, work)
- Manager (person, work relationship)
```

This is like giving someone a database schema so they can write better queries.

---

## Integration: How Understanding Develops

Integration is NOT about graph maintenance or deduplication. It's about how the system **develops understanding over time**—like how humans consolidate memories during sleep.

### What Integration Does

1. **Connects the Dots**: Links new memories to existing ones based on meaning
   - "This conversation mentions Sarah" → link to Sarah entity
   - "This frustration came after the manager meeting" → causal chain

2. **Evolves Identity**: Updates Psyche when patterns emerge
   - Multiple episodes about job frustration → "Considering career change"
   - Repeated mentions of side project → "Passionate about AI tooling"

3. **Resolves Entities**: Understands "my girlfriend", "Sarah", and "her" are the same person

4. **Maintains Narrative**: Ensures temporal and causal chains stay coherent

### When Integration Happens

Integration runs in the background—not blocking the user's experience. Like how you don't consciously consolidate memories; it happens while you sleep.

```
Hot Path (immediate):
User message → Extract memories → Store → Respond
                                    ↓
Cold Path (background):            Queue for integration
                                    ↓
                            Integration develops understanding:
                            - Entity resolution
                            - Psyche consolidation  
                            - Relationship linking
                            - Narrative coherence
```

### The Human Parallel

| Human Cognition | Persona System |
|-----------------|----------------|
| Experience an event | Ingest conversation |
| Form immediate memory | Extract Episode, Entity, Note |
| Sleep consolidation | Integration agent |
| Updated self-understanding | Evolved Psyche, linked graph |

---

## The Pipeline: Life Events to Structured Memory

### Ingestion Flow

```
Raw Life Event (conversation, note, import)
         │
         ▼
    Extraction (LLM)
    ├── What happened? → Episode
    ├── Who/what is mentioned? → Entity  
    ├── What does this reveal about them? → Psyche (rare, significant only)
    └── Any intentions/commitments? → Note
         │
         ▼
    Immediate Storage
    (Episodes linked temporally, Entities with basic MENTIONS links)
         │
         ▼
    Integration Queue
         │
         ▼
    Background Understanding
    (Entity resolution, Psyche consolidation, deeper linking)
```

### Retrieval Flow

```
User Query: "What have I been working on lately?"
         │
         ▼
    Working Memory Construction
    ├── UserCard (identity anchor)
    ├── Recent episodes (time-windowed)
    ├── Active notes (ongoing tasks)
    └── Memeplex context (query guidance)
         │
         ▼
    Agent Loop (LLM decides what tools to use)
    ├── recall(query) → semantic search with filters
    ├── expand_neighbors(id) → graph traversal from anchor
    ├── follow_relationship(id, type) → trace causal/temporal chains
    └── record(text) → save new memories if needed
         │
         ▼
    Synthesized Response
    (Grounded in memories, personalized to identity)
```

---

## Design Principles

### LLM-First

No manual routers. No keyword matching. No heuristic gating.

All decisions are made by LLMs through prompt engineering:
- Tool selection: LLM decides which tool based on context
- Write vs defer: LLM decides when to record based on guidance
- Retrieval strategy: LLM chooses vector vs graph based on query semantics

**Anti-patterns (NEVER do these):**
- `if "remind" in message: enable_record_tool()` — NO keyword routing
- `has_immediate_write_intent(message)` — NO intent classifiers
- Manual routing logic before LLM calls — NO preprocessing gates

### Recency Bias

Human memory weights the recent past heavily. What happened yesterday is more accessible than what happened a year ago. Persona mirrors this with time-windowed retrieval.

### Associative, Not Exhaustive

When you remember something, related memories come to mind unbidden. Persona's graph structure enables this: retrieving Memory A can surface Memory B through their relationship, even if B wasn't directly queried.

### Identity Continuity

The UserCard provides stability across sessions. While individual memories come and go from working memory, the identity summary persists—ensuring consistent understanding of who you are.

---

## Cognitive Science Foundations

The architecture draws from research on memory systems:

**Episodic-Semantic Distinction** (Tulving, 1972): Separation of Episode and Psyche reflects autobiographical event memory vs. general self-knowledge.

**Working Memory** (Baddeley, 2000): Limited-capacity workspace for active information processing—what we construct for each interaction.

**Spreading Activation** (Collins & Loftus, 1975): Graph-based retrieval where one memory activates connected memories parallels semantic memory models.

**Consolidation** (Squire, 1992): UserCard regeneration and Psyche evolution reflect how experiences transform into stable knowledge.

**Context-Dependent Retrieval** (Godden & Baddeley, 1975): Including contextual information in memories aids retrieval.

---

## What We're Building Toward

### Intelligent Forgetting
Not all memories should persist forever. Decay mechanisms reduce salience of irrelevant memories, focusing retrieval on what matters.

### Contradiction Resolution
When new information conflicts with existing memories (changed jobs, moved cities, broke up), the system updates rather than accumulates contradictions.

### Temporal Reasoning
Beyond date filtering—understanding "before the wedding," "during my time at Google," "last summer."

### Intuition and Conjecture
The system develops intuitions: "This person seems stressed lately," "They're more excited about the side project than their job," "The relationship with their manager has shifted."

---

## Summary

Persona is a digital mind that syncs with human life:

| Component | Purpose | Human Parallel |
|-----------|---------|----------------|
| **4 Pillars** | Structure of personal memory | Episode, semantic self, semantic world, prospective |
| **Working Memory** | Active mental workspace | What's "on your mind" |
| **UserCard** | Identity anchor | Self-concept |
| **Memeplex** | How ideas cluster and relate | Associative memory |
| **Integration** | How understanding develops | Sleep consolidation |
| **Graph** | Associative structure | Neural networks of meaning |

We're not building infrastructure. We're building a system that understands a human's world made of words—their experiences, identity, relationships, intentions—and evolves with them.

---

*Persona: Memory that grows with you.*
