# Persona v2 Memory Design: Session-Based vs. Real-Time

**Date:** 2025-12-21
**Context:** Refactoring Episode/Memory Layer

## Core Problem
We need to balance two conflicting requirements:
1.  **Narrative Continuity**: "Episodes" should act as meaningful summaries of "sessions" or "days" to provide long-term continuity. (Best done asynchronously/post-hoc).
2.  **Real-Time Utility**: Users act in the moment ("Add this to my list"). Goals and Tasks must be capturable immediately, without waiting for a session to end.

## Definitions

| Concept | Definition | Storage Strategy |
| :--- | :--- | :--- |
| **Session** | A contiguous period of active user engagement (e.g., "Morning Chat", "Project Brainstorm"). Ends after $N$ minutes of inactivity. | Raw Logs (Buffer) |
| **Episode** | A **narrative memory unit** summarizing a Session. It is the atomic unit of the "Long Term Timeline". | `Memory(type="episode")` |
| **Day Log** | A logical grouping of all Episodes within 24 hours. (User's mental model of "One Day Log"). | `day_id` grouping query |
| **Goal/Task** | Actionable items. Must be captured **instantly**. | `Memory(type="goal")` |
| **Psyche** | Traits, beliefs, values. Can be lazily extracted. | `Memory(type="psyche")` |

---

## 2. The Hybrid Architecture

We separate the **Hot Path** (Actionable) from the **Cold Path** (Narrative).

### A. The Hot Path (Real-Time)
*Trigger*: Every user message.
*Goal*: Catch urgent items (tasks, reminders) and critical facts immediately.

1.  **Input Analysis**: Lightweight pass (or parallel call) to check: "Does this contain a command, task, or new entity?"
2.  **Extraction**: If yes, extract `Goal` object.
    *   *Example*: "Remind me to call Mom." -> `Goal(title="Call Mom", status="active")`.
3.  **Storage**: Store `Goal` immediately in `MemoryStore`.
4.  **Feedback**: Confirm to user ("Added 'Call Mom' to your tasks").

### B. The Cold Path (Session Consolidation)
*Trigger*: Session timeout (e.g., 30 mins quiet) OR specific "End of Conversation" signal.
*Goal*: Create the permanent narrative record (`Episode`) and refine Identity (`Psyche`).

1.  **Consolidation**: Take all raw messages from the `SessionBuffer`.
2.  **Narrative Generation**: LLM summarizes the session into an `Episode`.
    *   *Format*: "User discussed X, Y, and Z. They expressed frustration about W..."
3.  **Psyche Extraction**: Deep scan for personality traits/values revealed in the *whole* session (more accurate than per-message).
4.  **Linking**:
    *   Link new `Episode` to the `Goals` created during that session (Hot Path items).
    *   Link new `Episode` to PREVIOUS `Episode` (Temporal Chain).
5.  **Storage**: Store `Episode` and `Psyche` items in `MemoryStore`.

---

## 3. "Episode = Day Log" vs. "Episode = Session"

The user requested "Episode is one day log". However, strictly strictly one episode per day has downsides:
*   **Context Window**: A full day/active user might generate 50k tokens. Summarizing once is expensive and loses detail.
*   **Latency**: You don't get the memory until tomorrow.
*   **Granularity**: "What did we talk about this morning?" becomes a sub-part of a massive node.

**Recommendation**:
**Episode = Session**.
*   We maintain the **Day Log** as a *query view* (Get all Episodes where `day_id=TODAY`).
*   This gives the best of both worlds: Granularity + Aggregation.

---

## 4. Implementation Strategy

### Revised Ingestion Flow
Instead of `ingest_memory()` doing everything at once, we split it:

1.  `ingest_realtime(content)`:
    *   Extracts/Stores **Goals** only.
    *   Appends content to **Session Buffer**.

2.  `consolidate_session(session_id)`:
    *   Reads Session Buffer.
    *   Generates **Episode** (Narrative).
    *   Extracts **Psyche**.
    *   Links Episode ↔ Goals.
    *   Connects Temporal Chain (PREVIOUS/NEXT).

### Why this is better
*   **Goals are instant**: User adds a task, it's there.
*   **Narrative is high quality**: LLM sees the *whole* conversation before writing the story, rather than guessing line-by-line.
*   **Performance**: Heavy narrative processing happens async/background.

---

## 5. Potential Pitfalls

1.  **"Orphaned" Goals**: A goal created in Hot Path might lack context if the Cold Path fails or the session buffer is lost.
    *   *Fix*: Store a `source_ref` raw text snippet with the Goal immediately.
2.  **Conversation Continuity**: If a user comes back 2 hours later (new session), they need context from the *previous* session (which might just have finished).
    *   *Fix*: `consolidate_session` must be fast, OR the "Context Window" for the new chat includes the *raw buffer* of the previous unfinished session until it's consolidated.
