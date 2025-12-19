# Memory Indexing Diagnosis Report

## Executive Summary
The Persona memory system achieves only **44.38% overall accuracy** on the LongMemEval benchmark, with particularly poor performance on:
- **Temporal-reasoning**: 30.83% accuracy (133 questions)
- **Multi-session**: 33.33% accuracy (132 questions)

This diagnostic report analyzes the root causes by examining how the knowledge graph is constructed, what specific failures occur, and why the current indexing approach fails for temporal and cross-session queries.

---

## Part 1: System Architecture Analysis

### How the Graph is Built

#### 1.1 Node Extraction Process
The system extracts nodes from conversations using the `GET_NODES` prompt, which creates:
```
- name: "Short, unique handle (5-20 words)"  
- type: Identity, Memory, Preference, Event, Relationship, etc.
```

**Key Issue**: Nodes are extracted as semantic fragments without temporal metadata. For example:
- ❌ Node: "Got car serviced for first time"
- ❌ Node: "GPS system issue with car"
- Missing: When these events occurred relative to each other

#### 1.2 Relationship Generation  
The `GET_RELATIONSHIPS` prompt supports temporal relationships (`PRECEDES`, `FOLLOWS`) but:
- Relationships are generated based on node content, not actual timestamps
- No explicit date/time nodes are created
- Temporal order is lost during extraction

#### 1.3 Session Ingestion
```python
# From ingest.py
for session_idx, (session_turns, session_date) in enumerate(sorted_sessions):
    session_title = f"longmemeval_session_{session_date.replace('/', '_')}"
    await self.ingest_session(user_id, session_content, session_title)
```

**Critical Flaw**: Each session is ingested independently with no cross-session linking mechanism.

---

## Part 2: Temporal-Reasoning Failures (30.83% accuracy)

### Example 1: Car Service Timeline
**Question**: "What was the first issue I had with my new car after its first service?"
- **Gold Answer**: "GPS system not functioning correctly"
- **Generated Answer**: Correct facts but verbose explanation with reconstructed timeline
- **Accuracy**: Marked correct (lucky guess from context)

**Root Cause Analysis**:
```
Graph State:
- Node 1: "First car service on March 15th - great experience"
- Node 2: "GPS system malfunction - replaced by dealership"
- Missing: TEMPORAL_AFTER relationship between nodes
```

The system found both facts through vector similarity but had to infer temporal order from the text content rather than graph structure.

### Example 2: Event Ordering
**Question**: "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?"
- **Gold Answer**: "'Data Analysis using Python' webinar"
- **Generated Answer**: Guessed based on "two months ago" text reference

**Root Cause Analysis**:
```
What the graph has:
- Node: "Attended Data Analysis using Python webinar two months ago"
- Node: "Attended Effective Time Management workshop"

What's missing:
- No Date nodes: "March 28, 2023", "May 15, 2023"
- No OCCURRED_ON relationships
- No BEFORE/AFTER relationships
```

### Example 3: Duration Calculation
**Question**: "How many days did I spend on my solo camping trip to Yosemite?"
- **Gold Answer**: "2 days"
- **Generated Answer**: "The context does not include any specific information..."

**Root Cause Analysis**:
The system cannot calculate durations because:
- Start/end dates are embedded in text, not structured
- No duration properties on event nodes
- Cannot perform date arithmetic on unstructured text

---

## Part 3: Multi-Session Failures (33.33% accuracy)

### Example 1: Aggregation Across Sessions
**Question**: "How many items of clothing do I need to pick up or return?"
- **Gold Answer**: 3
- **Generated Answer**: Only found 1 (navy blazer)

**Session Distribution**:
```
Session 1: "Need to pick up navy blazer from dry cleaner"
Session 2: "Bought shoes online, might need to return them"
Session 3: "Left my jacket at the restaurant last week"
```

**Root Cause**: Each session creates isolated subgraphs with no aggregation mechanism.

### Example 2: Cross-Session Synthesis
**Question**: "How many projects have I led or am currently leading?"
- **Gold Answer**: 2
- **Generated Answer**: Missing projects mentioned in different sessions

**Root Cause Analysis**:
```
Current Graph Structure:
User_1 → Session_1_Graph (Project A mentioned)
User_1 → Session_2_Graph (Project B mentioned)
User_1 → Session_3_Graph (Leading Project A update)

Missing:
- No PROJECT entity that spans sessions
- No aggregation of project mentions
- No deduplication of same project across sessions
```

### Example 3: Information Evolution
**Question**: "How much did I save on the Jimmy Choo heels?"
- **Gold Answer**: "$300"
- **Generated Answer**: "Only know purchase price $200, not original price"

**Session Timeline**:
```
Session 1: "Saw Jimmy Choo heels, regular price $500"
Session 2: "Found them at outlet for $200!"
```

**Root Cause**: Information about the same entity across sessions isn't linked or aggregated.

---

## Part 4: Root Causes Identified

### 1. No Temporal Graph Structure
- **Current**: Temporal information embedded in text nodes
- **Needed**: Explicit date nodes with temporal edges
- **Impact**: Cannot query "what happened after X" or "events between dates Y and Z"

### 2. Session Isolation
- **Current**: Each session creates disconnected subgraph
- **Needed**: Cross-session entity resolution and linking
- **Impact**: Cannot aggregate information across conversations

### 3. Vector Search Dominance
```python
# Current retrieval (rag_interface.py)
similar_nodes = await self.graph_ops.text_similarity_search(query=query, ...)
```
- **Problem**: Retrieves semantically similar content, not temporally or logically connected
- **Example**: Query about "first issue" returns all car issues, not chronologically first

### 4. Missing Aggregation Capabilities
- **Current**: Returns individual nodes
- **Needed**: COUNT, SUM, temporal ordering operations
- **Impact**: Cannot answer "how many", "total", or "in what order" questions

### 5. Lost Context During Extraction
- **Current**: Nodes extracted without preserving conversational context
- **Example**: "It cost $200" → Node: "Cost $200" (what is "it"?)

---

## Part 5: Actionable Recommendations

### Immediate Fixes (High Impact, Low Effort)

#### 1. Add Temporal Metadata to Nodes
```python
class Node(BaseModel):
    name: str
    type: str
    timestamp: Optional[datetime]  # Add this
    session_id: Optional[str]      # Add this
```

#### 2. Create Date Nodes and Relationships
```
Event Node: "Car GPS malfunction"
Date Node: "2023-03-22"
Relationship: Event --OCCURRED_ON--> Date
Relationship: Event --AFTER--> "First car service"
```

#### 3. Implement Cross-Session Linking
```python
# During ingestion, check for entity mentions
if entity_already_exists:
    create_relationship(new_mention, "REFERS_TO", existing_entity)
    update_entity_aggregate_properties()
```

### Medium-Term Improvements

#### 4. Enhance Retrieval for Different Query Types
```python
def get_context_by_type(query_type: str):
    if query_type == "temporal-reasoning":
        return get_temporal_context()  # Use graph traversal
    elif query_type == "multi-session":
        return get_aggregated_context()  # Aggregate across sessions
    else:
        return get_vector_context()  # Current approach
```

#### 5. Add Aggregation Functions
```cypher
# Example Cypher queries for Neo4j
MATCH (u:User)-[:HAS_NODE]->(n:Node {type: 'ClothingItem'})
WHERE n.status = 'needs_pickup'
RETURN COUNT(n) as items_to_pickup
```

### Long-Term Architecture Changes

#### 6. Dual-Index System
- **Temporal Index**: B-tree on timestamps for range queries
- **Semantic Index**: Current vector embeddings

#### 7. Session-Aware Graph Construction
- Maintain session continuity during extraction
- Preserve conversation flow in graph structure
- Link related information across session boundaries

---

## Validation Metrics

To confirm improvements, track:
1. **Temporal-reasoning accuracy**: Target 60%+ (from 30.83%)
2. **Multi-session accuracy**: Target 60%+ (from 33.33%)
3. **Overall accuracy**: Target 65%+ (from 44.38%)

---

## Conclusion

The current system treats each conversation as isolated semantic fragments, losing critical temporal and cross-session relationships. By implementing temporal graph structures, cross-session linking, and query-specific retrieval strategies, we can double the accuracy on failing task types while maintaining performance on successful ones.

The root issue is not the LLM's ability to generate answers, but rather the graph's inability to preserve and query temporal and aggregated information. The proposed fixes address these structural deficiencies in the memory indexing system.# Zep/Graphiti Benchmark Learnings

## Benchmark Results (Dec 18, 2024)

| Metric | Value |
|--------|-------|
| **Accuracy** | 55% (22/40 correct) |
| **Data Integrity** | 100% (0 errors) |
| **Total LLM Calls** | 89,524 |
| **Runtime** | ~8 hours |
| **Avg Query Latency** | 1.97s |

---

## The 45x LLM Call Multiplier

**Key Finding**: Graphiti makes ~45 LLM calls per session ingested.

For each `add_episode()` call, Graphiti runs:
1. Entity extraction (1-2 calls)
2. Entity resolution (1-3 calls)
3. Relation extraction (1-2 calls)
4. Edge deduplication (1-2 calls)
5. Community detection (1 call)
6. Summary generation (1 call)
7. Multiple embedding calls

**Impact**: 40 questions × 50 sessions × 45 calls = **90,000 LLM calls**

---

## Rate Limiting Patterns

### The "Thundering Herd"

With 75 concurrent workers hitting Azure's 750k TPM quota:

```
📊 Est. TPM: 1,440,000/750,000 (192.0%)  ← Burst exceeds quota
📊 Est. TPM: 2,000/750,000 (0.3%)         ← All 75 workers in backoff sleep
📊 Est. TPM: 614,000/750,000 (82%)        ← Recovery
```

**Why**: All workers hit rate limit simultaneously → all enter 2s exponential backoff → quota drops to 0.3% → all wake up → repeat.

### Why `time.sleep()` over `asyncio.sleep()`

We chose `time.sleep()` for the global rate limiter because:
- **Cross-Thread Safety**: `asyncio.sleep()` only works within one event loop. Each question runs in its own thread with its own loop.
- **Global Enforcement**: `time.sleep()` blocks the OS thread, ensuring the 50 RPS limit applies across ALL concurrent questions.
- **Trade-off**: This causes "stuttering" where workers pause together, but guarantees quota safety.

---

## Optimizations Applied

| Setting | Before | After |
|---------|--------|-------|
| Concurrency | 25 workers | 75 workers |
| Rate Limit | 18 RPS | 50 RPS |
| `await` bug (line 345) | ❌ Crashed | ✅ Fixed |
| Missing deps | ❌ pycurl, aiohttp | ✅ Installed |
| Eval method | 1-5 grading | Binary (LongMemEval) |

## Not Applied (Future Work)

- **Token-based limiting**: We limit requests, Azure limits tokens
- **Model swapping**: Use cheaper model for extraction phase
- **Session sampling**: Ingest only relevant sessions, not all

---

## Files Reference

- **Benchmark Code**: `evals/benchmark_runner.py`
- **Zep Adapter**: `evals/adapters/zep_adapter.py`
- **Analysis Examples**: `evals/analysis_examples.json`
- **Results**: `evals/results/zep_graphiti_binary_eval_final.jsonl`
