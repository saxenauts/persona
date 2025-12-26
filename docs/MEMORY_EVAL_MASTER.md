# Memory Eval System: Master Document

**Status:** Living document - consolidates all eval research and planning
**Last Updated:** 2025-12-25
**Branch:** `evals/v1`

---

## Executive Summary

**Goal:** Build a custom evaluation system for memory systems (Persona vs Graphiti vs others).

**Key Insight:** Existing benchmarks (LongMemEval, LoCoMo) are outdated - modern LLMs can fit 100K+ tokens in context. We need BEAM-scale (10M tokens) benchmarks that actually stress memory systems.

**Our Edge:** Not just accuracy scores, but **root cause analysis** - understanding WHY systems fail (cross-session linking? temporal reasoning? entity resolution?).

**Budget:** ~$2000 Azure credits

---

## Part 1: Current Eval Results

### LongMemEval (500 questions) - COMPLETE

| System | Accuracy | Multi-Session | Temporal | Knowledge-Update | Single-Session |
|--------|----------|---------------|----------|------------------|----------------|
| **Persona** | **64.1%** | **68.3%** | **36.7%** | **75.0%** | 85.7% |
| **Graphiti** | 53.2% | 29.6% | 22.5% | 59.8% | **91.9%** |

**Persona wins by +10.9%** overall. Key differentiators:
- Multi-session: Persona +38.7% (cross-session linking)
- Temporal: Both weak (<40%), but Persona leads
- Single-session: Graphiti wins (better immediate retrieval)

### PersonaMem (589 questions) - IN PROGRESS

| System | Progress | Current Accuracy |
|--------|----------|------------------|
| Graphiti | 319/589 (54%) | ~65% |
| Persona | Pending | - |

---

## Part 2: The Benchmark Landscape (Dec 2025)

### What to Use

| Benchmark | Tokens | Questions | Priority | Why |
|-----------|--------|-----------|----------|-----|
| **BEAM** | Up to 10M | 2,000 | **MUST HAVE** | Only benchmark that actually stresses memory systems |
| **ConvoMem** | Varies | 75,336 | **ADD** | Shows where memory > full-context (150+ conversations) |
| **PersonaMem** | ~175 turns | 589 | **HAVE** | Tests personalization specifically |
| **LongMemEval** | ~115K | 500 | **HAVE** | Cross-system comparison baseline |

### What to Drop

| Benchmark | Tokens | Why Drop |
|-----------|--------|----------|
| **LoCoMo** | 16-26K | Too short - modern LLMs handle easily. No knowledge-update tests. |

### Key Research Findings

**ConvoMem bombshell:**
> For histories under 150 interactions, full-context beats RAG (70-82% vs 30-45%). Memory systems only prove value at 150+ conversations.

**Honcho's assessment:**
> "LongMem/LoCoMo are no longer well suited to test memory systems today. Expensive frontier models can take all tokens in-context."

**BEAM innovation:** First benchmark scaling to 10M tokens. Tests 10 distinct memory abilities:
1. Information Extraction
2. Temporal Reasoning  
3. Contradiction Resolution
4. Instruction Following
5. Preference Following
6. Multi-Session Reasoning
7. Knowledge Update
8. Event Ordering
9. Working Memory
10. Scratchpad (accumulating facts)

---

## Part 3: Framework Stack Decision

### Comparison

| Framework | Language | Foundation | Memory Benchmarks | Extensibility |
|-----------|----------|------------|-------------------|---------------|
| **OpenBench** (Groq) | Python | Inspect-AI (UK AISI) | None (add ourselves) | Plugin system ✓ |
| **DeepEval** | Python | Custom | N/A (metrics layer) | 50+ metrics ✓ |
| **memorybench** | TypeScript | Custom | LoCoMo, LongMemEval, ConvoMem | Limited |
| **Anthropic Bloom** | Python | Custom | N/A (behavioral testing) | Dynamic scenarios |

### Decision: OpenBench + DeepEval + Our Memory Layer

```
┌─────────────────────────────────────────────────┐
│                 PersonaEval                      │
├─────────────────────────────────────────────────┤
│  Memory Layer (our additions)                   │
│  - BEAM benchmark loader                        │
│  - ConvoMem benchmark loader                    │
│  - Session provenance tracking                  │
│  - Retrieval audit / failure analysis           │
│  - Root cause visualization                     │
├─────────────────────────────────────────────────┤
│  DeepEval (metrics)                             │
│  - Answer quality, faithfulness, retrieval      │
├─────────────────────────────────────────────────┤
│  OpenBench (foundation)                         │
│  - Inspect-AI core, CLI, checkpointing          │
└─────────────────────────────────────────────────┘
```

**Why this stack:**
1. Python (matches our codebase, unlike memorybench's TypeScript)
2. Inspect-AI is industry standard (UK AISI, METR, Apollo use it)
3. Plugin system allows clean benchmark additions
4. DeepEval's metrics are battle-tested
5. Can contribute BEAM/ConvoMem loaders back to community

### Inspiration from Anthropic Bloom (Dec 19, 2025)

Bloom is Anthropic's open-source framework for **dynamic behavioral evaluations**. Instead of static benchmarks that become stale, Bloom generates evaluation scenarios on-the-fly.

**The 4-Stage Pipeline:**

| Stage | What It Does | Output |
|-------|--------------|--------|
| **Understanding** | Analyzes target behavior + example transcripts | Knowledge base of mechanisms/motivations |
| **Ideation** | Generates diverse test scenarios | 100+ unique scenarios per behavior |
| **Rollout** | Runs scenarios against target model | Conversation transcripts |
| **Judgment** | Scores transcripts for behavior presence | 1-10 behavior presence score |

**Key Metric:** Elicitation rate = proportion of rollouts scoring ≥7/10

**Why This Matters for Memory Evals:**

Static benchmarks (LongMemEval, LoCoMo) suffer from:
1. Training set contamination
2. Capability obsolescence (modern LLMs handle easily)
3. Limited scenario diversity

Bloom's approach: **describe the behavior you want to test → generate scenarios automatically**.

**Adaptation Concept for Memory Testing:**

Instead of static question sets, define memory behaviors to test:

```yaml
# Example: memory_behaviors.yaml
behaviors:
  - name: "cross_session_entity_linking"
    description: "Can the system recognize the same entity mentioned across different sessions?"
    example: "User mentions 'my project Alpha' in session 1, then 'Alpha project' in session 5"
    
  - name: "temporal_ordering"
    description: "Can the system order events correctly based on when they occurred?"
    example: "User asks 'What did I do before starting the new job?'"
    
  - name: "preference_evolution"  
    description: "Can the system track how user preferences change over time?"
    example: "User liked coffee in early sessions, switched to tea recently"
```

Then Bloom-style pipeline:
1. **Understanding**: Parse behavior definition
2. **Ideation**: Generate 100 synthetic conversations that should exercise this behavior
3. **Rollout**: Run memory system on these conversations, ask probing questions
4. **Judgment**: Score whether memory system correctly handled the behavior

**Status:** Aspirational - Phase 2 after core static benchmarks work. But the concept is powerful: **generate tests, don't just run fixed benchmarks**.

**Link:** https://github.com/safety-research/bloom

---

## Part 4: What We're Building

### Core Visualization Need

```
Question Analysis View:
├── Question: "How many projects have I led?"
│   ├── Answer: "2" (Gold) vs "1" (Generated) ❌
│   ├── Sessions Involved: 5 sessions
│   │   ├── Session 1 (Day 1): Project A mentioned
│   │   ├── Session 2 (Day 3): Project B started
│   │   └── ...
│   ├── Memories Ingested:
│   │   ├── Node: "User leads Project A" ✓ retrieved
│   │   ├── Node: "User leads Project B" ✗ NOT retrieved
│   └── Root Cause: Cross-session linking failure
```

### Root Cause Categories

| Category | Description | Example |
|----------|-------------|---------|
| **Cross-Session Gap** | Info across sessions not linked | Projects in different sessions not aggregated |
| **Temporal Blindness** | Date ordering not captured | "What happened first" fails |
| **Entity Resolution** | Same entity not recognized | "Project Alpha" vs "Alpha project" |
| **Insufficient Context** | Retrieved context too short | <500 tokens when 2000 needed |
| **Semantic Mismatch** | Query doesn't match embeddings | "led projects" vs "managed initiatives" |
| **Ingestion Failure** | Data not indexed properly | Node not created from session |

### Data Model

```sql
-- Core provenance tracking
CREATE TABLE questions (
    id TEXT PRIMARY KEY,
    benchmark TEXT,
    question_text TEXT,
    question_type TEXT,
    gold_answer TEXT
);

CREATE TABLE question_sessions (
    question_id TEXT,
    session_id TEXT,
    session_date DATE,
    turn_count INT,
    content_preview TEXT
);

CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    question_id TEXT,
    session_id TEXT,
    node_type TEXT,
    node_content TEXT
);

CREATE TABLE retrievals (
    question_id TEXT,
    memory_id TEXT,
    retrieval_rank INT,
    similarity_score FLOAT,
    was_relevant BOOLEAN
);

CREATE TABLE results (
    question_id TEXT PRIMARY KEY,
    generated_answer TEXT,
    correct BOOLEAN,
    failure_category TEXT,
    retrieval_recall FLOAT
);
```

---

## Part 5: Production Learnings

### Azure OpenAI Quotas

| Deployment | Model | RPM | TPM | Best For |
|------------|-------|-----|-----|----------|
| gpt-5 | gpt-5 | 100K | 10M | Batch evals (highest throughput) |
| gpt-5.2 | gpt-5.2 | 100K | 10M | Batch evals |
| text-embedding-3-small | - | 10K/10s | 10M | Embeddings |

### LLM Call Economics

Different memory systems have vastly different call patterns:

| System | Calls/Ingest | Calls/Retrieve | 175-turn session |
|--------|--------------|----------------|------------------|
| **Persona** | 1 | 1-2 | ~3 calls |
| **Graphiti** | 40-60 | 1-20 | 40-80+ calls |
| **Honcho** | 1-3 | 1 | ~5 calls |

**Key insight:** Graphiti's O(N) entity resolution creates call explosion. This explains the rate limit issues in our evals.

### Rate Limit Patterns

**The Thundering Herd:**
```
Workers burst → all hit 429 → all backoff → quota drops to 0% → all wake → repeat
```

**Solution:** Client-side rate limiting (`GRAPHITI_RPS=3`)

### Architectural Comparison

| Dimension | Persona | Graphiti | Honcho |
|-----------|---------|----------|--------|
| **Core Model** | Knowledge graph + psyche | Entity-relationship graph | Peer representations |
| **LLM Calls/Ingest** | 1 | 40-60 | 1-3 |
| **Retrieval Speed** | 12.8s | 1.2s | ~1s |
| **Ingestion Speed** | Instant | 3-16 min | Fast (async) |
| **Cross-Session** | Graph structure | Weak linking | Global representations |

---

## Part 6: Implementation Plan

### Phase 1: Deep Logging (Week 1)

**Goal:** Capture all provenance data during eval runs.

```python
@dataclass
class QuestionEvalLog:
    question_id: str
    question_text: str
    gold_answer: str
    
    # Session provenance
    sessions: List[SessionLog]
    session_count: int
    day_spread: int
    
    # Ingestion provenance
    memories_created: List[MemoryNode]
    
    # Retrieval provenance
    query_text: str
    retrieved_nodes: List[MemoryNode]
    relevant_nodes: List[MemoryNode]  # Post-hoc
    
    # Result
    generated_answer: str
    correct: bool
    failure_category: Optional[str]
```

### Phase 2: Failure Analysis (Week 2)

**Goal:** Auto-classify failures and map to root causes.

```python
class FailureAnalyzer:
    def analyze(self, log: QuestionEvalLog) -> FailureAnalysis:
        if not self._check_ingestion(log):
            return FailureAnalysis(category="ingestion_failure")
        if self._retrieval_recall(log) < 0.5:
            return FailureAnalysis(category="retrieval_gap")
        # ... more categories
```

### Phase 3: Visualization (Week 3-4)

**Views:**
1. Overview Dashboard - Accuracy by type, failure distribution
2. Question Deep Dive - Full provenance chain
3. Session Timeline - Visual timeline of sessions and memories
4. Retrieval Audit - Retrieved vs relevant comparison
5. Failure Patterns - Heatmap by type × category

**Stack:** FastAPI + React + D3.js + SQLite

### Phase 4: Benchmark Integration (Week 5)

**Goal:** Add BEAM and ConvoMem to our system.

**Contribution strategy:**
- Add to OpenBench as plugins (open source)
- Keep root cause analysis internal (competitive advantage)

---

## Part 7: Open Questions

1. **BEAM dataset:** Is it publicly available yet? (Oct 2025 paper)
2. **OpenBench maturity:** Alpha release - stable enough?
3. **Dynamic scenarios:** Can we adapt Bloom's approach for memory testing?
4. **Post-hoc relevance:** How to determine which nodes SHOULD have been retrieved? (LLM judge? Manual annotation?)

---

## Part 8: Action Items

### Immediate

- [ ] Let Graphiti PersonaMem eval finish (54% complete)
- [ ] Run Persona on PersonaMem golden set
- [ ] Compare results, identify failure patterns

### Short-term (This Week)

- [ ] Set up OpenBench + DeepEval environment
- [ ] Create Persona adapter for OpenBench
- [ ] Implement deep logging in eval runner
- [ ] Find/download BEAM dataset

### Medium-term (Next 2 Weeks)

- [ ] Build failure analysis pipeline
- [ ] Create first visualization views
- [ ] Run BEAM on Persona + Graphiti
- [ ] Document where Persona wins (marketing story)

### Aspirational

- [ ] Explore Bloom-style dynamic scenario generation
- [ ] Contribute BEAM loader to OpenBench
- [ ] Build real-time eval monitoring dashboard

---

## Links

| Resource | URL |
|----------|-----|
| OpenBench | https://github.com/groq/openbench |
| DeepEval | https://deepeval.com |
| Inspect-AI | https://inspect.ai-safety-institute.org.uk |
| BEAM Paper | https://arxiv.org/abs/2510.27246 |
| ConvoMem Paper | https://arxiv.org/abs/2511.10523 |
| Anthropic Bloom | https://github.com/safety-research/bloom |
| memorybench | https://github.com/supermemoryai/memorybench |
| Eugene Yan QA Evals | https://eugeneyan.com/writing/qa-evals/ |
| Hamel Husain Evals FAQ | https://hamel.dev/blog/posts/evals-faq/ |
| Langfuse | https://langfuse.com |
| LangChain OpenEvals | https://github.com/langchain-ai/openevals |

---

## Part 9: Research Deep Dive (Dec 2025)

### Eugene Yan: Long-Context Q&A Evaluation

**Two orthogonal dimensions for Q&A:**

| Dimension | What It Measures | Why It Matters |
|-----------|------------------|----------------|
| **Faithfulness** | Does answer rely ONLY on source document? | Critical for legal, medical, financial |
| **Helpfulness** | Is answer relevant, comprehensive, concise? | User satisfaction |

**Key insight:** A faithful answer isn't always helpful. Both must be measured.

**For memory systems specifically:**
- False positives = hallucinations (made up answers)
- False negatives = missed information (retrieval failure) ← **This is our core problem**

### Hamel Husain: Sharp Opinions on Evals

**Build custom annotation tools.** Most impactful investment. With Cursor/Lovable, build in hours. Teams with custom tools iterate 10x faster.

**Why custom beats off-the-shelf:**
- Shows all context from multiple systems in one place
- Renders data in product-specific way
- Designed for your specific workflow

**Error Analysis is the MOST important activity:**

```
1. Create Dataset → representative traces
2. Open Coding → human annotator writes open-ended notes about failures
3. Axial Coding → categorize notes into "failure taxonomy"
4. Iterative Refinement → until theoretical saturation (~100 traces)
```

**Binary (pass/fail) over Likert scales.** 1-5 ratings create noise. Pass/fail forces clarity.

**Minimum Viable Eval Setup:**
1. A dataset of representative examples
2. A domain expert who can judge pass/fail
3. A way to view traces (custom tool or notebook)
4. Error analysis to find failure patterns

**Anti-pattern:** Automated rubrics that both create AND score evaluations. "Stacking abstractions" hides flaws behind high scores.

### The Simplicity Question: Framework vs Python Scripts

**Hamel's actual workflow:**
> "I tend to use these tools as a backend data store and use Jupyter notebooks as well as my own custom built annotation interfaces for most of my needs."

**Conclusion for us:** Don't over-invest in frameworks. Start simple:

| Approach | When to Use |
|----------|-------------|
| **Plain Python + notebooks** | Prototyping, error analysis, custom workflows |
| **Langfuse/Phoenix for tracing** | Production observability, storing traces |
| **Custom annotation UI** | High-volume human review |
| **Framework (Inspect/OpenBench)** | Only when contributing to community or need their specific benchmarks |

### Observability Tools Comparison

| Tool | Strength | Best For |
|------|----------|----------|
| **Langfuse** | Deep tracing, prompt management, evals | Teams wanting full control, self-hosted option |
| **Phoenix (Arize)** | Embedding visualization, drift detection | Data science teams, notebook-centric |
| **Braintrust** | Experiments, A/B testing | Rapid iteration on prompts |
| **Weave (W&B)** | ML experiment tracking integration | Teams already using W&B |

**For our memory eval system:** Langfuse is the best fit. Open source, self-hostable, good Python SDK, traces capture exactly what we need.

### RAG Evaluation Metrics (RAGAS and Beyond)

**Two-stage evaluation for memory systems:**

| Stage | What to Measure | Metrics |
|-------|-----------------|---------|
| **Retrieval** | Did we get the right memories? | Precision, Recall, MRR, NDCG |
| **Generation** | Did we use them correctly? | Faithfulness, Answer Relevance, Hallucination Rate |

**RAGAS metrics we should adopt:**
- **Context Precision**: % of retrieved docs that are relevant
- **Context Recall**: % of relevant docs that were retrieved  
- **Faithfulness**: Does answer stick to retrieved context?
- **Answer Relevance**: Does answer address the question?

**Our adaptation for memory:**
- **Memory Precision**: % of retrieved memories that are relevant to question
- **Memory Recall**: % of memories that SHOULD have been retrieved that WERE retrieved
- **Cross-Session Linking**: For multi-session questions, did we aggregate correctly?
- **Temporal Accuracy**: For time-based questions, did we order correctly?

### LLM-as-Judge Best Practices

**From Eugene Yan's research:**

| Approach | When to Use | Reliability |
|----------|-------------|-------------|
| **Direct scoring** | Objective assessments (faithfulness, policy violations) | Medium |
| **Pairwise comparison** | Subjective assessments (tone, coherence) | Higher |
| **Reference-based** | When gold answer exists | Highest |

**Known biases to avoid:**
- Position bias (first option preferred)
- Verbosity bias (longer answers scored higher)
- Self-preference (model prefers its own outputs)

**Mitigation:**
- Swap position in pairwise comparisons
- Control for answer length
- Use different model for judging than generating

**For our memory evals:** Use reference-based (we have gold answers) with direct scoring for pass/fail.

---

## Part 10: Revised Architecture Decision

Based on research, **simplify the stack**:

### What We Actually Need

```
┌─────────────────────────────────────────────────┐
│           PersonaEval (Simple Python)            │
├─────────────────────────────────────────────────┤
│  1. Benchmark Loaders                           │
│     - BEAM, ConvoMem, PersonaMem, LongMemEval   │
│     - Just Python classes, no framework magic   │
├─────────────────────────────────────────────────┤
│  2. Memory System Adapters                      │
│     - Persona, Graphiti interfaces              │
│     - Abstract: ingest(sessions), retrieve(q)   │
├─────────────────────────────────────────────────┤
│  3. Provenance Tracking                         │
│     - SQLite: questions, sessions, memories,    │
│       retrievals, results                       │
│     - Langfuse for trace storage (optional)     │
├─────────────────────────────────────────────────┤
│  4. Evaluation Pipeline                         │
│     - LLM-as-judge with reference answers       │
│     - Binary pass/fail + critique               │
│     - RAGAS-style metrics for retrieval         │
├─────────────────────────────────────────────────┤
│  5. Error Analysis Tools                        │
│     - Custom annotation UI (build ourselves)    │
│     - Failure taxonomy auto-extraction          │
│     - Jupyter notebooks for exploration         │
├─────────────────────────────────────────────────┤
│  6. Visualization (Phase 2)                     │
│     - Question deep-dive view                   │
│     - Session timeline                          │
│     - Failure pattern heatmap                   │
└─────────────────────────────────────────────────┘
```

### What We're NOT Using

| Tool | Why Not |
|------|---------|
| **OpenBench** | Overkill - we don't need 95 benchmarks, just 4 |
| **DeepEval** | Good metrics but adds dependency; RAGAS patterns are simple to implement |
| **memorybench** | TypeScript, and we're building something more specialized |

### Core Principle: Just Write Python

```python
# This is all we need for a memory eval
@dataclass
class EvalResult:
    question_id: str
    question_text: str
    gold_answer: str
    generated_answer: str
    retrieved_memories: List[Memory]
    relevant_memories: List[Memory]  # post-hoc annotation
    correct: bool
    failure_category: Optional[str]
    critique: Optional[str]

def evaluate_question(q: Question, memory_system: MemoryAdapter) -> EvalResult:
    # 1. Ingest sessions
    memory_system.ingest(q.sessions)
    
    # 2. Retrieve memories
    retrieved = memory_system.retrieve(q.question_text)
    
    # 3. Generate answer
    answer = generate_answer(q.question_text, retrieved)
    
    # 4. Judge correctness (LLM-as-judge with reference)
    correct, critique = judge_answer(answer, q.gold_answer, q.question_text)
    
    # 5. Compute retrieval metrics
    relevant = identify_relevant_memories(q, retrieved)  # post-hoc
    
    return EvalResult(
        question_id=q.id,
        question_text=q.question_text,
        gold_answer=q.gold_answer,
        generated_answer=answer,
        retrieved_memories=retrieved,
        relevant_memories=relevant,
        correct=correct,
        failure_category=classify_failure(correct, retrieved, relevant) if not correct else None,
        critique=critique
    )
```

---

*This document consolidates: INTERNAL_LEARNINGS.md, EVAL_SYSTEM_PLAN.md, BENCHMARK_RESEARCH.md, EVAL_FRAMEWORK_RESEARCH.md*

*Supersedes all previous docs. Single source of truth.*

*Last research update: Dec 25, 2025 - Added Eugene Yan, Hamel Husain insights, simplified architecture.*
