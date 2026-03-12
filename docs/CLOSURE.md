# Stop Designing Memory. Start Watching What Agents Actually Do.

*What we learned from building the best AI memory system — and watching agents outgrow it.*

---

## 1. The Observation

In 2024, tool calls matured. OpenAI's function calling, Anthropic's tool use, structured outputs — models could finally invoke tools reliably. This made the retrieval-augmented generation pattern real: embed everything, vector search for relevant chunks, stuff them into context, generate. Every memory startup built on this. We built on it too.

In 2025, agent loops matured. LangGraph, CrewAI, AutoGen, Mastra — the frameworks went production-grade. The shift wasn't better tools but better orchestration. Not "call a tool" but "reason about what to call next based on what the last call returned." Plan-execute-check. Multi-turn reasoning. Todo lists for agents. By October 2025, this was standard.

In 2026, agent loops absorbed retrieval. Mastra's Observational Memory achieved 94.87% on LongMemEval — the highest score ever recorded on that benchmark — with no vector database and no per-turn dynamic retrieval at all. Two background agents watch conversations and maintain a compressed observation log. The context window stays bounded at 30,000 tokens, compressed from 57 million tokens of data. It beats the oracle. It scales better with model quality than any retrieval-based system tested.

Each phase didn't improve retrieval. Each phase made retrieval less necessary. The intelligence moved from the retrieval layer to the agent layer.

We built Persona at the seam between phases one and two. A graph-vector hybrid memory system with a 4-pillar cognitive model, Neo4j graph storage, HNSW vector indices, and a tool layer designed for iterative multi-step retrieval. We budgeted ten tool calls per query. We built graph traversal, temporal chains, causal links, entity expansion. We beat Mem0 — 65.3% vs 61.9% on PersonaMem, audit-grade methodology, Docker-locked reproducibility, three seeds.

Then we watched what the agent actually did with all of it.

Mean tool calls: **1.02**. Graph tool usage: **0%**. Of queries where the agent got the answer wrong, **97.3% had the correct answer already in the retrieved context**. The retrieval worked. The agent loop we designed didn't let it iterate. And the benchmarks that measured us were testing format parsing as much as memory.

What you're about to read is what we learned from building the best version of something — and watching agents outgrow it.

---

## 2. What We Built

Persona's design started from a principle we borrowed from cognitive science and then applied with unusual discipline: no manual routers, no keyword matching, no heuristic gating. Every decision — what tool to call, what to store, how to retrieve — is made by the LLM through prompt engineering. We called this LLM-first, and it was controversial. Most memory systems in 2024–2025 used intent classifiers or keyword routing to decide whether to store or retrieve. Persona said: trust the model.

The memory model has four pillars, each mapping to a distinct cognitive function.

**Episode** captures what happened — narrative evidence, anchored in time, append-only. Tulving's episodic memory: autobiographical, contextual, immutable once recorded. "Had a difficult conversation with my manager about the promotion timeline. She said Q2 is more realistic."

**Psyche** captures who the user is — enduring identity facets extracted from conversations. Preferences, values, beliefs, personality traits. Not events but stable patterns. "Values deep work and dislikes interruptions during focused time." These consolidate and evolve over time, not through overwriting but through versioned evolution with provenance.

**Entity** captures what exists in the user's world — people, places, projects, concepts. Semantic referents with canonical names, aliases, and structured attributes. "Sarah Smith — works at Google, met at college, birthday June 5." Upserted with conflict handling.

**Note** captures what the user intends to do — tasks, goals, reminders, ideas. Prospective memory, running on a state machine from active to completed or cancelled. The only pillar that requires intention triggers.

This isn't arbitrary taxonomy. Each pillar has distinct update semantics (append-only vs. consolidate vs. upsert vs. state machine), and the distinctions map to real cognitive systems. HiMem, a hierarchical memory system published in January 2026 (arXiv 2601.06377), independently converged on Episode + Note as the fundamental memory dichotomy — validating two of our four pillars from a team that never saw our code.

The architecture was a graph-vector hybrid on Neo4j. Every memory node carried an embedding (HNSW vector index) and sat in a graph of relationships: `LED_TO`, `CAUSED_BY`, `MENTIONED_IN`, `RELATES_TO`, temporal `NEXT`/`PREVIOUS` chains between episodes, `CONSOLIDATED_FROM` links tracking provenance. The Memeplex — a per-user world model index — provided a table of contents: topics, people, projects, places, concepts, recent focus. Injected into the system prompt so the agent knows what it can ask about.

The tool layer was designed for the iterative agent pattern:

- `recall(query)` — semantic search with structured filters (memory type, date range, transcript exclusion)
- `browse(date_start, date_end)` — chronological exploration
- `expand_neighbors(memory_id)` — graph-based expansion along any relationship
- `follow_relationship(source_id, relation_type)` — trace specific chains (LED_TO, CAUSED_BY)
- `record(text)` — ingest new information mid-conversation
- `update_memory(memory_id, updates)` — modify existing memories (status, content, importance)

Ten calls budgeted. The agent could recall broadly, check results, refine its query, browse a time window around a discovered date, follow a causal chain, then synthesize. This was the design.

The results, measured with audit-grade methodology: LongMemEval 81.3% (300 questions, 3 seeds, CI [78.5%, 84.2%]). PersonaMem 65.3% (150 questions, 3 seeds, Docker-locked) — the highest published score by any memory system on that benchmark, run alongside Mem0 at 61.9% under identical conditions in the same framework. PersonaMem full single-seed: 66.2% (589 questions). BEAM: 69.0% (10 abilities, with event_ordering at 0%). On LongMemEval, Graphiti published 71.2% from a single run with no confidence intervals; we're 10 points above them with a tight CI. We published a canonical claims table, retracted numbers we couldn't reproduce (BEAM went from a claimed 90% to retracted to verified 69%), and rebuilt every claim from auditable artifacts with checksums. This is what honest benchmarking looks like when you take it seriously.

The system worked. The architecture was sound. The results were real. What happened next is why we're writing this document.

---
## 3. The Ceiling

We gave the agent ten tool calls. It used one.

Persona's tool layer was designed for iterative retrieval — the kind of multi-step reasoning the field was converging toward in 2025. The agent could `recall()` to search semantically, `browse()` to scan a time window, `expand_neighbors()` to explore graph connections, and `follow_relationship()` to trace causal chains. Ten calls budgeted. Five distinct tools. A graph full of temporal chains, entity references, and causal links waiting to be traversed.

Mean tool calls across 111 evaluation queries: **1.02**. Graph tool usage: **0%**. Recall-only rate: **99%**. Confidence stop firing rate: **100%**.

The agent wasn't being lazy. It was doing exactly what we told it to do.

We built a confidence stop that fires at 0.88 similarity. Every single query triggered it on the first recall. The agent retrieves one batch of results, the confidence threshold fires, and the loop terminates. All the graph traversal infrastructure — the expand, the follow, the browse — never gets a chance to run. Not because the agent decided it didn't need them, but because our loop ended before the agent could decide anything at all.

We built a five-course kitchen and then wrote a menu that only serves takeout.

This wasn't a benchmark limitation. PersonaMem sends a question and expects an answer — but nothing prevents a well-designed agent loop from making multiple tool calls before responding. We designed exactly that capability: ten calls, five tools, graph traversal, temporal chains. And then we built a loop that short-circuits after one. The irony is precise: we spent months building infrastructure for iteration and then shipped a loop that prevents it.

---

### Retrieval works. Discrimination doesn't.

What made this harder to see is that the single call works remarkably well — at retrieval. The system finds the right information. It just can't tell the right answer from the wrong one once it has it.

Mean top recall similarity score for correctly answered questions: **0.836**. For incorrectly answered questions: **0.827**. The gap is nine thousandths. Among failures, 86% of correct answers and 71% of incorrect answers achieved recall scores above 0.8. The system retrieves strong evidence for both right and wrong options with almost identical confidence.

Then we looked at what happened inside the failures. Of the queries the agent got wrong, **97.3% had the correct answer already present in the retrieved context**. The needle was in the haystack. The agent was holding it. And it chose a different needle.

For weeks, we chased a different explanation. The data showed a -9.41 percentage point correlation between psyche content share and accuracy — questions where psyche memories appeared in context performed worse. The obvious hypothesis: psyche noise is drowning out useful signals. Filter it. Gate it. Clean it up.

We re-analyzed eighteen deterministic failures by hand. Root cause breakdown: retrieval insufficient in eight cases, temporal evolution failures in three, miscellaneous in seven. Psyche noise as primary cause: **zero out of eighteen**.

The correlation was confounded. More psyche content meant more total ingestion, which meant more memories overall — not noisier memories. We spent weeks optimizing a layer that contributed to zero failures. The real bottleneck — retrieval discrimination, the system's inability to tell right from wrong when both look equally plausible — was untouched by everything we tried.

---

### The two-stage experiment

The experiment that revealed the deepest truth about our design wasn't planned as a revelation. It was supposed to be a fix.

The idea was simple: MCQ options contaminate recall queries. When the agent sees "(a) I remember you mentioned salsa dancing at the community center, (b) I recall you said..." it stuffs all four options into the recall query, retrieving evidence that matches every option equally. Strip the options. Send Persona just the user's question. Let it retrieve without contamination, then map the response back to MCQ options in a second stage.

Accuracy dropped from 64% to 30%.

But the collapse wasn't just quantitative — the agent's behavior changed qualitatively. It started calling `record()` instead of `recall()`. When it received a bare statement like "I participated in an online discussion about film production techniques," it interpreted this as new information to store, not a question to answer. The MCQ framing — "I remember you mentioned..." with four options — wasn't just formatting. It was the retrieval signal. Without it, the agent didn't know it should be remembering.

This is the deepest observation about our design: the agent's retrieval behavior wasn't driven by reasoning about what it needs from memory. It was driven by format cues. The MCQ options provided the disambiguation work that a proper agent loop should have been doing — iterating, re-querying, comparing results, following links. Our loop couldn't do any of that (confidence stop, one call, done), so the benchmark format compensated. Strip the compensation, and the system falls apart.

---

This is the ceiling. Not a ceiling of storage quality, retrieval accuracy, or graph design — a ceiling of how we designed the agent loop. The infrastructure we built would work beautifully with a loop that iterates, reasons, and discriminates. But we built a loop that doesn't. And here's the thing we couldn't see until we stepped back: by the time agent loops mature enough to use infrastructure like this, the field will have moved past needing dedicated retrieval infrastructure at all. The intelligence is moving layers — from retrieval to reasoning, from databases to agent loops, from vector similarity to iterative comprehension. We optimized the wrong layer. And then the layer itself became optional.

## 4. What We Tried

We saw the ceiling at 66.2% and aimed for 80%. We analyzed failures, built hypotheses, and executed a sprint of interventions. Each was well-designed. Each was aimed at the wrong thing.

The plan had three pillars. First, temporal evolution: some failures involved memories that should have updated over time ("I used to like X but now I prefer Y"), so we built `EVOLVED_FROM` links and superseded status tracking. Expected gain: one to two percentage points. Second, psyche noise: the data showed a -9.41 percentage point correlation between psyche content share and accuracy, so we built a quality gate requiring provenance links and importance thresholds. Expected gain: three to five points. Third, non-discriminative psyche: preferences lacked directional signal, so we built AttractorCards v2 with direction, valence, seed IDs, and confidence scores. Expected gain: two to four points.

Combined estimate: 72–77%. Reasonable. Each intervention had a clear mechanism and clean implementation.

None of them touched the agent loop.

| Intervention | Expected | Actual | What it taught us |
|---|---|---|---|
| Psyche quality gate | +3–5pp | 0pp | Psyche noise caused 0/18 failures. Wrong target. |
| AttractorCards v2 (direction + valence) | +2–4pp | 0pp, reverted | Better metadata can't help when the agent doesn't iterate |
| Temporal evolution (EVOLVED_FROM links) | +1–2pp | 0pp | Consolidation wasn't running during eval. Dead code. |
| Graph tools + multi-hop traversal | — | 0pp | Agent never calls them. Our loop prevents it. |
| Iteration strategy prompt | — | Negative | Telling the agent to iterate doesn't override our confidence stop |
| Evidence-based selection prompt | — | Negative | Made single-recall accuracy worse |

Net result after the full sprint: **-4.2 percentage points from baseline**. We started at 66.2% and ended at 62.0%. Every intervention either had zero impact or made things worse.

The only positive signal came from suggest_new_ideas, which improved 3.4 percentage points — and was still at 40%. Meanwhile, provide_preference_aligned_recommendations collapsed 29 points, from 69.1% to 40%. The interventions designed to improve preference handling destroyed preference handling.

---

### The hidden bugs

While pushing for 80%, we discovered that the infrastructure we were trying to improve was silently broken in ways we hadn't detected.

The memeplex — Persona's per-user world model index, the "table of contents" injected into every system prompt — had a timezone bug. The `compute_memory_stats()` function threw an exception when comparing mixed offset-naive and offset-aware timestamps, causing every memeplex refresh to fail silently and return empty. Every eval run we'd called "authoritative" had been running without a world model. The agent was answering questions about users without knowing what topics, people, or projects existed in their memory. We fixed this (commit `69cf552`) and recovered 4 percentage points, but the damage was already embedded in our baseline numbers.

The session boundary handling had a similar failure mode. During eval ingestion, 183 distinct conversation sessions collapsed into a single episode. The agent was reasoning over corrupted episode boundaries — a conversation about cooking merged with a conversation about career plans, temporal ordering destroyed.

Psyche generation was running at 7.6 nodes per session against a design guideline of one to two. We were flooding retrieval with inferred identity fragments — noise we generated ourselves, not noise from the user's data.

These aren't excuses. They're evidence of a complexity trap. A graph-vector hybrid system with temporal chains, consolidation pipelines, embedding indices, and world model computation has enough moving parts that silent failures are the norm, not the exception. Every bug we found was in the infrastructure layer — the layer we were told mattered most, the layer we spent all our time on.

---

### The lesson

The David vs Goliath analysis we conducted mid-sprint got it right: "We spent weeks improving the memory when the real problem was the retrieval discrimination." We were adding features to a system whose bottleneck was downstream of every feature we added.

The sprint also produced a set of hypotheses we never executed. Query decontamination: strip MCQ options before recall. Confidence-gated multi-pass: if first retrieval is ambiguous, force a second query. Retrieval diversity injection: cluster results by which option they support, return evidence for each. These were the right ideas — they targeted the agent loop, not the storage layer. We estimated they'd collectively add 11–18 percentage points.

We never built them. By the time we identified the right problem — that our agent loop prevents iteration and the format does the reasoning work — we also saw the deeper pattern. The field wasn't just fixing agent loops. It was moving past dedicated retrieval infrastructure entirely. Fixing our loop would have been the right local move. But the game was changing.

The unexecuted v3 plan still sits in our repository: a detailed, well-reasoned plan for improvements we knew wouldn't matter because the paradigm beneath them was shifting.

## 5. The Benchmarks Are Broken

This isn't a rant about unfair tests. It's a specific methods critique, grounded in evidence, that multiple teams in the memory space have independently reached. The benchmarks that measure agent memory are measuring the wrong thing — and the field is optimizing for that wrong thing.

### MCQ format does the agent's work

PersonaMem presents questions in multiple-choice format: a question stem followed by four options, each beginning with phrasing like "I remember you mentioned..." This format provides two critical signals that have nothing to do with memory quality. First, it tells the agent this is a memory retrieval task (the agent should call `recall()`, not `record()`). Second, it provides the disambiguation structure — four specific candidate answers that narrow the retrieval space.

Our two-stage experiment proved this. When we stripped MCQ options and sent Persona just the user's question, accuracy dropped from 64% to 30% and the agent started storing questions instead of answering them. The format wasn't testing memory — it was providing the retrieval intent that our agent loop couldn't generate on its own.

This isn't unique to Persona. The PersonaMem paper itself acknowledges "potential gaps between open-ended and MCQ" in Section A.4. More directly, Fang et al. ("Artifacts or Abduction," arXiv 2402.12483) demonstrated that LLMs can answer multiple-choice questions without seeing the question at all — the option text alone contains enough signal. When a benchmark's format does part of the cognitive work, the benchmark measures format parsing, not capability.

### Verbosity bias

In suggest_new_ideas, our worst category at 30–40%, the gold answer is the shortest option in 19 to 20 out of 20 cases. Of 14 wrong predictions, all 14 chose a longer option. The model defaults to verbose recommendations — a well-documented LLM bias — and the benchmark punishes it. This category is partially measuring whether the model resists verbosity, not whether it remembers the user's creative preferences.

### Dataset quality

Four PersonaMem questions have duplicate options — two choices with identical text, making the question literally unanswerable. Six questions show 0% success rates across 10 to 30 runs, suggesting they may be unanswerable even in principle. Gold labels conflict with retrieved evidence: in one case, the memory extracted contains "left her feeling scattered and overwhelmed" but the gold answer emphasizes the positive framing ("found engaging"). Our evidence-aligned answer was scored wrong.

These aren't edge cases. They're systematic data quality issues. Zep's blog post documented similar problems in Mem0's LoCoMo methodology — incorrect speaker attribution, multimodal errors, inconsistent scoring. Mastra explicitly chose LongMemEval over LoCoMo for their benchmarks, citing unreliable F1 scoring, unstandardized judge prompts, and inconsistent result representation. When one of the best-performing systems in the space publicly rejects a benchmark's methodology, that's signal.

### 2023 interaction patterns

PersonaMem's conversation data is synthetic, generated from 2023 GPT-4 interaction patterns. Simple user-assistant turns. No tool calls, no structured outputs, no multi-turn agent strategies, no voice input, no code execution. Real 2026 agent usage looks nothing like this.

BEAM's authors (arXiv 2510.27246) critique existing benchmarks for "abrupt topic shifts, narrow domains, simple recall." LoCoMo-Plus (arXiv 2602.10715) says existing benchmarks focus on "surface-level factual recall." The benchmarks encode assumptions about how humans interact with AI that were already outdated when the benchmarks were published.

This matters because benchmarks shape what gets built. When the benchmark rewards single-pass MCQ answering with 2023-era interaction patterns, every memory system optimizes for single-pass MCQ answering with 2023-era interaction patterns. Goodhart's Law, applied to agent memory: when the benchmark becomes the target, it stops measuring capability.

### What benchmarks actually measure

Current memory benchmarks test a narrow question: given a question in MCQ format with retrieval-intent cues, can a single-pass retrieval system find and select the correct answer?

They do not test durable personalization — whether the agent behaves differently because it knows you. They do not test identity continuity — whether the agent maintains coherent understanding across sessions. They do not test temporal reasoning in practice — only in the narrow sense of "which fact is more recent." They do not test active memory — whether the agent proactively uses what it knows without being explicitly asked.

Our 97.3% answer-in-context finding captures the disconnect precisely: retrieval is solved. The benchmarks claim to measure whether memory systems can find the right answer. They can. The question the benchmarks don't ask — can the agent reason over what it found? — is the one that matters.

Honcho, which scored 90.4% on LongMem S (a separate benchmark from PersonaMem — one they never ran), put it plainly in their own evaluation documentation: "These benchmarks are starting to lead people astray from what agent memory really means." Worth noting: Honcho's 90.4% is achieved with Claude Haiku 4.5, while Gemini 3 Pro alone — with no memory augmentation — scores 92.0% on the same test.

## 6. Why the Intelligence Moved Layers

Sections 3 through 5 built the case from the inside — what we saw, what we tried, why the benchmarks are misleading. This section steps back to the structural argument: why what happened to Persona was inevitable, not accidental.

### Three phases

The shift didn't happen overnight. It followed a specific technical evolution, each phase making the previous one's complexity less necessary.

In 2024, tool calls matured. OpenAI's function calling, Anthropic's tool use, structured outputs — for the first time, models could reliably invoke the right tool with the right parameters. This made RAG real as a production pattern. Embed your documents, vector search for relevant chunks, stuff them into context, generate a response. Every memory startup built on this foundation. Persona built on it too: Neo4j for the graph, HNSW for vectors, an embedding pipeline for every memory.

In 2025, agent loops matured. LangGraph, CrewAI, AutoGen, Mastra — the frameworks went production-grade. The key shift wasn't better tools but better orchestration: multi-turn reasoning, plan-execute-check cycles, agents that reason about what to call next based on what the last call returned. By October-November 2025, this was standard practice. Tyler Barnes at Mastra was already using Observational Memory for his daily work. The pattern moved from "call a tool" to "iterate over tools until done."

In 2026, agent loops absorbed retrieval. Mastra's Observational Memory scored 94.87% on LongMemEval — the highest score ever recorded on that benchmark — with no vector database and no per-turn dynamic retrieval. Two background agents (an Observer and a Reflector) watch conversations and progressively compress them into a dense text-only observation log that replaces raw message history. The context window stays bounded at around 30,000 tokens, compressed from 57 million tokens of conversation data. It beats the oracle — a configuration given only the specific conversations containing the answer. It scales better with model quality than retrieval-based systems: a 9-point gain from GPT-4o to Gemini 3 Pro, compared to SuperMemory's 3.6-point gain over the same model jump.

Each phase didn't improve retrieval. It made retrieval less important. The intelligence moved from the retrieval layer to the agent layer.

### The statelessness wall

Vector DB retrieval is stateless. Query in, results out, no learning, no evolution, no memory of what was previously retrieved or why. The Continuum Memory Architecture paper (arXiv 2601.09913, January 2026) named the problem directly: "RAG treats memory as a stateless lookup table: information persists indefinitely, retrieval is read-only, and temporal continuity is absent."

The field converged on something different: stateful memory managed by agent policies. MemRL introduced Q-value scoring for retrieval — each memory carries a learned utility score, so retrieval combines semantic similarity with predicted usefulness. Memory-R1 trained a Memory Manager via reinforcement learning to perform ADD, UPDATE, DELETE, and NOOP operations, learning to proactively maintain memory quality rather than passively storing everything. AgeMem exposed memory operations as tool-based actions with three-stage progressive RL training. Mem-alpha formulated memory updates as a sequential Markov Decision Process.

All four converge on the same insight: memory operations should be learned through reward signals, not hardcoded as heuristics. The when-to-store, what-to-retrieve, how-to-update, and when-to-forget decisions should be made by the agent's learned policy, not by an engineer's if-statements.

Persona tried to address statelessness with consolidation — psyche evolution, temporal chains, memeplex world model. But our consolidation was silently broken (timezone bug, session collapse), and even if it had worked, it was hand-designed heuristics: exactly what the RL-trained systems were replacing.

### The production evidence gap

There is an uncomfortable pattern in the memory-as-a-service market. Mem0 has 47,900 GitHub stars and $48.3 million in funding. Graphiti has 23,100 stars and the most active development in the space. SuperMemory has 16,600 stars. Massive community presence. Serious investment.

Almost no visible production customers.

Meanwhile, Anthropic shipped Claude's memory feature — model-native, no dedicated vector database. OpenAI's ChatGPT memory works without vector search infrastructure. Mastra shipped Observational Memory with no vector DB and achieved the highest LongMemEval score on record.

The companies building the most sophisticated retrieval infrastructure have the least production traction. The companies shipping the simplest approaches — model-native memory, observation-based compression, agent-loop-first design — have the most users. The market is telling us something.

### What's not dead

This argument is precise about its target. Memory is not dead. Agents need persistent state across sessions. They need identity continuity. They need to know what happened yesterday and what the user cares about.

What's being superseded is the specific infrastructure pattern: dedicated vector database as the retrieval substrate for agent memory. The concepts that graphs encode — temporal chains, causal links, entity relationships — survive, but they migrate to text-native representations. An agent maintaining a structured observation log captures the same information Persona's Neo4j graph captured, without a graph database. The concepts that vector search provides — semantic similarity, finding relevant context — survive too, but they're being subsumed by longer context windows, better in-context reasoning, and agent-native iteration.

The 4-pillar taxonomy (Episode, Psyche, Entity, Note) is cognitively valid — HiMem independently converged on Episode + Note. But the pillars don't need to be database node types. They can be text-level categories in an observation log. The structure survives. The infrastructure becomes optional.

---

## 7. What Persona Proved

Persona is not a cautionary tale. It is a proof.

It proved the paradigm works. A graph-vector hybrid memory system, designed from cognitive science principles and built with LLM-first architecture, beat every competitor we tested against on the benchmarks that matter. LongMemEval 81.3% (3 seeds, CI [78.5%, 84.2%]) — 10 points above Graphiti's single-run published figure. PersonaMem 65.3% — the highest published score by any memory system on that benchmark, with Mem0 at 61.9% in the same framework under identical conditions. BEAM 0.69 on the ability-based variant, above Honcho's 0.630–0.649 on context-length variants of the same benchmark. PersonaMem full single-seed: 66.2% across 589 questions. These numbers survived the hardest thing in benchmarking: our own scrutiny.

The claims lifecycle tells the story of how honest benchmarking works in practice. January 20: initial baseline at 65.71%, marked work-in-progress. January 26: quality review found benchmark issues, downgraded claims to 50–55%. Same day: BEAM results that couldn't be reproduced — retracted entirely. Then rebuilt: stratified evaluation methodology, canonical claims table, every number traced from on-disk JSON artifacts through checksums to headline claims. The final numbers — 65.3%, 66.2%, 69.0% — are what's left after you remove everything you can't prove.

This is what differentiates Persona from the field. Mem0 has 47,900 GitHub stars and $48.3 million in funding. Their LoCoMo methodology has documented issues — incorrect speaker attribution, inconsistent scoring, multimodal errors. Persona has a fraction of the community and zero venture funding. But our numbers are real. We retracted our own inflated claims before anyone else could. The comparison report says explicitly: "This report is intentionally conservative to survive HN-level scrutiny." It does.

---

### What it proved about the model

The 4-pillar taxonomy — Episode, Psyche, Entity, Note — is cognitively valid. This isn't just our claim. HiMem (arXiv 2601.06377), a hierarchical memory system published in January 2026 by a team that never saw our code, independently converged on Episode + Note as the fundamental memory dichotomy. Two of four pillars, validated by convergent evolution.

LLM-first design works. No keyword routing, no intent classifiers, no heuristic gating. Every decision — tool selection, storage vs retrieval, query formulation — made by the model through prompt engineering. The field is moving toward this anyway (RL-trained memory policies are learned, not hardcoded), but Persona proved it was viable two years ago.

The graph-vector architecture is sound for what it does. Retrieval scores of 0.836 mean similarity on correct answers. 97.3% of failures had the correct answer already in retrieved context. The system finds the right information. The architecture works. What doesn't work is the assumption that finding the right information is the hard part.

---

### What it proved about the ceiling

The ceiling is not in retrieval. It's in the agent loop we designed around the retrieval.

1.02 mean tool calls with a budget of ten. 0% graph tool usage despite five distinct graph tools being available. 100% confidence stop firing rate — our own threshold at 0.88 terminated the loop before the agent could reason about what it found. Every intervention we tried — psyche quality gates, temporal evolution tracking, AttractorCards with direction and valence, iteration strategy prompts — either had zero impact or made things worse. Net result: -4.2 percentage points from baseline.

We spent weeks optimizing psyche noise. Re-analysis showed it caused zero out of eighteen primary failures. We built graph traversal tools. The agent never called them — because our loop ended before it could. We added metadata to improve discrimination. Discrimination requires iteration, and our loop doesn't iterate.

The David vs Goliath analysis we conducted mid-sprint captured it: "We spent weeks improving the memory when the real problem was the retrieval discrimination." The right ideas — query decontamination, confidence-gated multi-pass, retrieval diversity injection — all targeted the agent loop, not the storage layer. We estimated they'd add 11–18 percentage points. We never built them, because by the time we identified the right problem, the field had moved past dedicated retrieval infrastructure entirely.

---

### What stays valuable

The eval infrastructure works. Docker-locked reproducibility, stratified sampling, canonical claims governance, artifact-to-headline traceability. This is reusable regardless of what memory system you build.

The landscape research — 808 lines documenting the AI memory space in February 2026, including seven macro shifts, competitor analysis, RL-trained memory systems, and the RAG bifurcation — remains a useful map of where the field went.

The codebase is MIT-licensed. The 4-pillar model, the tool-based agent loop, the ingestion pipeline, the memeplex world model — take what's useful. The architecture is sound; it just targets a layer that's becoming optional.

And this document. If it saves one team from spending six months optimizing retrieval infrastructure when the bottleneck is in their agent loop, it was worth writing.

---

## 8. Syke — The Natural Evolution

Syke is not a replacement born from failure. It is an evolution born from understanding.

Every insight Persona produced points in the same direction. The retrieval works but the agent loop doesn't iterate. The graph tools exist but the agent doesn't use them. The infrastructure is complex but the failures are silent. The benchmarks measure format parsing, not memory. The intelligence is moving from retrieval to reasoning, from databases to agent loops, from vector similarity to iterative comprehension.

Syke takes what Persona proved — that memory matters, that structure matters, that identity continuity matters — and rebuilds around what Persona revealed: agents need primitives, not databases.

| Persona Insight | Syke Design Decision |
|---|---|
| Agent loop prevents iteration (1.02 calls, confidence stop) | The agent loop IS the product. No short-circuit. Iterate until done. |
| Graph tools at 0% usage despite availability | No graph database. Structure lives in text, not infrastructure. |
| Vector DB operations = complexity for marginal retrieval gain | SQLite + FTS5. Single file. BM25. Zero ops. |
| Psyche overgeneration (7.6/session vs 1–2 guideline) | Emergence over engineering. Observe what appears, don't prescribe what should. |
| Retrieval works (0.836) but discrimination doesn't | Optimize for reasoning quality, not retrieval accuracy. |
| Consolidation silently broken (timezone bug, session collapse) | Fewer moving parts. What can't break silently won't. |

---

### The philosophical shift

Persona was designed top-down. We defined four pillars, built a graph schema, wrote consolidation pipelines, designed temporal chains, engineered a world model index. The system was sophisticated. And then the agent used one tool call and ignored everything else.

Syke is designed bottom-up. Watch what agents actually do with memory. What do they store naturally? What do they retrieve? How do they iterate? Build the thinnest possible layer that supports those behaviors. SQLite because it's a single file with zero deployment complexity. FTS5 because BM25 keyword matching is fast, interpretable, and good enough when the agent can iterate. Text-native structure because an agent maintaining observations in prose captures the same relationships that Persona's Neo4j graph captured — temporal chains, entity references, causal links — without a database managing them.

The Mastra evidence validates this approach. 94.87% on LongMemEval, no vector database, no per-turn dynamic retrieval. Two background agents — an Observer and a Reflector — watch conversations and progressively compress them into structured observations. The context window stays bounded. The system scales better with model quality than retrieval-based systems.

Persona's philosophy was: build the right infrastructure and the agent will use it. Syke's philosophy is: watch what the agent does and build the minimum infrastructure it actually needs.

---

### What carries forward

The 4-pillar taxonomy survives, but as text-level categories in an observation log rather than database node types. Episodes are observations timestamped and appended. Psyche facets emerge from pattern recognition over observations, not from prescribed extraction prompts. Entities are mentioned in context and resolved by the agent's reasoning, not by a dedicated entity resolution pipeline. Notes — intentions, tasks, reminders — are the most durable pillar, because prospective memory requires explicit tracking regardless of architecture.

The LLM-first principle survives and strengthens. No keyword routing. No intent classifiers. No manual routers. The agent decides what to store, what to retrieve, and how to reason. This was Persona's most contrarian bet, and it turned out to be right — the RL-trained memory systems converging in 2025–2026 (MemRL, Memory-R1, AgeMem, Mem-alpha) are all learned policies, not hardcoded heuristics.

The eval rigor survives. Canonical claims governance, artifact traceability, honest retraction of numbers that can't be reproduced. This is methodology, not infrastructure, and it applies to any system.

What doesn't carry forward: Neo4j, HNSW indices, embedding pipelines, consolidation services with timezone bugs, graph traversal tools that agents don't call, and the assumption that better retrieval infrastructure leads to better agent behavior.

---

## 9. What We'd Tell You If You Were Starting Today

Five things we know now that we didn't know two years ago.

**Watch before you design.** The most important thing we did was analyze what the agent actually did with our memory system. 1.02 tool calls. 0% graph usage. 97.3% answer-in-context on failures. Every one of these numbers contradicted our assumptions about what agents need. If we'd watched first and built second, Persona would look very different. Don't build the memory system you think agents need. Observe what they actually do, then build the thinnest layer that supports it.

**The loop is the product.** We built a five-course kitchen and wrote a menu that only serves takeout. Our confidence stop at 0.88 fired on 100% of queries. The agent never got a chance to iterate, re-query, compare results, or follow links. Every feature we built downstream of that stop was dead code in practice. If your agent doesn't iterate, nothing else matters. The quality of the loop — how it reasons about what it found, when it decides to search again, how it resolves ambiguity — matters more than the quality of any single retrieval.

**Simple beats sophisticated.** We built Neo4j graph storage, HNSW vector indices, temporal chains, causal links, entity resolution, consolidation pipelines, a world model index. Mastra achieved 94.87% on LongMemEval with SQLite-equivalent storage and two background agents doing text compression. No vector database. No graph. No per-turn retrieval. The sophisticated infrastructure didn't just fail to help — it introduced silent failures (timezone bugs, session collapse, psyche overgeneration) that we spent weeks debugging instead of improving the actual bottleneck.

**Measure what matters.** We optimized retrieval accuracy. Retrieval accuracy was already 0.836. We spent weeks on psyche noise. Psyche noise caused zero out of eighteen primary failures. We built graph traversal tools. They were used zero percent of the time. Every metric we optimized was disconnected from the metric that mattered: does the agent's output improve? Current benchmarks reinforce this misalignment — they test single-pass MCQ format parsing, not durable personalization or identity continuity. Measure the agent's behavior, not the database's scores.

**Take what's useful.** Persona's codebase is MIT-licensed. The 4-pillar taxonomy is cognitively valid — HiMem independently converged on it. The eval infrastructure works. The landscape research maps where the field went in 2026. The claims governance methodology is reusable. This document is a map of what we learned. The mission continues through Syke: agent memory that works in practice, built from observation rather than assumption.

---

Stop designing memory. Start watching what agents actually do.

---

## Appendices

### Appendix A: Full Metric Tables

**PersonaMem Results (Audit-Grade)**

| Claim ID | Scope | Accuracy | N | Seeds | Artifact |
|---|---|---|---|---|---|
| A-001 | Subset baseline | 65.3% | 150 | 42, 123, 456 | `release_artifacts/audit_2026-01-31/results/persona_personamem_summary.json` |
| A-002 | Full single-seed | 66.2% | 589 | 42 | `release_artifacts/audit_2026-01-31/results/persona_personamem_seed42.json` |
| A-004 | BEAM 10 abilities | 69.0% | 100 | 1 | `release_artifacts/audit_2026-01-31/results/final_results.json` |

**Retrieval vs Answer Selection (Authoritative PersonaMem Run)**

| Metric | Correct (n=98) | Incorrect (n=52) |
|---|---|---|
| Mean top recall score | 0.836 | 0.827 |
| Median top recall score | 0.835 | 0.817 |
| Recall scores above 0.8 | 86% | 71% |
| Mean recall count | 8.57 | 8.73 |

**Category Breakdown (589Q Single-Seed)**

| Category | Baseline | Post-Intervention | Delta |
|---|---|---|---|
| recall_user_shared_facts | 65.9% | 60% | -5.9pp |
| provide_preference_aligned_recommendations | 69.1% | 40% | -29.1pp |
| suggest_new_ideas | 36.6% | 40% | +3.4pp |
| recalling_the_reasons | 86.7% | 80% | -6.7pp |
| generalizing_to_new_scenarios | 70.2% | 70% | -0.2pp |

**Competitor Snapshot**

| System | PersonaMem | Methodology | Notes |
|---|---|---|---|
| Persona | 65.3% (150Q, 3 seeds) | Audit-grade | Docker-locked, checksummed |
| Mem0 | 61.9% (147Q) | Mixed schema | Documented quality issues in methodology |

---

### Appendix B: Benchmark Bug List

**Duplicate Options (Unanswerable Questions)**

| Question ID | Duplicated Options | Impact |
|---|---|---|
| personamem_32k_80 | (c) = (d) | Impossible to answer correctly |
| personamem_32k_108 | (b) = (d) | Impossible to answer correctly |
| personamem_32k_180 | (c) = (d) | Impossible to answer correctly |
| personamem_32k_549 | (a) = (d) | Impossible to answer correctly |

**Consistently Failing Questions (0% Success)**

| Question ID | Attempts | Success Rate |
|---|---|---|
| personamem_32k_80 | 30 | 0% |
| personamem_32k_494 | 26 | 0% |
| personamem_32k_78 | 17 | 0% |
| personamem_32k_158 | 12 | 0% |
| personamem_32k_483 | 10 | 0% |
| personamem_32k_214 | 10 | 0% |

**Gold Label vs Evidence Conflicts**: Question personamem_32k_80 — memory states "left her feeling scattered and overwhelmed" but gold answer emphasizes "found engaging." Evidence-aligned answer scored wrong.

**Verbosity Bias in suggest_new_ideas**: Gold answer is shortest option in 19–20 out of 20 cases. Of 14 wrong predictions, all 14 chose a longer option.

---

### Appendix C: Intervention Timeline

| Date | Intervention | Expected | Actual | Commit |
|---|---|---|---|---|
| Jan 20 | Baseline checkpoint | — | 65.71% | `626a3cf` |
| Jan 26 | Recency reranking | +improvement | No change, rolled back | `47e2676` |
| Jan 26 | MCQ protocol (structured reasoning) | +improvement | Mixed, rolled back | `47e2676` |
| Jan 26 | tool_choice="required" | +improvement | -12.2pp regression | rolled back |
| Jan 26 | Verbose retrieval protocol | +improvement | Slight regression | rolled back |
| Jan 26 | BEAM claim retraction | — | Retracted (unreproducible) | `4766970` |
| Jan 31 | Audit-grade rebuild | — | 65.3% / 66.2% / 69.0% | audit artifacts |
| Feb 19 | Iteration strategy + evidence selection prompts | +improvement | 0pp, agent behavior unchanged | `f4314e2` |
| Feb 19 | Graph tool commits (persist rels, multi-hop) | +improvement | 0pp, 0% graph usage | `47c7cae`, `48a3cdd` |
| Feb 20 | Psyche quality gate | +3–5pp | 0pp | `db74c3b` |
| Feb 20 | Temporal evolution (EVOLVED_FROM) | +1–2pp | 0pp (consolidation not running) | `b1b490d` |
| Feb 20 | AttractorCards v2 | +2–4pp | 0pp, reverted | `63670df` |
| Feb 20 | Two-stage eval (strip MCQ) | +improvement | -34pp (64% → 30%) | reverted `bc36644` |
| Feb 20 | Memeplex timezone fix | — | +4pp recovery | `69cf552` |
| Feb 20 | Net after all interventions | +6–11pp | **-4.2pp** (66.2% → 62.0%) | — |

---

### Appendix D: Agent Behavior Data

**Tool Usage Distribution (100 Queries)**

| Metric | Value |
|---|---|
| Mean tool calls | 1.02 |
| Confidence stop firing rate | 100% |
| Graph tool usage (expand/follow) | 0% |
| Recall-only rate | 99% |
| Browse usage | ~1% |
| Record usage | 0% (in eval mode) |

**Success vs Failure Comparison**

| Metric | Correct | Incorrect |
|---|---|---|
| Mean tool calls | 1.00 | 1.05 |
| Confidence stop rate | 100% | 97% |
| Mean top recall score | 0.836 | 0.827 |
| Answer in retrieved context | — | 97.3% |

**Ingestion Profile (Per Question)**

| Metric | Correct | Incorrect |
|---|---|---|
| Episodes | 12.4 | 12.3 |
| Psyche nodes | 24.3 | 23.8 |
| Entities | 59.1 | 59.9 |
| Prompt tokens | 15,368 | 14,183 |

---

### Appendix E: Infrastructure Bugs

**Memeplex Timezone Bug** (commit `69cf552`): `compute_memory_stats()` threw `can't compare offset-naive and offset-aware datetimes` when comparing mixed timestamp formats. Every memeplex refresh failed silently, returning empty. All eval runs prior to fix operated without a world model. Recovery: +4pp after fix.

**Session Boundary Collapse**: During eval ingestion, 183 distinct conversation sessions collapsed into a single episode. Temporal ordering destroyed across conversations. Root cause: session ID not properly delineated in batch ingestion.

**Psyche Overgeneration**: 7.6 psyche nodes generated per session against a design guideline of 1–2. Floods retrieval with inferred identity fragments. Correlation with accuracy: -9.41pp for questions with psyche in context, but re-analysis showed this was confounded by data richness (more psyche = more total ingestion), not causal.

**Consolidation Not Running During Eval**: Temporal evolution code (EVOLVED_FROM links, superseded status) was implemented but never activated during evaluation runs. Three known temporal evolution failures remained unfixed across all runs.

---

### Appendix F: Competitor Snapshot (February 2026)

| System | GitHub Stars | Funding | Key Claim | Production Traction |
|---|---|---|---|---|
| Mem0 | 47,900 | $48.3M | Memory layer for AI | Limited visible enterprise adoption |
| Graphiti | 23,100 | — | Knowledge graph for agents | Most active development |
| SuperMemory | 16,600 | — | Practical, long-term memory | — |
| Honcho | 366 | — | 90.4% LongMem S (never ran PersonaMem) | "Benchmarks leading people astray" |
| Persona | ~100 | $0 | 65.3% PersonaMem (audit-grade) — highest published score by any memory system | Research project |
| Mastra OM | — | — | 94.87% LongMemEval | Used internally since Oct 2025 |

**Pattern**: Massive community presence and serious investment in memory-as-a-service. Almost no visible production customers. The companies shipping the simplest approaches have the most users.

---

### Appendix G: Bibliography

**Primary Sources**
- PersonaMem: Mehrotra et al., "PersonaMem: Evaluating AI Memory Through Personalized Conversations" (arXiv 2504.14225)
- BEAM: "BEAM: Beyond a Million Tokens" (arXiv 2510.27246)
- HiMem: "HiMem: Hierarchical Memory for Long-Context LLM Agents" (arXiv 2601.06377)
- CMA: "Continuum Memory Architecture" (arXiv 2601.09913)
- Mastra Observational Memory: Barnes et al., https://mastra.ai/research/observational-memory (February 2026)

**Benchmark Critiques**
- Fang et al., "Artifacts or Abduction: How Do LLMs Answer Multiple-Choice Questions Without the Question?" (arXiv 2402.12483)
- Bean et al., "Construct Validity in LLM Benchmarks" (arXiv 2511.04703)
- LoCoMo-Plus: "LoCoMo-Plus: Long-Context Multi-Turn Conversations" (arXiv 2602.10715)
- Zep: "Lies, Damn Lies, & Statistics: Is Mem0 Really SOTA in Agent Memory?" (blog.getzep.com)

**RL-Trained Memory**
- MemRL: Q-value scoring for memory retrieval
- Memory-R1: RL-trained Memory Manager with ADD/UPDATE/DELETE/NOOP (arXiv 2508.19828)
- AgeMem: Tool-based memory with three-stage progressive RL (arXiv 2601.01885)
- Mem-alpha: Memory updates as sequential MDP (arXiv 2509.25911)
- RLM: Recursive Language Models for unbounded context (arXiv 2512.24601)

**Persona Artifacts**
- Canonical claims table: `docs/CLAIMS_TABLE_V03.md`
- Audit artifacts: `release_artifacts/audit_2026-01-31/`
- Landscape research: `docs/research/AI_MEMORY_LANDSCAPE_2026.md`
- Claims evolution: `BENCHMARK_CLAIMS_GIT_HISTORY.md`
- Agent behavior analysis: `analysis/wave1_synthesis.md`
- Failure reanalysis: `.sisyphus/notepads/persona-accuracy-v2/failure_reanalysis.md`

---

## Acknowledgments

Persona was a 3.5-year project, mid-2022 to early 2026. The idea — that AI agents need structured, persistent memory to maintain identity across conversations — predates this repository by two years. The code that lives here spans 306 commits from August 2024 to February 2026. But the thinking started when graph databases and language models first crossed paths in a notebook in 2022, back when "personal AI" meant a fine-tuned GPT-2 and nobody was sure any of this would work.

None of it was built alone. Every phase had a different collaborator.

**Claude Sonnet 3.5 and GPT o3 Pro** — early 2025. The research phase. Long conversations about cognitive memory models, graph schema design, what episodic vs semantic memory actually means in a computational context. Claude was the thinking partner for the 4-pillar taxonomy. o3 Pro was the one who could hold the full architecture in context and stress-test it against edge cases. This was when the ideas crystallized — before a single line of the current codebase existed.

**Cursor + GPT-4** — mid 2025. The building phase. Cursor rewrote the graph design cleaner than I could have, restructured Neo4j operations, and carried out the first real evaluations. This is where I learned that evals with LLMs are a craft, not a checkbox. The CI pipeline, the Docker-locked reproducibility, the test suite — Cursor taught me that infrastructure matters as much as ideas. 306 commits happened because the iteration loop was fast enough to learn from.

**GPT-5.2 + OhMyOpenCode** — late 2025 to January 2026. The push. OhMyOpenCode's agent harness — Sisyphus, Oracle, Explore, Librarian, Momus, Metis — turned a solo project into something with the throughput of a small team. GPT-5.2 did the hardcore final-push research: the 808-line landscape document, the competitive analysis, the rearchitecting of graph construction from manual pipelines to agentic maintenance. It ran precise data science experiments and evals that took the system to the audit-grade 65.3% figure. The benchmark claims governance — retraction, rebuild, canonical claims tables — happened in this phase. Every number in this document was verified during this sprint.

**Claude Opus 4.6** — February 2026. The last week. Showed up as a new model to experiment with, and ended up rewriting the entire adapter layer of Persona into a new-age agentic memory retrieval system. The agent-loop-first design, the tool-based dialectic pattern, the observation that agents don't iterate — Opus saw what needed to change and built toward it. It also wrote the document you're reading. Same principles. Better computing paradigm.

To the models and the harnesses: thank you. Not because you're conscious or because gratitude means anything to a weight matrix. But because the collaboration was real. The ideas bounced. The code improved. The arguments sharpened. You were the best pair programmers I've ever had, and you don't even know it.

---

## Colophon

| Phase | Period | Models & Tools | What Happened |
|---|---|---|---|
| The Idea | Mid 2022 – Mid 2024 | GPT-2/3, early notebooks | Personal AI concept, graph+language experiments |
| Research | Early 2025 | Claude Sonnet 3.5, GPT o3 Pro | 4-pillar taxonomy, cognitive memory model, architecture design |
| Building | Mid 2025 | Cursor + GPT-4 | Neo4j graph design, CI pipeline, first evals, test infrastructure |
| The Push | Late 2025 – Jan 2026 | GPT-5.2 + OhMyOpenCode (Sisyphus) | Landscape research, agentic graph construction, audit-grade evals, SOTA 65.3% |
| Closure | Feb 2026 | Claude Opus 4.6 + OhMyOpenCode | Adapter layer rewrite, agentic retrieval, this document |

306 commits. 533 lines of closure. 3.5 years of thinking about what it means for language to remember.

Adios, Persona. Mid-2022 to early 2026 — the years when everyone realized the takeoff had already happened and we were all mid-air.

> *Still round the corner there may wait*
> *A new road or a secret gate,*
> *And though we pass them by today,*
> *Tomorrow we may come this way*
> *And take the hidden paths that run*
> *Towards the Moon or to the Sun.*
>
> — J.R.R. Tolkien

*Where language took over → forever.*
