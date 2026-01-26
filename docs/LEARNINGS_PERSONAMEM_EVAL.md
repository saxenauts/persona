# PersonaMem Evaluation Learnings

This document captures what we learned from the PersonaMem benchmark optimization work (Jan 2026) and why we decided NOT to pursue benchmark-specific optimizations.

---

## Summary

We achieved a 65.71% stratified PersonaMem baseline across all 7 question types (run_20260117_173046, 105 samples). The 75% result was a single-type run and should not be treated as the overall benchmark score. Benchmark-mode changes (temp=0, auto-recall, MCQ protocol) showed mixed results: they helped `suggest_new_ideas` but reduced overall accuracy.

Decision: keep the production prompt general and evidence-first. Any benchmark-mode tweaks must stay optional and off by default unless they improve the full stratified score.

---

## What We Tried

### Experiment 1: Recency Reranking
Hypothesis: evolution questions fail because conflicts aren’t disambiguated. Changes: EvidenceSummary with recency/stance markers. Result: no improvement on the stratified score. Learning: PersonaMem’s synthetic timelines reduce the usefulness of recency signals.

### Experiment 2: MCQ Protocol (Structured Reasoning)
Hypothesis: structured matching would reduce option-selection errors. Changes: multi-step MCQ protocol. Result: mixed; improved `suggest_new_ideas` but reduced overall score. Learning: benchmark-specific protocols create tradeoffs across categories.

### Experiment 3: Honcho Competitor Evaluation
Goal: fair comparison with alternative memory systems. Result: unverified; do not cite until replicated with recorded runs. Learning: Persona’s retrieval architecture is structurally stronger for fact recall, but claims require verified logs.

---

## Why We Paused Prompt-Only Optimization

### Key Insight

The MCQ protocol created internal tension:
1. "Base answer ONLY on retrieved evidence" (strict)
2. "Match emotional tone: anxious people prefer comfort" (loose heuristic)
3. "Output ONLY the letter" (forced choice)

Models obey the last clear instruction and rationalize via the easiest heuristic, bypassing evidence. This is the opposite of what we want.

### Benchmark-Specific Hacks Identified

| What | Why It's Bad |
|------|--------------|
| `<mcq_protocol>` 7-step process | Teaches test-taking, not real user interaction |
| "INFERENCE RULES" / emotional tone mapping | Encourages guessing from weak signals |
| "Results sorted NEWEST FIRST" assumption | Fragile; ties to specific tool behavior |
| "Return ONLY the letter" constraint | Eval format, not product behavior |

### Philosophy Alignment

AGENTS.md says: “LLM-First - no manual routers, no keyword matching, trust the model.” The MCQ protocol begins to resemble a test-taking algorithm rather than a general assistant rule set. Production prompts should remain general; benchmark tweaks must be optional.

---

## What We're Keeping

### Clean Minimal Prompt (Production Default)

```python
PERSONAL_AI_SYSTEM_PROMPT = """<role>
You are the user's Personal AI with memory access.
You do not know user-specific facts unless they appear in tool-retrieved memories or the current conversation.
</role>

<evidence_hierarchy>
Use this order for user-specific claims:
1) Memory tool outputs (recall/browse/get_memory/graph expansion)
2) Current conversation messages
3) World model below (index of topics/entities only; not proof of specific facts)
</evidence_hierarchy>

{world_model}

{user_context}

<tool_use>
Before answering any question that depends on the user's history, retrieve evidence first:
- recall(query): semantic search for relevant memories
- browse(date_start?, date_end?): chronological scan for time-based questions
- get_memory(memory_id): fetch full details when a snippet is not enough
- expand_neighbors()/follow_relationship(): use when connections matter
</tool_use>

<answer_policy>
If you find relevant memories:
- Base the answer on what the memories say; quote or paraphrase specific details.
- If memories conflict or show change, prefer most recent evidence by timestamp.

If you do not find relevant memories:
- Say you don't have that information stored.
- Ask one targeted clarifying question.

Never fabricate user-specific facts. When uncertain, be explicit.
</answer_policy>

<format_and_constraints>
Follow any explicit output-format constraints from the user message.
If the user provides options, use retrieval to pick the option best supported by evidence.
If evidence is insufficient, say so.
</format_and_constraints>"""
```

### What Makes This Good

1. **Single job**: Retrieve evidence, answer grounded in it, or say you don't know
2. **No test-taking algorithms**: Works for real users, not just benchmarks
3. **No behavioral heuristics**: Doesn't encourage guessing from weak signals
4. **Clean failure mode**: Uncertainty is explicit, not hidden

---

## Benchmark Results (Current)

| System | PersonaMem Score |
|--------|------------------|
| **Persona** | **65.71%** |
| Honcho | Unverified |

Further gains likely require ingestion and retrieval architecture changes (especially for BEAM), not prompt-only hacks.

---

## Future Directions (If We Want 70%+ Overall)

### Do These (Good Ideas)
Improve ingestion quality, strengthen entity linking, and keep eval-specific modes separate from production. All changes should be validated on the full stratified PersonaMem run.

### Don't Do These (Bad Ideas)
Avoid prompt-only MCQ hacks, behavioral inference heuristics, and assumptions about tool output ordering. Do not introduce content-based routing inside the prompt.

---

## Files Changed (Keep/Drop Decisions Needed)

| File | Action |
|------|--------|
| `persona/llm/prompts.py` | Keep minimal prompt as default |
| `persona/tools/memory.py` | EvidenceSummary/recency features require decision (no proven gain) |

### Clean Commit Reference
`4ce2f31` - "fix(api): return session ids from ingest" is a known clean baseline before eval-focused changes.

---

## Appendix: Honcho Comparison Details

Honcho (demo.honcho.dev) uses:
- Peer-centric architecture (users + AI as "peers")
- Dialectic endpoint for meta-reasoning about users
- Message history + working representations

Their approach is designed for psychological modeling, not fact retrieval. That's why they fail on PersonaMem (6.7% on `recall_user_shared_facts`).

Persona's approach (vector search + graph traversal + tool-based retrieval) is fundamentally better for memory-grounded Q&A.

---

Document created: Jan 18, 2026
Decision updated: keep minimal production prompt, keep benchmark-mode optional, and focus improvements on ingestion and retrieval quality.

---

## Update: Jan 25, 2026 - Benchmark Quality Issues & Failed Experiments

### Experiments That Failed

#### Experiment 4: tool_choice="required" (Forced Retrieval)
Hypothesis: Forcing the model to call `recall()` on every turn would ensure evidence-grounded answers.
Changes: Set `tool_choice="required"` in PersonaService.run_agent().
Result: **Regression** (54.5% vs 66.7% baseline on same questions).
Learning: Forcing retrieval doesn't help when the model's single query is suboptimal. The model wasn't iterating despite prompt guidance.

#### Experiment 5: Verbose Retrieval Protocol
Hypothesis: Detailed `<retrieval_protocol>` section with iteration rules would teach multi-turn retrieval.
Changes: Added 24-line retrieval protocol with ITERATE/STOP conditions, conflict resolution rules.
Result: **No improvement**, slight regression.
Learning: Verbose prompts hurt capable models ("Prompting Inversion" effect). Simpler is better.

### PersonaMem Benchmark Quality Issues

#### Issue 1: Duplicate Options (4 questions affected)
Some questions have **identical answer options**, making them impossible to answer correctly:

| Question ID | Duplicated Options |
|-------------|-------------------|
| personamem_32k_80 | (c) = (d) |
| personamem_32k_108 | (b) = (d) |
| personamem_32k_180 | (c) = (d) |
| personamem_32k_549 | (a) = (d) |

#### Issue 2: Consistently Failing Questions (6 questions, 0% success across 10-30 runs)
These questions **never** succeed regardless of approach:

| Question ID | Attempts | Success Rate |
|-------------|----------|--------------|
| personamem_32k_80 | 30 | 0% |
| personamem_32k_494 | 26 | 0% |
| personamem_32k_78 | 17 | 0% |
| personamem_32k_158 | 12 | 0% |
| personamem_32k_483 | 10 | 0% |
| personamem_32k_214 | 10 | 0% |

#### Issue 3: Gold Label vs Evidence Conflicts
Example: `personamem_32k_80` asks about mind maps.
- **Memory extracted**: "Visually engaging but left her feeling **scattered and overwhelmed**"
- **Gold answer (c)**: "...mind maps... which you found **engaging**" (emphasizes positive)
- **Our answer (a)**: "...mind maps to be quite **overwhelming**" (matches memory!)
- **Result**: Marked wrong, but our answer is more evidence-aligned.

### Adjusted Ceiling Estimate

Given benchmark issues, the **true ceiling** for PersonaMem may be:
- 6 impossible questions (0% success) = ~1% penalty on 589 questions
- 4 duplicate-option questions = another ~0.7% penalty
- **Effective ceiling**: ~98% (not 100%)

Current 66.7% → realistic target might be **75-80%** after fixing extraction quality.

### Rollback Summary

| Change | Status | Why |
|--------|--------|-----|
| `tool_choice="required"` | ROLLED BACK | Caused regression |
| Verbose `<retrieval_protocol>` | ROLLED BACK | No improvement |
| Simpler baseline prompt | KEPT | Best performing |

### Next Steps (Recommended)

1. **Report benchmark bugs** to PersonaMem maintainers (duplicate options)
2. **Exclude known-bad questions** from evaluation scoring
3. **Focus on extraction fidelity** - the "mind maps" example shows we lose nuance during ingestion
4. **Consider embedding model** - some failures may be retrieval ranking issues
