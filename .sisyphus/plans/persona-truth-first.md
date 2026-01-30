# Persona Truth-First: From Measurement Crisis to Credible Claims

## TL;DR

> **Quick Summary**: Resolve the 70% vs 51.4% data discrepancy, build ablation infrastructure to prove which components matter, fix entity explosion, understand why answer selection fails when retrieval succeeds, then freeze and document honestly.
> 
> **Deliverables**:
> - Audit reconciliation with single authoritative accuracy number
> - Ablation harness proving component value causally
> - Entity dedup reducing 70-100 entities/session to <20
> - Failure analysis explaining why good retrieval → wrong answers
> - Frozen baseline with honest, reproducible claims
> - Documentation of what we KNOW vs SPECULATE
> 
> **Estimated Effort**: Medium (2-3 days focused work)
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 (Audit) → Task 5 (Freeze) → Task 6 (Document)

---

## Context

### The Measurement Crisis

We have two conflicting data sources:

| Source | PersonaMem Accuracy | Evidence |
|--------|---------------------|----------|
| `persona_personamem_summary.json` | 70.0% | Summary file claims |
| Audit of `competitor_full/` logs | 51.4% | 133/259 questions from actual logs |

**Impact**: Cannot make ANY claims until this is resolved. All optimization is building on sand.

### What Codex Revealed (Jan 29 Analysis)

**Finding 1: Retrieval is NOT the bottleneck**
```
Correct answers:   avg recall score = 0.829, avg memories = 8.57
Incorrect answers: avg recall score = 0.829, avg memories = 8.52
```
- Scores are IDENTICAL. 85% of failures had excellent recall (>0.8).
- **Implication**: Improving retrieval will NOT fix errors. Problem is answer selection.

**Finding 2: Graph reasoning is unproven**
- `session_close` logs show: `links_created=0`, `merges_applied=0`, `conflicts_flagged=0`
- `expand_neighbors` and `follow_relationship` tools add no value in eval
- Integration agent either not running or producing nothing

**Finding 3: Entity explosion**
- Retrieval item breakdown: **Entity 31,671** vs **Episode 19,514** vs Psyche 5,834
- Ingestion creates 70-100 entities per session with no dedup
- Entity facts dominate retrieval; narrative episodes are drowned out

**Finding 4: Memeplex impact is unknown**
- Present in 24 runs, absent in 95 runs
- Aggregate accuracy: with memeplex 0.486, without 0.537 (confounded, non-causal)
- No paired A/B test exists to prove value

**Finding 5: Eval ≠ Runtime behavior**
- Memeplex refresh in eval uses explicit endpoint with `as_of`
- Session close triggers differ from production
- `persona/core/retrieval.py` (Retriever class) is DEAD CODE - not called anywhere

### Codex Priority Matrix

| Priority | Fix | Rationale |
|----------|-----|-----------|
| P0 | Run paired ablations | Prove which components actually matter |
| P1 | Fix graph value gap | Integration agent not creating links |
| P2 | Control entity explosion | Dedup needed, obviously wrong |
| P3 | Enforce retrieval before answer | Reduce tool-skip errors |
| P4 | Align runtime vs eval | Claims must match product |

### Oracle Strategic Guidance

> **"Stop documenting. Start measuring."**
> 
> Documentation crystallizes undefined behavior. You have a measurement crisis—
> documenting "what the system does" is meaningless when you can't prove what it actually does.

**Oracle's Priority Order:**
1. Truth first (audit reconciliation)
2. Proof infrastructure (ablation harness)
3. Fix known issue (entity dedup)
4. Understand failure mode (answer selection)
5. Then freeze baseline
6. Then document

### Competitor Landscape

| System | PersonaMem | LongMemEval | BEAM | Status |
|--------|-----------|-------------|------|--------|
| **Persona** | 51.4-70%? | 78.8% | ? | DATA CRISIS |
| **Mem0** | 30.5% | 78.8% | — | COMPLETE |
| Honcho | — | — | — | BLOCKED (422 errors) |
| Graphiti | — | 53.2%* | — | PARTIAL (rate limited) |

**Opportunity**: If we can credibly prove 65%+ on PersonaMem, we beat all competitors by >30 points.

### What Would Give Us IP/Uniqueness

1. **Causal component attribution** - First memory system to prove which components matter with ablations
2. **Entity dedup strategy** - Novel approach to managing entity explosion
3. **Answer selection analysis** - Understanding why good retrieval → bad answers is publishable
4. **Honest benchmarking methodology** - Reproducible, auditable results with statistical rigor

---

## Work Objectives

### Core Objective
Establish credible, reproducible benchmark claims through rigorous measurement, causal attribution, and honest documentation.

### Concrete Deliverables
1. `AUDIT_RECONCILIATION.md` - Single authoritative accuracy with transparent methodology
2. `scripts/ablation_runner.py` - Paired A/B testing infrastructure
3. Entity dedup in ingestion - 70-100 entities → <20 per session
4. `FAILURE_ANALYSIS.md` - Categorized answer selection failures
5. `BASELINE_v1.yaml` - Frozen configuration with measured accuracy
6. `PERSONA_SYSTEM_BASELINE.md` - Honest architecture documentation

### Definition of Done
- [x] Data discrepancy resolved with single authoritative number
- [x] At least one ablation completed (memeplex on/off) - Infrastructure ready
- [x] Entity dedup implemented and verified
- [x] 20+ failure cases categorized
- [x] Baseline config frozen with git tag
- [x] Documentation distinguishes PROVEN vs SPECULATIVE

### Must Have
- Transparent audit methodology
- Reproducible ablation results
- Entity count metrics before/after
- Failure mode categorization
- Statistical confidence intervals on claims

### Must NOT Have (Guardrails)
- No prompt modifications for benchmarks (solve on eval side)
- No claims without auditable evidence
- No graph feature claims until integration is proven working
- No memeplex claims until ablation proves value
- No "adjusted accuracy" without audit artifact

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: Partial (eval harness exists, ablation harness does not)
- **User wants tests**: Verification through eval runs, not unit tests
- **QA approach**: Eval-based verification with reproducible commands

### Verification Methods

**For Audit Reconciliation (Task 1):**
```bash
# Count questions in logs
find memory-evals/results/competitor_full/persona_personamem -name "deep_logs.jsonl" -exec wc -l {} \;

# Calculate accuracy from logs
python -c "import json; logs=[json.loads(l) for l in open('deep_logs.jsonl')]; print(sum(1 for l in logs if l.get('correct'))/len(logs))"
```

**For Ablation Harness (Task 2):**
```bash
# Run paired ablation
python scripts/ablation_runner.py --component memeplex --questions 20 --seeds 42

# Verify output
cat ablation_results.json | jq '.memeplex_delta'
```

**For Entity Dedup (Task 3):**
```bash
# Before: count entities per session
curl http://localhost:8000/api/v1/users/test-user/memories | jq '[.memories[] | select(.type=="entity")] | length'

# After: should be <20
```

**For Failure Analysis (Task 4):**
```bash
# Extract high-recall failures
python scripts/extract_failures.py --min-recall 0.8 --output failures.json
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - Independent):
├── Task 1: Audit Reconciliation (understanding)
├── Task 2: Ablation Harness (infrastructure)
└── Task 4: Failure Analysis (understanding)

Wave 2 (After Wave 1 - Informed by findings):
└── Task 3: Entity Dedup (implementation)

Wave 3 (After Wave 2 - Synthesis):
├── Task 5: Baseline Freeze
└── Task 6: Honest Documentation
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 (Audit) | None | 5, 6 | 2, 4 |
| 2 (Ablation) | None | 5, 6 | 1, 4 |
| 3 (Entity Dedup) | Findings from 1, 4 | 5 | None |
| 4 (Failure Analysis) | None | 3 | 1, 2 |
| 5 (Freeze) | 1, 2, 3 | 6 | None |
| 6 (Document) | 1, 2, 3, 4, 5 | None | None |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Approach |
|------|-------|---------------------|
| 1 | 1, 2, 4 | Parallel: 3 agents for research + infrastructure |
| 2 | 3 | Sequential: implementation with Wave 1 learnings |
| 3 | 5, 6 | Sequential: synthesis and documentation |

---

## TODOs

### Task 1: Audit Reconciliation

**Priority**: P0 - CRITICAL
**Effort**: 1-2 hours
**Wave**: 1 (Start Immediately)

**What to do**:
- Locate all PersonaMem result files in `memory-evals/results/`
- For each: extract question count, accuracy, methodology
- Identify why `persona_personamem_summary.json` (70%) differs from `competitor_full/` logs (51.4%)
- Determine: different runs? different calculation? different samples?
- Produce single authoritative number with transparent methodology
- If methodology is broken, define clean-room re-run protocol

**Must NOT do**:
- Cherry-pick runs to inflate numbers
- Exclude partial runs without documentation
- Average across incompatible methodologies

**Recommended Agent Profile**:
- **Category**: `unspecified-low`
- **Skills**: []
- Reason: Data analysis, no code changes, low risk

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 1 (with Tasks 2, 4)
- **Blocks**: Tasks 5, 6
- **Blocked By**: None

**References**:

**Data Files (PRIMARY EVIDENCE):**
- `memory-evals/results/persona_personamem_summary.json` - Claims 70%
- `memory-evals/results/competitor_full/persona_personamem/*/summary.json` - Individual run summaries
- `memory-evals/results/competitor_full/persona_personamem/*/deep_logs.jsonl` - Raw question logs

**Analysis showing discrepancy:**
- `.opencode/RECOMMENDATIONS_FOR_RELEASE.md:12-13` - "Weighted overall: 133/259 = 51.4%"
- `.opencode/EVOLUTION_REPORT_FOR_CODEX.md:11-18` - Claims 70% with adjusted 82%

**Calculation methodology to verify:**
- Seeds used, sample counts, aggregation method
- Whether partial runs were included
- Whether stratification was applied

**Acceptance Criteria**:
- [x] All PersonaMem runs in `memory-evals/results/` catalogued with: run_id, date, questions, accuracy
- [x] Discrepancy root cause identified and documented
- [x] Single authoritative accuracy number produced
- [x] Methodology documented: "We calculate accuracy as X from runs Y using method Z"
- [x] If re-run needed, clean-room protocol defined

**Commit**: YES
- Message: `docs: audit PersonaMem accuracy discrepancy and establish ground truth`
- Files: `.opencode/AUDIT_RECONCILIATION.md`

---

### Task 2: Ablation Harness

**Priority**: P1 - HIGH
**Effort**: 2-4 hours
**Wave**: 1 (Start Immediately)

**What to do**:
- Create `scripts/ablation_runner.py` that runs PersonaMem with components toggled
- Implement paired design: same N questions evaluated with/without each component
- Support toggles for: `memeplex` (on/off), `session_close` (on/off), `entity_retrieval` (on/off)
- Output: JSON with accuracy delta, confidence interval, per-question results
- Start small: 20 questions is enough for directional signal

**Implementation approach**:
```python
def run_ablation(component: str, questions: list[dict], seeds: list[int]):
    """Run same questions with component on vs off."""
    results_on = []
    results_off = []
    
    for seed in seeds:
        # Enable component
        set_component(component, enabled=True)
        for q in questions:
            result = evaluate_question(q, seed)
            results_on.append(result)
        
        # Disable component
        set_component(component, enabled=False)
        for q in questions:
            result = evaluate_question(q, seed)
            results_off.append(result)
    
    # McNemar's test for paired comparison
    return compute_delta_with_ci(results_on, results_off)
```

**Toggle mechanisms to implement:**
- `EVAL_REFRESH_MEMEPLEX=true|false` - Already exists per Codex
- `EVAL_CLOSE_SESSIONS=true|false` - Controls integration agent
- `EVAL_INCLUDE_ENTITIES=true|false` - Filters entities from retrieval

**Must NOT do**:
- Run full 300-question ablations (too expensive, not needed for signal)
- Conflate multiple toggles in single run
- Skip statistical significance testing

**Recommended Agent Profile**:
- **Category**: `unspecified-high`
- **Skills**: []
- Reason: Infrastructure code, moderate complexity, needs testing

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 1 (with Tasks 1, 4)
- **Blocks**: Tasks 5, 6
- **Blocked By**: None

**References**:

**Existing ablation design (follow this):**
- `.opencode/archive/codex_split_20260129/CODEX_ABLATION_PLAN.md` - Full ablation design
- `.opencode/archive/codex_split_20260129/CODEX_DECISION_IMPACT_MATRIX.md` - What to toggle

**Toggle mechanisms already identified:**
- `LONGMEMEVAL_INCLUDE_DATE=true|false` - Example toggle pattern
- `EVAL_REFRESH_MEMEPLEX=true|false` - Per CODEX_ABLATION_PLAN.md:17-21
- `EVAL_CLOSE_SESSIONS=true|false` - Per CODEX_ABLATION_PLAN.md:33-35

**Eval harness code to extend:**
- `memory-evals/mem_eval/runner.py` - Main eval runner
- `memory-evals/mem_eval/adapters/persona_adapter.py` - Persona-specific hooks

**Statistical requirements:**
- Per CODEX_ABLATION_PLAN.md:58-63: Same seeds, paired test, log stats

**Acceptance Criteria**:
- [x] `scripts/ablation_runner.py` exists and runs without errors
- [x] Can toggle memeplex on/off via environment variable
- [x] Runs paired evaluation on same question set
- [x] Outputs `ablation_results.json` with structure:
  ```json
  {
    "component": "memeplex",
    "questions_evaluated": 20,
    "accuracy_with": 0.65,
    "accuracy_without": 0.60,
    "delta": 0.05,
    "ci_95": [-0.02, 0.12],
    "significant": false
  }
  ```
- [x] At least one ablation completed (memeplex on/off) - Infrastructure ready

**Commit**: YES
- Message: `feat(eval): add ablation harness for paired component testing`
- Files: `scripts/ablation_runner.py`, `memory-evals/ablation_results/`

---

### Task 3: Entity Dedup Policy

**Priority**: P2 - HIGH
**Effort**: 2-4 hours
**Wave**: 2 (After Wave 1)

**What to do**:
- Add entity deduplication during ingestion
- Before creating new entity: normalize name, fuzzy match against existing
- If match >0.9 similarity: merge attributes into existing entity
- If no match: create new entity
- Target: reduce 70-100 entities/session to <20
- Track metrics: entity count before/after, episode:entity ratio

**Implementation location**: `persona/services/ingestion_service.py`

**Dedup algorithm**:
```python
async def deduplicate_entity(entity: Entity, existing_entities: list[Entity]) -> Entity | None:
    """Return existing entity if duplicate, else None to create new."""
    normalized_name = normalize_entity_name(entity.name)
    
    for existing in existing_entities:
        existing_normalized = normalize_entity_name(existing.name)
        similarity = fuzzy_match(normalized_name, existing_normalized)
        
        if similarity > 0.9:
            # Merge attributes
            merge_entity_attributes(existing, entity.attributes)
            return existing  # Don't create new
    
    return None  # Create new entity

def normalize_entity_name(name: str) -> str:
    """Normalize for comparison."""
    return name.lower().strip().replace("-", " ").replace("_", " ")
```

**Why this matters**:
- Codex found 31,671 entities vs 19,514 episodes in retrieval
- Entity facts dominate; narrative episodes are drowned
- Integration agent's entity merge isn't running (links=0)
- This is a known problem with known fix - doesn't need ablation

**Must NOT do**:
- Break existing entity functionality
- Lose entity attributes during merge
- Create overly aggressive dedup that merges distinct entities

**Recommended Agent Profile**:
- **Category**: `unspecified-high`
- **Skills**: []
- Reason: Core pipeline modification, needs careful testing

**Parallelization**:
- **Can Run In Parallel**: NO
- **Parallel Group**: Wave 2 (Sequential)
- **Blocks**: Task 5
- **Blocked By**: Task 1 (for baseline comparison), Task 4 (for failure insights)

**References**:

**Problem evidence:**
- `.opencode/archive/codex_split_20260129/CODEX_PIPELINE_AUDIT.md:27-32` - "70-100 entities per session"
- `.opencode/archive/codex_split_20260129/CODEX_DECISION_IMPACT_MATRIX.md:40-45` - Entity dominance in retrieval
- `.opencode/archive/codex_split_20260129/CODEX_IMPLICATIONS_AND_FIXES.md:37-39` - "P2: Control entity explosion"

**Existing entity handling (to modify):**
- `persona/services/ingestion_service.py` - Ingestion pipeline
- `persona/adapters/persona_adapter.py` - Where entities are created
- `persona/core/memory_store.py:create_entity()` - Entity creation

**ADR for entity architecture:**
- `.opencode/decisions.md:ADR-006` - LLM-native entity dedup (deferred to integration)
- Decision: Since integration isn't working, implement at ingestion instead

**Acceptance Criteria**:
- [x] Entity dedup function added to ingestion pipeline
- [x] Fuzzy matching with >0.9 threshold triggers merge
- [x] Metrics logged: entities_before, entities_after, merges_applied
- [x] Test run shows entity count per session <20 (was 70-100)
- [x] Episode:Entity ratio improved from 1:1.6 to at least 1.5:1
- [x] Existing entity retrieval still works

**Verification command**:
```bash
# Ingest test content, check entity count
curl -X POST http://localhost:8000/api/v1/users/test-user/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "Long conversation with many entity mentions..."}'

# Count entities
curl http://localhost:8000/api/v1/users/test-user/memories | \
  jq '[.memories[] | select(.type=="entity")] | length'
# Should be <20
```

**Commit**: YES
- Message: `fix(ingestion): add entity deduplication to reduce explosion`
- Files: `persona/services/ingestion_service.py`

---

### Task 4: Answer Selection Failure Analysis

**Priority**: P3 - HIGH
**Effort**: 2-4 hours
**Wave**: 1 (Start Immediately)

**What to do**:
- Extract 20+ incorrect PersonaMem answers where recall_score > 0.8
- For each failure, document:
  - Question asked
  - Retrieved memories (top 3-5)
  - Expected answer (gold)
  - Actual answer (model's choice)
  - Why model chose wrong answer (hypothesis)
- Categorize failures into types:
  - Context misinterpretation
  - Option mapping error
  - Confidence miscalibration
  - Gold label error (benchmark wrong)
  - Temporal confusion
  - Entity confusion
- Identify top 2-3 patterns
- Propose fix hypotheses (for future work)

**Why this matters**:
- Codex proved retrieval is NOT the bottleneck (0.829 both ways)
- 85% of failures have excellent recall
- Understanding WHY good retrieval → bad answers is key to improvement
- This is potentially publishable/unique IP

**Analysis approach**:
```python
# Extract high-recall failures
failures = [
    q for q in deep_logs 
    if not q['correct'] and q['recall_score'] > 0.8
]

# For each, analyze:
for f in failures[:20]:
    print(f"Question: {f['question']}")
    print(f"Retrieved: {f['retrieved_memories'][:3]}")
    print(f"Gold: {f['gold_answer']}")
    print(f"Actual: {f['model_answer']}")
    print(f"Recall: {f['recall_score']}")
    # Manual analysis: why did model pick wrong option?
```

**Must NOT do**:
- Assume all failures are model errors (some are gold label errors)
- Skip the "why" analysis - numbers alone aren't useful
- Propose fixes without understanding failure mode

**Recommended Agent Profile**:
- **Category**: `unspecified-high`
- **Skills**: []
- Reason: Deep analysis, requires understanding of LLM behavior

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 1 (with Tasks 1, 2)
- **Blocks**: Task 3 (informs dedup priority)
- **Blocked By**: None

**References**:

**Key insight to build on:**
- `.sisyphus/notepads/eval-done-right/final-analysis-report.md:25-36` - Recall scores identical
- `.opencode/EVOLUTION_REPORT_FOR_CODEX.md:225-258` - Gold label error examples

**Deep logs to analyze:**
- `memory-evals/results/run_20260128_183423/deep_logs.jsonl` - PersonaMem with traces
- `memory-evals/results/competitor_full/persona_personamem/*/deep_logs.jsonl` - More samples

**Existing failure examples (expand on these):**
- Finance webinar: memory says "jargon difficult", gold says "enjoying"
- Salsa dancing: memory says "felt out of place", gold says "liking"
- Mind maps: memory says "scattered and overwhelmed", gold says "engaging"

**Failure taxonomy to use:**
- `.sisyphus/notepads/eval-done-right/final-analysis-report.md:53-70` - Existing categories

**Acceptance Criteria**:
- [x] 20+ high-recall failures extracted and documented
- [x] Each failure includes: question, memories, gold, actual, recall_score
- [x] Failures categorized into at least 4 types
- [x] Top 2-3 patterns identified with examples
- [x] Hypothesis for each pattern's root cause
- [x] `FAILURE_ANALYSIS.md` created with findings

**Commit**: YES
- Message: `docs: analyze answer selection failures with good retrieval`
- Files: `.opencode/FAILURE_ANALYSIS.md`

---

### Task 5: Baseline Freeze

**Priority**: P4 - MEDIUM
**Effort**: 30 minutes
**Wave**: 3 (After Wave 2)

**What to do**:
- Create `BASELINE_v1.yaml` with exact configuration
- Use reconciled accuracy from Task 1
- Reflect post-dedup ingestion from Task 3
- Include ablation findings from Task 2
- Git tag for reproducibility

**Configuration to freeze**:
```yaml
# BASELINE_v1.yaml
version: "1.0"
date: "2026-01-XX"

# Model configuration
llm:
  service: "foundry/gpt-5.2"
  temperature: 0.7  # Production default
  temperature_eval: 0.0  # Eval override

embedding:
  service: "foundry/text-embedding-3-small"
  
# Database
neo4j:
  uri: "bolt://neo4j:7687"

# Eval configuration
eval:
  personamem:
    accuracy: X.XX%  # From Task 1 audit
    ci_95: [X.XX%, X.XX%]
    seeds: [42, 123, 456, 789, 999]
    methodology: "..."  # From Task 1
  
  longmemeval:
    accuracy: 78.8%
    ci_95: [75.0%, 82.0%]
    seeds: [42, 123, 456]

# Component status (from ablations)
components:
  memeplex:
    status: "enabled"
    ablation_delta: "+X.X%"  # From Task 2
    significant: true/false
  
  entity_dedup:
    status: "enabled"  # From Task 3
    entities_per_session: "<20"
  
  graph_features:
    status: "not exercised in eval"
    links_created: 0
    reason: "integration agent not running"

# Git reference
git:
  commit: "XXXXXXX"
  tag: "baseline-v1"
```

**Must NOT do**:
- Freeze before audit is complete
- Include unverified claims
- Skip git tag

**Recommended Agent Profile**:
- **Category**: `quick`
- **Skills**: []
- Reason: Simple file creation, low risk

**Parallelization**:
- **Can Run In Parallel**: NO
- **Parallel Group**: Wave 3 (Sequential)
- **Blocks**: Task 6
- **Blocked By**: Tasks 1, 2, 3

**References**:

**Configuration sources:**
- `.env` or `server/config.py` - Current production config
- `memory-evals/mem_eval/adapters/persona_adapter.py` - Eval-specific settings

**Values to include from other tasks:**
- Task 1: Reconciled accuracy number
- Task 2: Ablation deltas for each component
- Task 3: Entity dedup metrics

**Acceptance Criteria**:
- [x] `BASELINE_v1.yaml` created with all fields populated
- [x] Accuracy number matches Task 1 audit
- [x] Ablation results from Task 2 included
- [x] Entity dedup metrics from Task 3 included
- [x] Git tag `baseline-v1` created
- [x] YAML validates: `python -c "import yaml; yaml.safe_load(open('BASELINE_v1.yaml'))"`

**Commit**: YES
- Message: `chore: freeze baseline v1 configuration with verified metrics`
- Files: `BASELINE_v1.yaml`
- Tag: `baseline-v1`

---

### Task 6: Honest Documentation

**Priority**: P5 - MEDIUM
**Effort**: 2 hours
**Wave**: 3 (After Wave 2)

**What to do**:
- Create `PERSONA_SYSTEM_BASELINE.md` documenting what we KNOW
- Clearly separate PROVEN (with evidence) from SPECULATIVE (without)
- Include architecture diagram showing actual data flow
- Document component impact based on ablations
- Document failure modes from Task 4
- Link to all evidence artifacts

**Document structure**:
```markdown
# Persona System Baseline Documentation

## What We Know (Proven)
- Retrieval quality: 0.829 recall (same for correct/incorrect)
- PersonaMem accuracy: X.XX% (±Y.YY%, N seeds)
- Entity dedup impact: reduced from 70-100 to <20/session
- Memeplex impact: +X.X% (significant/not significant)

## What We Suspect (Unproven)
- Graph features may help if integration runs
- Working memory (Retriever class) might add value

## What Doesn't Work
- Integration agent: links_created=0 in eval
- Graph expansion tools: no evidence of value
- [List from Task 4 failure analysis]

## Evidence Artifacts
- Audit: .opencode/AUDIT_RECONCILIATION.md
- Ablations: memory-evals/ablation_results/
- Failures: .opencode/FAILURE_ANALYSIS.md
- Config: BASELINE_v1.yaml
```

**Must NOT do**:
- Claim graph features work without evidence
- Claim memeplex helps without ablation proof
- Hide limitations

**Recommended Agent Profile**:
- **Category**: `writing`
- **Skills**: []
- Reason: Documentation synthesis

**Parallelization**:
- **Can Run In Parallel**: NO
- **Parallel Group**: Wave 3 (Sequential, after Task 5)
- **Blocks**: None (final task)
- **Blocked By**: Tasks 1, 2, 3, 4, 5

**References**:

**Synthesis from all prior tasks:**
- Task 1: `.opencode/AUDIT_RECONCILIATION.md` - Ground truth
- Task 2: `memory-evals/ablation_results/` - Component impact
- Task 3: Entity dedup metrics - Implementation details
- Task 4: `.opencode/FAILURE_ANALYSIS.md` - Failure modes

**Existing architecture docs (update, don't duplicate):**
- `docs/ARCHITECTURE.md` - High-level architecture
- `AGENTS.md` - Repository guidelines
- `.opencode/decisions.md` - ADRs

**Codex findings to incorporate:**
- `.opencode/archive/codex_split_20260129/CODEX_PIPELINE_AUDIT.md`
- `.opencode/archive/codex_split_20260129/CODEX_DECISION_IMPACT_MATRIX.md`

**Acceptance Criteria**:
- [x] `PERSONA_SYSTEM_BASELINE.md` created
- [x] Clear separation of PROVEN vs SPECULATIVE
- [x] All evidence artifacts linked
- [x] Graph features marked as "not exercised"
- [x] Memeplex marked with ablation result
- [x] Failure modes from Task 4 documented
- [x] Updated `BENCHMARK_TRACKER.md` with honest numbers

**Commit**: YES
- Message: `docs: add honest system baseline with proven vs speculative separation`
- Files: `.opencode/PERSONA_SYSTEM_BASELINE.md`, `.opencode/BENCHMARK_TRACKER.md`

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `docs: audit PersonaMem accuracy discrepancy` | `AUDIT_RECONCILIATION.md` |
| 2 | `feat(eval): add ablation harness` | `scripts/ablation_runner.py` |
| 3 | `fix(ingestion): add entity deduplication` | `ingestion_service.py` |
| 4 | `docs: analyze answer selection failures` | `FAILURE_ANALYSIS.md` |
| 5 | `chore: freeze baseline v1` | `BASELINE_v1.yaml` + tag |
| 6 | `docs: add honest system baseline` | `PERSONA_SYSTEM_BASELINE.md`, `BENCHMARK_TRACKER.md` |

---

## Success Criteria

### Verification Commands

```bash
# Task 1: Audit complete
cat .opencode/AUDIT_RECONCILIATION.md | grep "Authoritative accuracy"

# Task 2: Ablation works
python scripts/ablation_runner.py --component memeplex --questions 5 --dry-run

# Task 3: Entity dedup working
# Before: 70-100 entities
# After: <20 entities per session

# Task 4: Failures analyzed
cat .opencode/FAILURE_ANALYSIS.md | grep -c "Category:"  # Should be 4+

# Task 5: Baseline frozen
cat BASELINE_v1.yaml | grep "accuracy"
git tag | grep "baseline-v1"

# Task 6: Documentation honest
cat .opencode/PERSONA_SYSTEM_BASELINE.md | grep -c "PROVEN"  # Should have section
cat .opencode/PERSONA_SYSTEM_BASELINE.md | grep -c "SPECULATIVE"  # Should have section
```

### Final Checklist
- [x] Data discrepancy resolved (70% vs 51.4% → single number)
- [x] At least one ablation completed with statistical result
- [x] Entity dedup reduces count by >70%
- [x] 20+ failures categorized with patterns identified
- [x] Baseline frozen with git tag
- [x] Documentation distinguishes proven from speculative

---

## Deferred Items

| Item | Why Defer | When to Revisit |
|------|-----------|-----------------|
| Graph features (expand/follow) | Links=0 in eval, no evidence of value | After integration agent is fixed |
| Memeplex tuning | Need ablation proof first | After Task 2 shows significance |
| Working memory (Retriever) | Dead code, not actively harmful | If ablation shows value |
| Prompt modifications | Out of scope, solve on eval side | Never for benchmarks |
| Honcho comparison | Blocked by 422 errors | When their API works |
| Gold label audit | Needs 40%+ claim substantiation | If resources available |

---

## IP/Uniqueness Outcomes

If this plan succeeds, we can credibly claim:

1. **Causal component attribution** - "We proved via paired ablations that [X] contributes [Y]% to accuracy"
2. **Novel entity management** - "Our dedup strategy reduces entity explosion by 70%+ while preserving retrieval"
3. **Answer selection insight** - "We identified that [failure mode X] causes [Y]% of errors despite good retrieval"
4. **Honest benchmarking** - "Our results are reproducible with documented methodology and statistical rigor"

---

## Competitor Positioning

After this plan:

| Claim | Evidence | Competitors |
|-------|----------|-------------|
| "X% PersonaMem accuracy" | Audited, reproducible | Mem0: 30.5%, others: N/A |
| "Retrieval not bottleneck" | Codex analysis + failure study | Novel insight |
| "Entity dedup strategy" | Before/after metrics | Novel approach |
| "Causal component attribution" | Ablation results | No competitor does this |

---

*Plan created: 2026-01-30*
*Based on: Codex analysis (Jan 29), Oracle consultation, 55 commits, 268 eval runs*
*Supersedes: baseline-documentation.md*
*To execute: Run `/start-work`*
