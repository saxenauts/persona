# Eval & Git Strategy for v1 Release (REVISED)
> Created: Dec 26, 2025 | Branch: arch/v1-release
> Revised based on feedback: No task-adaptive retrieval, reordered checkpoints

---

## Core Principle: Universal `/chat` Endpoint

**The memory engine should work uniformly.** No task-adaptive retrieval that changes behavior based on detected question type. The internal mechanics must be consistent - we're building infrastructure, not a benchmark optimizer.

---

## Eval Appropriateness Analysis

### What PersonaMem CAN Measure

| Feature | Signal Strength | Best Question Types |
|---------|----------------|---------------------|
| Keyword index | ✅ Direct | `recall_user_shared_facts`, `recalling_facts_mentioned` |
| Temporal filtering | ✅ Direct | `track_full_preference_evolution` |
| Context-aware extraction | ⚠️ Indirect | All (better memory → better answers) |
| Smart link formation | ⚠️ Indirect | `recalling_reasons`, `generalizing_to_new_scenarios` |
| Multi-hop traversal | ⚠️ Indirect | `suggest_new_ideas`, reasoning chains |

### What PersonaMem CANNOT Measure

| Feature | Why Not | Alternative Eval Needed |
|---------|---------|------------------------|
| Goals → Notes rename | Terminology only | Tests pass |
| User Profile One-Pager | Single-session, no persistence | Multi-session test suite |
| Async consolidation | No time to run | Synthetic long-term test |
| Clustering | No multi-day accumulation | Cluster quality metrics |
| Link strength/decay | Requires time passage | Decay simulation |
| Intelligent forgetting | Nothing to forget yet | Retention curve test |
| Hebbian strengthening | Requires repeated access | Access pattern simulation |

### Implications

**Report honestly**: When eval results don't move for Checkpoint 2/3 features, that's expected - PersonaMem isn't designed to measure those. Add supplementary metrics.

---

## Revised Checkpoint Structure

### Checkpoint 0: Foundation (No Eval)
**Goal**: Clean terminology before any real work

| Change | Files |
|--------|-------|
| Goals → Notes rename | ~21 Python + ~10 Mintlify docs |

**Verify**: Tests pass, docs build, no runtime errors

---

### Checkpoint 1: Smarter Ingestion + Retrieval (Full Eval)
**Goal**: Better memory quality and retrieval - things PersonaMem CAN measure

| Change | Eval Signal | Files |
|--------|-------------|-------|
| Context-aware extraction (past 2 days) | ⚠️ Indirect | `ingestion_service.py` |
| Smart link formation (sync/async hybrid) | ⚠️ Indirect | New `link_discovery.py` |
| Keyword index for Notes | ✅ Direct | `memory_store.py` |
| Temporal filtering (natural language) | ✅ Direct | `retrieval.py` |
| Multi-hop traversal (depth 2-3) | ⚠️ Indirect | `retrieval.py` |

**NOT included** (moved to later checkpoints):
- ~~Task-adaptive retrieval~~ (violates universal endpoint principle)
- ~~Relationship weighting~~ (belongs with weighing/forgetting)

**Expected**: 51.7% → 60-65% (conservative, some features indirect)

**Supplementary Metrics**:
- Link coverage: # links created per memory
- Keyword index size and hit rate
- Temporal query resolution accuracy

---

### Checkpoint 2: Memory Card + Async Consolidation (Mixed Eval)
**Goal**: User profile and background processing - partial PersonaMem signal

| Change | Eval Signal | Files |
|--------|-------------|-------|
| User Profile One-Pager | ❌ Not measurable | New `profile.py` |
| Memory card as retrieval index | ⚠️ Indirect | `retrieval.py` |
| Async consolidation (dream cycle) | ❌ Not measurable | New `consolidation/` |
| Clustering into life themes | ❌ Not measurable | `consolidation/cluster.py` |

**Expected PersonaMem**: Modest improvement from memory card (65-68%)

**Custom Metrics Needed**:
- Profile completeness score
- Cluster coherence (silhouette score)
- Consolidation efficiency (memories merged/hour)

---

### Checkpoint 3: Weighing + Forgetting (Custom Eval Only)
**Goal**: Intelligent memory lifecycle - PersonaMem NOT appropriate

| Change | Eval Signal | Files |
|--------|-------------|-------|
| Link strength (Hebbian) | ❌ Not measurable | `models/memory.py`, `link_discovery.py` |
| Temporal decay | ❌ Not measurable | `memory_store.py` |
| Intelligent forgetting | ❌ Not measurable | `consolidation/prune.py` |
| Causal chain discovery | ❌ Not measurable | `consolidation/causal.py` |

**PersonaMem Expected**: No change (features not measured)

**Custom Eval Required**:
- Synthetic multi-session test (simulate weeks of use)
- Decay curve validation
- Retention quality after pruning

---

## Git Practices

### Branch Strategy
```
main
  └── arch/v1-release (working branch)
        ├── commit: "rename: Goals → Notes (exhaustive)"
        │
        ├── commit: "feat(ingestion): context-aware extraction"
        ├── commit: "feat(links): smart link formation with keyword index"
        ├── commit: "feat(retrieval): temporal filtering + multi-hop"
        │   └── TAG: checkpoint-1
        │   └── EVAL: Targeted subset (58 questions)
        │
        ├── commit: "feat(profile): user one-pager with retrieval index"
        ├── commit: "feat(consolidation): dream cycle + clustering"
        │   └── TAG: checkpoint-2
        │   └── EVAL: Targeted subset + custom metrics
        │
        ├── commit: "feat(links): hebbian strength + decay"
        ├── commit: "feat(consolidation): intelligent forgetting"
        │   └── TAG: checkpoint-3
        │   └── EVAL: Custom long-term simulation only
        │
        └── PR to main
```

### Commit Discipline
- **Atomic commits**: Each buildable and testable
- **Conventional commits**: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
- **Eval only at checkpoints**: Not after every commit
- **Tag checkpoints**: For easy rollback and comparison

---

## Eval Recording Template

```markdown
## Checkpoint N: [Name]
- Date: YYYY-MM-DD
- Commit: [hash]
- Tag: checkpoint-N

### Changes in this checkpoint
- [list]

### PersonaMem Results
| Metric | Before | After | Delta | Notes |
|--------|--------|-------|-------|-------|
| Overall | X% | Y% | ±Z% | |
| suggest_new_ideas | X% | Y% | ±Z% | Target weakness |
| ... | | | | |

### Eval Appropriateness Notes
- Features that SHOULD show improvement: [list]
- Features that WON'T show in PersonaMem: [list]
- Supplementary metrics collected: [list]

### Custom Metrics (if applicable)
- Link coverage: X links/memory
- Keyword index hit rate: X%
- ...

### Analysis
- What worked:
- What didn't:
- Next steps:
```

---

## Eval Subsets (NO FULL EVALS)

### Targeted Subset (58 questions) - PRIMARY
**File**: `questions_32k_targeted.json`

Contains failures we expect to fix + controls:

| Category | Count | What Should Fix It |
|----------|-------|-------------------|
| keyword_index failures | 12 | Keyword index for Notes |
| temporal_filter failures | 12 | Natural language temporal filtering |
| multi_hop failures | 12 | Deeper graph traversal |
| context_extraction failures | 12 | Better memory extraction |
| controls (currently correct) | 10 | Should stay correct |

**Use**: After Checkpoint 1 - measure if our changes fixed what we expected

**Success criteria**:
- Failures → Correct: Target 50%+ of the 48 failures
- Controls stay correct: 90%+ of the 10 controls
- If controls break, we regressed

### Fast Subset (59 questions) - SECONDARY
**File**: `questions_32k_fast.json`

Proportionally sampled across all question types. Use for general sanity checks.

### Full Eval (589 questions) - RARE
Only run when:
- Ready to cut a release
- Making a claim about SOTA
- Major architecture change needs validation

---

## Summary: What Changed from Original Plan

| Original | Revised | Reason |
|----------|---------|--------|
| Task-adaptive retrieval | **Removed** | Violates universal endpoint principle |
| Relationship weighting in CP1 | **Moved to CP3** | Belongs with weighing/forgetting |
| CP1 = Retrieval only | **CP1 = Ingestion + Retrieval** | Ingestion is prerequisite |
| CP2 = Ingestion | **CP2 = Profile + Consolidation** | Reordered |
| Single eval approach | **Eval appropriateness analysis** | Some features not measurable |

---

*Next action: Goals → Notes rename, then Checkpoint 1 implementation*
