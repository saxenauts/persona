# Evaluation & Release Timeline: Jan 27 – Feb 10, 2026

**Purpose**: Detailed git-backed timeline of evaluation/release work with commit hashes, dates, touched files, and inferred intent.

**Scope**: All commits from Jan 27 through Feb 10, 2026 on `refactor/v0.3-cognitive-memory` branch.

**Total Commits**: 50 commits across 14 days.

---

## Phase 1: Memeplex Serialization Fixes (Jan 28)

### Commit 1: `50d13535` – Jan 28, 17:37 UTC-8
**Message**: `fix(eval): add as_of parameter to memeplex for eval compatibility`

**Intent**: Enable eval harness to anchor memeplex snapshots to dataset timestamps (year-2000 bug prevention).

**Files Changed**:
- `persona/services/consolidation_service.py` (+3, -1)
- `server/routers/graph_api.py` (+13, -2)

**Details**: Added optional `as_of` parameter to `refresh_memeplex()` and exposed via `/users/{user_id}/memeplex?as_of=<timestamp>` endpoint. Allows eval to fetch memeplex state at specific points in time.

---

### Commit 2: `bfdabe21` – Jan 28, 18:06 UTC-8
**Message**: `fix(consolidation): fix timezone comparison and datetime serialization bugs`

**Intent**: Resolve "can't compare offset-naive and offset-aware datetimes" and "Object of type datetime is not JSON serializable" errors blocking memeplex refresh.

**Files Changed**:
- `persona/services/consolidation_service.py` (+9, -2)

**Details**: 
- Fixed `_normalize_tz()` to handle timezone-aware datetimes with `astimezone()`
- Normalize 'now' to UTC before passing to Memeplex
- Enables memeplex refresh to complete successfully

---

### Commit 3: `b6b6113c` – Jan 28, 18:22 UTC-8
**Message**: `fix(memeplex): use model_dump(mode='json') to serialize datetime fields`

**Intent**: Fix JSON serialization of MemoryStats datetime fields (earliest_memory, latest_memory).

**Files Changed**:
- `persona/core/memory_store.py` (+1, -1)

**Details**: Pydantic's `model_dump(mode='json')` automatically converts datetime to ISO strings before `json.dumps()`.

---

## Phase 2: Baseline Establishment & Failure Analysis (Jan 30)

### Commit 4: `a605c5cd` – Jan 30, 09:29 UTC-8
**Message**: `feat: complete truth-first baseline with verified 65.3% PersonaMem`

**Intent**: Freeze baseline v1 with comprehensive audit, ablation harness, entity dedup, and honest failure analysis.

**Files Changed**:
- `.opencode/FAILURE_ANALYSIS.md` (+468, -238)
- `BASELINE_v1.yaml` (+119)
- `persona/adapters/persona_adapter.py` (+117, -1)
- `persona/core/memory_store.py` (+67)
- `scripts/ablation_runner.py` (+583)

**Key Findings**:
- **Authoritative accuracy**: 65.3% (150q, 3 seeds: 42, 123, 456)
- **Retrieval NOT bottleneck**: 0.829 both ways
- **Answer selection is primary issue**: 43% generic responses
- **Entity dedup**: Implemented with 0.9 threshold to reduce explosion
- **Integration agent**: Not creating links (links=0)
- **Memeplex**: Pending ablation measurement

**Deliverables**:
- Audit reconciliation resolving 70% vs 51.4% discrepancy
- Ablation harness for paired component testing
- Entity dedup with cross-session fuzzy matching
- Failure analysis identifying answer selection bottleneck
- Git tag `baseline-v1` frozen

---

### Commit 5: `967d650f` – Jan 30, 09:40 UTC-8
**Message**: `fix(prompts): improve answer selection to reduce generic responses`

**Intent**: Address 43% of PersonaMem failures where model had good retrieval but chose generic responses.

**Files Changed**:
- `persona/llm/prompts.py` (+53, -11)

**Changes**:
- Split `answer_policy` into `retrieval_policy` + `answering_rules`
- Add explicit guidance to infer from behavioral evidence (repeated participation → enjoyment)
- Prefer personalized over generic responses when evidence exists
- Handle temporal evolution (prefer recent evidence)
- Explicit MCQ response selection guidance

**Expected Impact**: +5-10% PersonaMem accuracy

---

### Commit 6: `33900e3` – Jan 30, 09:33 UTC-8
**Message**: `chore: mark all 47 acceptance criteria complete in persona-truth-first plan`

**Intent**: Document completion of truth-first baseline sprint.

**Files Changed**: Plan metadata only.

---

### Commit 7: `9f15c73` – Jan 30, 09:42 UTC-8
**Message**: `docs: add session continuation summary`

**Intent**: Document session handling patterns discovered during baseline work.

**Files Changed**: Documentation only.

---

### Commit 8: `e463cb6` – Jan 30, 10:00 UTC-8
**Message**: `fix(prompts): revert benchmark hacks, apply production-general improvement`

**Intent**: Remove eval-specific hacks and apply sustainable improvements to production prompts.

**Files Changed**: `persona/llm/prompts.py`

---

### Commit 9: `e3df2f0` – Jan 30, 12:01 UTC-8
**Message**: `feat(ingestion): relax Psyche extraction to capture evaluative language`

**Intent**: Improve Psyche pillar extraction by capturing evaluative language (e.g., "I love X", "I hate Y").

**Files Changed**: `persona/services/ingestion_service.py`

---

### Commit 10: `106bfc3` – Jan 30, 12:07 UTC-8
**Message**: `feat(consolidation): add Psyche inference from behavioral patterns`

**Intent**: Infer Psyche traits from behavioral patterns (e.g., repeated participation → enjoyment).

**Files Changed**: `persona/services/consolidation_service.py`

---

### Commit 11: `e56358` – Jan 30, 12:25 UTC-8
**Message**: `chore: add PersonaMem validation eval results`

**Intent**: Log baseline eval results for audit trail.

**Files Changed**: `.opencode/` metadata

---

### Commit 12: `42fecd0` – Jan 30, 12:27 UTC-8
**Message**: `docs: mark plan complete and document learnings`

**Intent**: Capture learnings from psyche extraction improvement sprint.

**Files Changed**: Documentation only.

---

### Commit 13: `3debb8a` – Jan 30, 12:33 UTC-8
**Message**: `docs: document verification limitations and final status`

**Intent**: Honest documentation of what was verified vs. speculative.

**Files Changed**: Documentation only.

---

### Commit 14: `139eb8e` – Jan 30, 12:34 UTC-8
**Message**: `docs: add final summary for psyche extraction improvement`

**Intent**: Summarize psyche extraction improvements and their impact.

**Files Changed**: Documentation only.

---

## Phase 3: Regression Test Suite & Tiny Fixes (Jan 30)

### Commit 15: `92cf37c` – Jan 30, 15:06 UTC-8
**Message**: `fix(tools): use user timezone in resolve_date_range (H1)`

**Intent**: Fix timezone bug where users in different timezones got wrong results for 'today'/'yesterday' queries.

**Files Changed**:
- `persona/services/ingestion_service.py` (+4, -1)
- `persona/tools/memory.py` (+5, -1)
- `tests/unit/test_h1_timezone.py` (+172)

**Details**: Replace `datetime.now()` with timezone-aware datetime using `ctx.user_timezone` in `resolve_date_range_handler`. Added 5 regression test cases covering PST, EST, Tokyo, and UTC timezones.

---

### Commit 16: `60ad655` – Jan 30, 15:07 UTC-8
**Message**: `fix(ingestion): remove 1000x multiplier on microsecond offset (H4)`

**Intent**: Fix microsecond offset calculation bug.

**Files Changed**:
- `tests/regression/test_h4_microsecond_offset.py` (+135)

**Details**: Added regression test with 135 lines documenting the fix.

---

### Commit 17: `09bfe20` – Jan 30, 15:07 UTC-8
**Message**: `fix(context): normalize status comparison to lowercase (H3)`

**Intent**: Fix status comparison bug where case mismatches caused failures.

**Files Changed**:
- `persona/core/context.py` (+2, -1)
- `persona/services/consolidation_service.py` (+2, -1)
- `tests/regression/test_h3_status_case.py` (+151)

**Details**: Normalize status to lowercase before comparison. Added 151-line regression test.

---

### Commit 18: `4b05fae` – Jan 30, 15:15 UTC-8
**Message**: `fix(tools): push date filtering into graph query for browse (H2)`

**Intent**: Optimize browse tool by pushing date filtering into Neo4j query instead of post-processing.

**Files Changed**:
- `persona/core/backends/neo4j_graph.py` (+33, -1)
- `persona/core/memory_store.py` (+29, -1)
- `persona/services/persona_service.py` (+31, -1)
- `persona/tools/memory.py` (+24, -12)
- `tests/regression/test_h2_browse_historical.py` (+214)
- `tests/regression/test_h5_retriever_wired.py` (+91)
- `tests/unit/test_services.py` (+103, -37)

**Details**: Major refactor moving date filtering from Python to Cypher. Added comprehensive regression tests for historical browsing and retriever integration.

---

### Commit 19: `2b66a88` – Jan 30, 15:21 UTC-8
**Message**: `fix(tests): correct fixture usage in H5 regression tests`

**Intent**: Fix test fixture bugs in H5 retriever tests.

**Files Changed**:
- `tests/regression/test_h5_retriever_wired.py` (+64, -10)

**Details**: 
- Remove incorrect async for loop with isolated_graph_ops fixture
- Manually create and manage Neo4j connections
- Fix `delete_user_data` → `delete_user` method name
- Use 2 hours ago instead of yesterday for reliable retrieval window
- All 3 tests now PASS

---

## Phase 4: Benchmark Validation & Documentation (Jan 30)

### Commit 20: `e944f9c` – Jan 30, 16:29 UTC-8
**Message**: `docs: add reproducibility package for v0.3 benchmarks`

**Intent**: Document reproducibility methodology for v0.3 benchmarks.

**Files Changed**: `docs/` and `release_artifacts/`

---

### Commit 21: `b5e563d` – Jan 30, 16:33 UTC-8
**Message**: `docs: update benchmarks with verified v0.3 results`

**Intent**: Update benchmark documentation with verified results.

**Files Changed**: `docs/BENCHMARKS.md`

---

### Commit 22: `a75944c` – Jan 30, 16:34 UTC-8
**Message**: `chore: mark benchmark-validation-sprint complete`

**Intent**: Close benchmark validation sprint.

**Files Changed**: Plan metadata only.

---

## Phase 5: Experimental Eval-Strict Rules (Jan 31)

### Commit 23: `c25022c` – Jan 31, 11:32 UTC-8
**Message**: `feat(prompts): add eval-strict rules for ordering and factual questions`

**Intent**: Add specialized prompt rules for ordering and factual questions in eval context.

**Files Changed**:
- `persona/llm/prompts.py` (+33, -1)

**Details**: Added 32 lines of eval-specific guidance for ordering constraints and factual accuracy.

---

### Commit 24: `5753a2d` – Jan 31, 11:33 UTC-8
**Message**: `feat(tools): strengthen timeline schema with ordering mandate`

**Intent**: Enforce ordering constraints in timeline tool schema.

**Files Changed**:
- `persona/tools/schemas.py` (+13, -5)

**Details**: Updated timeline schema with explicit ordering mandate.

---

### Commit 25: `56b8aca` – Jan 31, 12:23 UTC-8
**Message**: `fix(prompts): scope eval rules to ordering questions only`

**Intent**: Limit eval-strict rules to ordering questions only (not all questions).

**Files Changed**: `persona/llm/prompts.py`

---

### Commit 26: `2667fa1` – Jan 31, 12:37 UTC-8
**Message**: `feat(prompts): add multi-hop retrieval guidance for cross-session aggregation`

**Intent**: Add guidance for multi-hop retrieval across sessions.

**Files Changed**: `persona/llm/prompts.py`

---

## Phase 6: Revert Experimental Rules (Jan 31)

### Commit 27: `1a9ac70` – Jan 31, 13:18 UTC-8
**Message**: `Revert "feat(prompts): add multi-hop retrieval guidance for cross-session aggregation"`

**Intent**: Revert multi-hop guidance (experimental, not production-ready).

**Files Changed**: `persona/llm/prompts.py` (-1 insertion)

---

### Commit 28: `d457dd7` – Jan 31, 13:18 UTC-8
**Message**: `Revert "fix(prompts): scope eval rules to ordering questions only"`

**Intent**: Revert scoped eval rules.

**Files Changed**: `persona/llm/prompts.py`

---

### Commit 29: `c281c9a` – Jan 31, 13:18 UTC-8
**Message**: `Revert "feat(tools): strengthen timeline schema with ordering mandate"`

**Intent**: Revert timeline schema changes.

**Files Changed**: `persona/tools/schemas.py`

---

### Commit 30: `600b76a` – Jan 31, 13:18 UTC-8
**Message**: `Revert "feat(prompts): add eval-strict rules for ordering and factual questions"`

**Intent**: Revert eval-strict rules (experimental, not production-ready).

**Files Changed**: `persona/llm/prompts.py` (-32 insertions)

**Reasoning**: These experimental rules were tested but reverted because they were too specialized for eval and didn't generalize to production. The baseline approach (answer selection improvement) proved more effective.

---

## Phase 7: Observability & Prompt Refinement (Feb 4)

### Commit 31: `d9f601d` – Feb 4, 21:03 UTC-8
**Message**: `fix(retrieval): normalize timezone comparisons to UTC`

**Intent**: Fix timezone comparison bugs in retrieval layer.

**Files Changed**:
- `persona/core/retrieval.py` (+36, -7)

**Details**: Add `_to_utc()` helper to handle timezone-naive datetimes. Fixes comparison issues between UTC and local timestamps.

**Attribution**: Ultraworked with Sisyphus

---

### Commit 32: `caac789` – Feb 4, 21:03 UTC-8
**Message**: `fix(persona-service): preserve working memory section structure`

**Intent**: Avoid double-wrapping working memory prose with redundant headers.

**Files Changed**:
- `persona/services/persona_service.py` (+8, -4)

**Details**: Working memory prose already includes `<user>/<recent_context>/<active_context>` sections. Don't add redundant headers.

**Attribution**: Ultraworked with Sisyphus

---

### Commit 33: `3c28268` – Feb 4, 21:03 UTC-8
**Message**: `fix(tools): handle dict/string patch_json and normalize session filter`

**Intent**: Improve integration tool robustness and observability.

**Files Changed**:
- `persona/tools/integration.py` (+16, -3)
- `server/routers/graph_api.py` (+11, -1)

**Details**:
- Accept `patch_json` as str or dict in `commit_patch_handler`
- Handle 'all'/'*' `session_id` to process all sessions
- Extend `IntegrateResponse` with observability fields

**Attribution**: Ultraworked with Sisyphus

---

### Commit 34: `655026f` – Feb 4, 21:04 UTC-8
**Message**: `feat(integration): add observability counters and temporal link guidance`

**Intent**: Improve integration agent observability and link creation.

**Files Changed**:
- `persona/services/integration_agent.py` (+46, -14)

**Details**:
- Add separate `links_created` and `operations_applied` counters
- Update prompt with NEXT/PREVIOUS temporal link guidance
- Add CONTRADICTS link guidance for conflicting facts
- Fix `commit_patch` result parsing for better error capture

**Attribution**: Ultraworked with Sisyphus

**Expected Impact**: Better observability into link creation (was 0 in baseline).

---

### Commit 35: `e3cdcbf` – Feb 4, 21:04 UTC-8
**Message**: `feat(ingestion): add anti-bloat guidance for entity extraction`

**Intent**: Reduce entity explosion (7.36 median entity/episode ratio).

**Files Changed**:
- `persona/services/ingestion_service.py` (+7)

**Details**:
- Only create entities for stable referents (people, orgs, places, projects)
- Prefer umbrella entity + attributes over many atomic entities
- Fold ephemeral nouns into Episodes

**Attribution**: Ultraworked with Sisyphus

---

### Commit 36: `0711845` – Feb 4, 21:04 UTC-8
**Message**: `fix(prompts): update doc reference path`

**Intent**: Fix broken documentation reference in prompts.

**Files Changed**:
- `persona/llm/prompts.py` (+2, -1)

**Details**: Point learnings reference to `.opencode/eval/` where it was moved.

**Attribution**: Ultraworked with Sisyphus

---

### Commit 37: `efe31ef` – Feb 4, 21:04 UTC-8
**Message**: `docs: align architecture docs with LLM-first vision`

**Intent**: Document LLM-first design principles and working memory composition.

**Files Changed**:
- `docs/ARCHITECTURE.md` (+24)
- `docs/MEMORY_MODEL.md` (+32)
- `docs/architecture/ARCHITECTURE.md` (+18, -3)

**Details**:
- Add LLM-first design principle (no manual routing, tool choice is model-driven)
- Document working memory composition (`<user>/<recent_context>/<active_context>`)
- Add session handling notes (product-defined triggers)
- Document future roadmap (quantification, dreaming)

**Attribution**: Ultraworked with Sisyphus

---

## Phase 8: Testing & Cleanup (Feb 4)

### Commit 38: `c254d42` – Feb 4, 21:05 UTC-8
**Message**: `test(context): add episode link rendering test`

**Intent**: Verify episode links are rendered with 'preceded by' text.

**Files Changed**:
- `tests/unit/test_context.py` (+34)

**Attribution**: Ultraworked with Sisyphus

---

### Commit 39: `35e1af2` – Feb 4, 21:05 UTC-8
**Message**: `chore: remove completed sisyphus plans`

**Intent**: Clean up completed plan documentation.

**Files Changed**: `.sisyphus/notepads/` cleanup

---

### Commit 40: `c0ecfd5` – Feb 4, 21:04 UTC-8
**Message**: `chore: cleanup old evaluation docs and plans`

**Intent**: Remove obsolete evaluation documentation.

**Files Changed**: `.opencode/` cleanup

---

### Commit 41: `d71b290` – Feb 4, 21:32 UTC-8
**Message**: `chore(repo): remove memory-evals submodule, update paths to sibling`

**Intent**: Clean up orphaned gitlink and update paths.

**Files Changed**:
- `.gitignore` (+1)
- `memory-evals` (-1, removed gitlink)
- `scripts/ablation_runner.py` (+2, -1)

**Details**:
- Remove orphaned gitlink to memory-evals/ (was mode 160000 with no .gitmodules)
- Add memory-evals/ to .gitignore to prevent re-tracking
- Fix ablation_runner.py path: `parent.parent` → `parent.parent.parent`
- memory-evals now lives as sibling at `../memory-evals/`

---

## Summary by Category

### Serialization & Compatibility Fixes (3 commits)
- Memeplex `as_of` parameter for eval anchoring
- Timezone-aware datetime handling
- JSON serialization of datetime fields

### Baseline & Analysis (1 major commit)
- Truth-first baseline v1 with 65.3% PersonaMem accuracy
- Comprehensive failure analysis identifying answer selection as bottleneck
- Entity dedup implementation
- Ablation harness for component testing

### Prompt Improvements (2 commits)
- Answer selection guidance (43% generic response problem)
- Psyche inference from behavioral patterns

### Regression Test Suite (5 commits, H1-H5)
- **H1**: Timezone handling in date range resolution
- **H2**: Date filtering optimization in browse tool
- **H3**: Status comparison case normalization
- **H4**: Microsecond offset calculation
- **H5**: Retriever integration and fixture management

### Experimental Rules (4 commits + 4 reverts)
- Eval-strict rules for ordering/factual questions
- Timeline schema ordering mandate
- Multi-hop retrieval guidance
- **All reverted** on Jan 31 (not production-ready)

### Observability & Production Hardening (6 commits)
- Timezone UTC normalization in retrieval
- Working memory section structure preservation
- Integration tool robustness (dict/string patch_json)
- Observability counters for link creation
- Anti-bloat guidance for entity extraction
- Documentation reference path fix

### Documentation & Cleanup (5 commits)
- Architecture docs alignment with LLM-first vision
- Episode link rendering test
- Benchmark reproducibility package
- Plan completion markers
- Submodule cleanup

---

## Key Insights

### Evaluation Methodology
- **Baseline**: 65.3% PersonaMem accuracy (150 questions, 3 seeds)
- **Methodology**: Honest documentation with verified vs. speculative separation
- **Bottleneck**: Answer selection (43% generic responses), not retrieval (0.829 both ways)

### Technical Debt Addressed
- **H1-H5 bugs**: Timezone, microsecond offset, status case, date filtering, fixture management
- **Entity bloat**: 7.36 median entity/episode ratio → anti-bloat guidance added
- **Link creation**: Integration agent not creating links (0) → observability counters added

### Production Readiness
- Experimental eval-strict rules reverted (too specialized)
- Focus on sustainable improvements (answer selection, psyche inference)
- Comprehensive regression test suite (H1-H5)
- Observability instrumentation for deep logging

### Release Artifacts
- `BASELINE_v1.yaml`: Frozen baseline configuration
- `scripts/ablation_runner.py`: Paired component testing harness
- `release_artifacts/methodology.md`: Reproducibility documentation
- Git tag `baseline-v1`: Immutable baseline reference

---

## Timeline Visualization

```
Jan 28  │ Memeplex serialization fixes (3 commits)
        │
Jan 30  │ Baseline v1 + failure analysis (1 major)
        │ Answer selection improvement (1)
        │ Psyche inference (2)
        │ Regression test suite H1-H5 (5)
        │ Benchmark validation (3)
        │
Jan 31  │ Experimental eval-strict rules (4 commits)
        │ Revert all experimental rules (4 commits)
        │
Feb 4   │ Observability & hardening (6 commits)
        │ Documentation & cleanup (5 commits)
        │
Total: 50 commits across 14 days
```

---

## Files Most Frequently Modified

| File | Commits | Purpose |
|------|---------|---------|
| `persona/llm/prompts.py` | 8 | Prompt engineering (answer selection, eval rules, doc refs) |
| `persona/services/consolidation_service.py` | 4 | Memeplex refresh, timezone handling, psyche inference |
| `persona/tools/memory.py` | 3 | Tool implementations (timezone, date filtering) |
| `tests/regression/` | 5 | H1-H5 regression test suite |
| `persona/core/` | 4 | Serialization, context, retrieval fixes |
| `docs/` | 3 | Architecture alignment, reproducibility |

---

## Evaluation Readiness Checklist

- ✅ Baseline frozen (v1) with git tag
- ✅ Failure analysis documented (answer selection bottleneck identified)
- ✅ Regression test suite (H1-H5) passing
- ✅ Observability instrumentation added
- ✅ Entity dedup implemented (0.9 threshold)
- ✅ Psyche inference working (0% generic responses in partial validation)
- ✅ Reproducibility package documented
- ✅ Architecture docs aligned with LLM-first vision
- ⚠️ Integration agent link creation (observability added, impact TBD)
- ⚠️ Full PersonaMem eval pending (partial validation: 50q, seed 42)

