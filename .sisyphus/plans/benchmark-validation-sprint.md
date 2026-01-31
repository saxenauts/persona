# Benchmark Validation Sprint

## TL;DR

> **Quick Summary**: Fix 5 high-severity bugs that invalidate benchmarks, then run paper-grade evals with full receipts to get honest numbers for v0.3 release.
> 
> **Deliverables**:
> - H1-H5 bugs fixed with regression tests
> - PersonaMem: 70%+ (150q, 5 seeds, 95% CI)
> - LongMemEval: 80%+ (170q, 5 seeds, 95% CI)
> - Full reproducibility package (docker, poetry.lock, per-question logs)
> 
> **Estimated Effort**: Medium-Large (5-7 days)
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Bug Fixes → Validation Gate → Paper-Grade Runs

---

## Context

### Original Request
Get paper-grade benchmark numbers for Persona v0.3 release. No pre-announcements, no "baseline now improvements later" - do it right the first time.

### Current State (from 236 eval runs analyzed)

| Benchmark | Current | Target | Gap |
|-----------|---------|--------|-----|
| PersonaMem | 65.3% (150q/3 seeds) | 70%+ (5 seeds) | Psyche inference should close this |
| LongMemEval | 78.8% (170q/1 seed) | 80%+ (5 seeds) | Need multi-seed validation |
| BEAM | 0 valid runs | Do NOT mention | Infrastructure broken |

### Root Cause Analysis

**Why 65.3% and not higher?**
- 85% of failures have excellent recall (>0.8) - retrieval is NOT the bottleneck
- 43% of high-recall failures: **generic response selection** (Psyche inference fix targets this)
- 29% of high-recall failures: **sentiment evolution confusion**
- 19% of high-recall failures: **missing evidence**

**What's blocking improvement?**
1. **H1-H5 bugs invalidate any benchmark results** - must fix first
2. **Psyche inference (commit 106bfc3) not validated** - expected to add 5-10%
3. **Single-seed runs have high variance** - need 5 seeds for credibility

### Metis Review Findings

**Critical guardrails identified**:
- Methodology lock before runs (no mid-run changes)
- Pre-register seeds (42, 123, 456, 789, 1337)
- Bug fixes isolated with smoke tests between
- Go/no-go thresholds defined before running

---

## Work Objectives

### Core Objective
Get paper-grade benchmark numbers with full receipts that can withstand HN and academic scrutiny.

### Concrete Deliverables
1. H1-H5 bugs fixed with regression tests in CI
2. PersonaMem: Mean accuracy with 95% CI < ±3%
3. LongMemEval: Mean accuracy with 95% CI < ±3%
4. Reproducibility package: docker-compose.yml, poetry.lock, model_config.json, per-question outputs
5. Methodology document: pre-registered protocol

### Definition of Done
- [ ] `pytest tests/regression/` passes (all H1-H5 fixes verified)
- [ ] PersonaMem 5-seed mean > 68% (baseline was 65.3%)
- [ ] LongMemEval 5-seed mean > 78% (baseline was 78.8%)
- [ ] 95% CI width < 5% for both benchmarks
- [ ] `release_artifacts/` contains full reproducibility package

### Must Have
- Multi-seed runs (5 seeds minimum)
- Per-question outputs for error analysis
- Locked methodology before official runs
- Regression tests for each bug fix

### Must NOT Have (Guardrails)
- **NO cherry-picking seeds** - use pre-registered seeds only
- **NO mid-run code changes** - methodology locks before runs
- **NO BEAM claims** - 0 valid runs means it doesn't exist for v0.3
- **NO prompt optimization during eval phase** - Psyche inference (106bfc3) is the last change
- **NO single-seed headline numbers** - all claims require ≥3 seeds

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest + memory-evals harness)
- **User wants tests**: TDD for bug fixes, automated for evals
- **Framework**: pytest for unit/regression, memory-evals for benchmarks

### Automated Verification Protocol

**For Bug Fixes (TDD)**:
```bash
# Each H1-H5 fix needs:
# 1. Failing test that demonstrates the bug
pytest tests/regression/test_h{N}_bug.py  # FAIL before fix

# 2. Fix applied
# 3. Same test passes
pytest tests/regression/test_h{N}_bug.py  # PASS after fix

# 4. Full regression suite still passes
pytest tests/unit/ -x --tb=short
```

**For Benchmark Runs**:
```bash
# Single validation run (gate before full eval)
cd memory-evals
poetry run python -m mem_eval.runner \
  --adapter persona \
  --benchmark personamem \
  --questions 150 \
  --seed 42 \
  --output results/validation/

# Assert: accuracy > 68% (baseline + expected improvement)
# Assert: generic_response_rate < 30% (was ~43%)
```

**For Paper-Grade Runs**:
```bash
# Multi-seed PersonaMem
for seed in 42 123 456 789 1337; do
  poetry run python -m mem_eval.runner \
    --adapter persona \
    --benchmark personamem \
    --questions 150 \
    --seed $seed \
    --output results/paper_grade/personamem_seed_${seed}/
done

# Calculate 95% CI
poetry run python scripts/calculate_ci.py results/paper_grade/personamem_*
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Day 1-2): Bug Fixes - CAN PARALLELIZE
├── Task 1: Fix H1 (timezone bug)
├── Task 2: Fix H2 (browse filtering bug)
├── Task 3: Fix H3 (status case mismatch)
├── Task 4: Fix H4 (microsecond offset bug)
└── Task 5: Fix H5 (Retriever not wired)

Wave 2 (Day 3): Validation Gate - SEQUENTIAL
├── Task 6: Smoke test all fixes together
└── Task 7: Single-seed validation run (GATE)

Wave 3 (Day 4-5): Paper-Grade Runs - CAN PARALLELIZE
├── Task 8: 5-seed PersonaMem runs
├── Task 9: 5-seed LongMemEval runs
└── Task 10: Generate reproducibility package

Wave 4 (Day 6): Documentation - SEQUENTIAL
└── Task 11: Update README with verified numbers
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1-5 | None | 6 | Each other |
| 6 | 1-5 | 7 | None |
| 7 | 6 | 8, 9, 10 | None (GATE) |
| 8 | 7 | 11 | 9, 10 |
| 9 | 7 | 11 | 8, 10 |
| 10 | 7 | 11 | 8, 9 |
| 11 | 8, 9, 10 | None | None (final) |

---

## TODOs

### Phase 1: Bug Fixes (Wave 1)

- [x] 1. Fix H1: Timezone bug in resolve_date_range

  **What to do**:
  - Change `datetime.now()` at line 538 to use `ctx.timezone` from ToolContext
  - Add regression test with user_timezone != server timezone
  - Verify "today"/"yesterday" resolves correctly for different timezones

  **Must NOT do**:
  - Don't change resolve_date_range behavior beyond timezone fix
  - Don't add new date parsing features

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file change, clear fix, well-scoped
  - **Skills**: [`git-master`]
    - `git-master`: Atomic commit after fix

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: Task 6 (smoke test)
  - **Blocked By**: None

  **References**:
  - `persona/tools/memory.py:538-540` - Bug location: `now = datetime.now()` ignores timezone
  - `persona/tools/context.py` - ToolContext has `timezone` field to use
  - `.opencode/archive/codex_split_20260129/CODEX_REVIEW_NOTES.md:17-26` - H1 description

  **Acceptance Criteria**:
  ```bash
  # Create regression test
  pytest tests/regression/test_h1_timezone.py -v
  # Assert: Test exists and PASSES
  # Test should verify: resolve_date_range("today", ctx_with_pst_timezone) returns PST date, not server date
  ```

  **Commit**: YES
  - Message: `fix(tools): use user timezone in resolve_date_range (H1)`
  - Files: `persona/tools/memory.py`, `tests/regression/test_h1_timezone.py`
  - Pre-commit: `pytest tests/regression/test_h1_timezone.py`

---

- [x] 2. Fix H2: browse() filtering bug for historical ranges

  **What to do**:
  - Push date range filtering into graph query (Cypher WHERE clause)
  - Remove Python-side filtering that drops memories outside 2*limit
  - Add test for "what happened in June 2023" style queries

  **Must NOT do**:
  - Don't change browse() API signature
  - Don't add new parameters

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Requires understanding graph query patterns, moderate complexity
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5)
  - **Blocks**: Task 6 (smoke test)
  - **Blocked By**: None

  **References**:
  - `persona/tools/memory.py:405-437` - Bug location: fetches limit*2, then Python filters
  - `persona/core/backends/neo4j_vector.py` - Graph query methods to add date filtering
  - `.opencode/archive/codex_split_20260129/CODEX_REVIEW_NOTES.md:28-34` - H2 description

  **Acceptance Criteria**:
  ```bash
  pytest tests/regression/test_h2_browse_historical.py -v
  # Assert: browse(date_start="2023-06-01", date_end="2023-06-30") returns ALL June 2023 memories
  # Not just the most recent 2*limit
  ```

  **Commit**: YES
  - Message: `fix(tools): push date filtering into graph query for browse (H2)`
  - Files: `persona/tools/memory.py`, `persona/core/backends/neo4j_vector.py`, `tests/regression/test_h2_browse_historical.py`

---

- [x] 3. Fix H3: Status case mismatch ("active" vs "COMPLETED")

  **What to do**:
  - Normalize status to lowercase at write-time in NoteMemory
  - Update context.py filtering to use `.lower()` comparison
  - Update consolidation_service.py to use `.lower()` comparison
  - Add test verifying completed notes don't appear in active context

  **Must NOT do**:
  - Don't change status enum values in API
  - Don't break existing notes in database (add migration if needed)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: String normalization, straightforward
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5)
  - **Blocks**: Task 6 (smoke test)
  - **Blocked By**: None

  **References**:
  - `persona/models/memory.py:81-115` - NoteMemory.status defaults to lowercase "active"
  - `persona/core/context.py:120` - Filters `status != "COMPLETED"` (uppercase)
  - `persona/services/consolidation_service.py:57-59` - Same uppercase check
  - `.opencode/archive/codex_split_20260129/CODEX_REVIEW_NOTES.md:36-44` - H3 description

  **Acceptance Criteria**:
  ```bash
  pytest tests/regression/test_h3_status_case.py -v
  # Assert: Note with status="completed" does NOT appear in format_working_memory_prose()
  # Assert: Note with status="COMPLETED" does NOT appear in format_working_memory_prose()
  ```

  **Commit**: YES
  - Message: `fix(context): normalize status comparison to lowercase (H3)`
  - Files: `persona/core/context.py`, `persona/services/consolidation_service.py`, `tests/regression/test_h3_status_case.py`

---

- [x] 4. Fix H4: Microsecond offset bug in ingestion

  **What to do**:
  - Change `microsecond * 1000` to just `microsecond` at line 333
  - Current code can shift events by up to 16 minutes (999 * 1000 = 999000 microseconds = 16.65 minutes)
  - Add test asserting event_time offsets are < 1 second apart

  **Must NOT do**:
  - Don't change the sequencing logic, just the multiplier
  - Don't break existing event ordering for new ingests

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single line change with clear fix
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5)
  - **Blocks**: Task 6 (smoke test)
  - **Blocked By**: None

  **References**:
  - `persona/services/ingestion_service.py:333` - Bug: `memory_seq = observed_at.microsecond * 1000`
  - `.opencode/archive/codex_split_20260129/CODEX_REVIEW_NOTES.md:46-53` - H4 description
  - Commit `4766970` introduced this behavior

  **Acceptance Criteria**:
  ```bash
  pytest tests/regression/test_h4_microsecond_offset.py -v
  # Assert: For memories in same ingest, event_time differences are < 1 second
  # Assert: Event ordering is preserved within an ingest
  ```

  **Commit**: YES
  - Message: `fix(ingestion): remove 1000x multiplier on microsecond offset (H4)`
  - Files: `persona/services/ingestion_service.py`, `tests/regression/test_h4_microsecond_offset.py`

---

- [x] 5. Fix H5: Wire Retriever into PersonaService

  **What to do**:
  - Import and instantiate Retriever in PersonaService
  - Call `Retriever.get_working_memory()` in run_agent before building prompt
  - Include working memory context in system prompt alongside user_card + memeplex
  - Add stats for working_memory_chars in response

  **Must NOT do**:
  - Don't change Retriever implementation
  - Don't remove tool-based retrieval (this is additive)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Requires understanding service architecture, moderate complexity
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4)
  - **Blocks**: Task 6 (smoke test)
  - **Blocked By**: None

  **References**:
  - `persona/services/persona_service.py:33-88` - run_agent builds prompt only from user_card + memeplex
  - `persona/core/retrieval.py` - Retriever.get_working_memory() exists but never called
  - `.opencode/ARCHITECTURE_DEEP_DIVE.md:108-111` - Claims Retriever is used (doc drift)
  - `.opencode/archive/codex_split_20260129/CODEX_REVIEW_NOTES.md:55-64` - H5 description

  **Acceptance Criteria**:
  ```bash
  pytest tests/regression/test_h5_retriever_wired.py -v
  # Assert: PersonaService.run_agent() includes working_memory in prompt
  # Assert: Response stats include working_memory_chars > 0
  
  # Functional verification
  poetry run python -c "
  from persona.services.persona_service import PersonaService
  # ... setup code ...
  result = service.run_agent('What did I do yesterday?', include_stats=True)
  assert result.stats.get('working_memory_chars', 0) > 0
  "
  ```

  **Commit**: YES
  - Message: `fix(persona): wire Retriever.get_working_memory into run_agent (H5)`
  - Files: `persona/services/persona_service.py`, `tests/regression/test_h5_retriever_wired.py`

---

### Phase 2: Validation Gate (Wave 2)

- [x] 6. Smoke test all fixes together

  **What to do**:
  - Run full unit test suite to verify no regressions
  - Run integration tests to verify services work end-to-end
  - Create a git tag for the fixed state: `v0.3-pre-eval`

  **Must NOT do**:
  - Don't proceed to eval runs if any test fails
  - Don't make code changes at this point

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Just running tests and tagging
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Wave 1)
  - **Blocks**: Task 7 (validation run)
  - **Blocked By**: Tasks 1, 2, 3, 4, 5

  **References**:
  - `tests/unit/` - Unit test suite
  - `tests/integration/` - Integration test suite
  - `tests/regression/` - New regression tests from H1-H5 fixes

  **Acceptance Criteria**:
  ```bash
  # All tests pass
  poetry run pytest tests/ -v --tb=short
  # Exit code 0, no failures
  
  # Tag the fixed state
  git tag v0.3-pre-eval
  git log -1 --oneline
  # Assert: Tag exists at current commit
  ```

  **Commit**: NO (just tagging)

---

- [x] 7. Single-seed validation run (GATE)

  **What to do**:
  - Run PersonaMem with seed 42 on 150 questions
  - Check if accuracy > 68% (baseline 65.3% + expected Psyche inference lift)
  - Check if generic_response_rate < 35% (was ~43%)
  - This is a GATE: if it fails, investigate before proceeding

  **Must NOT do**:
  - Don't proceed to multi-seed runs if validation fails
  - Don't change code to "fix" validation issues (that violates methodology lock)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Running eval harness, interpreting results
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (GATE checkpoint)
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 8, 9, 10 (paper-grade runs)
  - **Blocked By**: Task 6 (smoke test)

  **References**:
  - `memory-evals/mem_eval/runner.py` - Eval runner
  - `memory-evals/results/competitor_full/persona_personamem/` - Previous 65.3% baseline
  - `.opencode/FAILURE_ANALYSIS.md` - Failure pattern breakdown (43% generic responses)

  **Acceptance Criteria**:
  ```bash
  cd memory-evals
  poetry run python -m mem_eval.runner \
    --adapter persona \
    --benchmark personamem \
    --questions 150 \
    --seed 42 \
    --output results/validation_gate/
  
  # Check results
  cat results/validation_gate/summary.json | jq '.accuracy'
  # Assert: accuracy > 0.68
  
  # Analyze failure patterns
  poetry run python scripts/analyze_failures.py results/validation_gate/
  # Assert: generic_response_rate < 0.35
  ```

  **GO/NO-GO Decision**:
  - **GO**: Accuracy > 68% AND generic_response_rate < 35% → proceed to Wave 3
  - **NO-GO**: Either metric fails → STOP and investigate. Options:
    - Debug why Psyche inference didn't help
    - Check if bug fixes introduced regressions
    - Consult with user before proceeding

  **Commit**: NO (just running eval)

---

### Phase 3: Paper-Grade Runs (Wave 3)

- [x] 8. 5-seed PersonaMem runs

  **What to do**:
  - Lock methodology in `release_artifacts/methodology.md`
  - Run PersonaMem with seeds [42, 123, 456, 789, 1337]
  - Calculate mean accuracy and 95% CI
  - Save all per-question outputs

  **Must NOT do**:
  - Don't change code after methodology lock
  - Don't discard runs with "bad" seeds
  - Don't re-run with different questions

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Long-running eval, statistical analysis
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 10)
  - **Blocks**: Task 11 (documentation)
  - **Blocked By**: Task 7 (validation gate)

  **References**:
  - `memory-evals/mem_eval/runner.py` - Eval runner
  - `memory-evals/scripts/calculate_ci.py` - CI calculation (may need to create)
  - Pre-registered seeds: 42, 123, 456, 789, 1337

  **Acceptance Criteria**:
  ```bash
  # Create methodology lock
  mkdir -p release_artifacts
  cat > release_artifacts/methodology.md << 'EOF'
  # Evaluation Methodology
  
  ## PersonaMem
  - Questions: 150 (fixed set)
  - Seeds: 42, 123, 456, 789, 1337
  - Model: GPT-5.2 via Azure Foundry
  - Commit: $(git rev-parse HEAD)
  - Date: $(date -I)
  EOF
  
  # Run all seeds
  for seed in 42 123 456 789 1337; do
    poetry run python -m mem_eval.runner \
      --adapter persona \
      --benchmark personamem \
      --questions 150 \
      --seed $seed \
      --output results/paper_grade/personamem_${seed}/
  done
  
  # Calculate statistics
  poetry run python -c "
  import json
  from pathlib import Path
  import numpy as np
  
  scores = []
  for seed in [42, 123, 456, 789, 1337]:
      with open(f'results/paper_grade/personamem_{seed}/summary.json') as f:
          scores.append(json.load(f)['accuracy'])
  
  mean = np.mean(scores)
  std = np.std(scores, ddof=1)
  ci_95 = 1.96 * std / np.sqrt(len(scores))
  
  print(f'PersonaMem: {mean:.1%} ± {ci_95:.1%} (95% CI)')
  print(f'Range: {min(scores):.1%} - {max(scores):.1%}')
  
  assert mean > 0.68, f'Mean {mean:.1%} below 68% threshold'
  assert ci_95 < 0.05, f'CI width {ci_95:.1%} exceeds 5% threshold'
  "
  ```

  **Commit**: NO (eval artifacts, not code)

---

- [x] 9. 5-seed LongMemEval runs

  **What to do**:
  - Run LongMemEval with seeds [42, 123, 456, 789, 1337]
  - Calculate mean accuracy and 95% CI
  - Save all per-question outputs

  **Must NOT do**:
  - Don't change code after methodology lock
  - Don't discard runs with "bad" seeds

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Long-running eval, statistical analysis
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 10)
  - **Blocks**: Task 11 (documentation)
  - **Blocked By**: Task 7 (validation gate)

  **References**:
  - `memory-evals/mem_eval/runner.py` - Eval runner
  - `memory-evals/results/competitor_full/persona_longmemeval/` - Previous 78.8% baseline

  **Acceptance Criteria**:
  ```bash
  for seed in 42 123 456 789 1337; do
    poetry run python -m mem_eval.runner \
      --adapter persona \
      --benchmark longmemeval \
      --questions 170 \
      --seed $seed \
      --output results/paper_grade/longmemeval_${seed}/
  done
  
  # Calculate statistics
  poetry run python -c "
  import json
  import numpy as np
  
  scores = []
  for seed in [42, 123, 456, 789, 1337]:
      with open(f'results/paper_grade/longmemeval_{seed}/summary.json') as f:
          scores.append(json.load(f)['accuracy'])
  
  mean = np.mean(scores)
  std = np.std(scores, ddof=1)
  ci_95 = 1.96 * std / np.sqrt(len(scores))
  
  print(f'LongMemEval: {mean:.1%} ± {ci_95:.1%} (95% CI)')
  assert mean > 0.78, f'Mean {mean:.1%} below 78% threshold'
  assert ci_95 < 0.05, f'CI width {ci_95:.1%} exceeds 5% threshold'
  "
  ```

  **Commit**: NO (eval artifacts, not code)

---

- [x] 10. Generate reproducibility package

  **What to do**:
  - Create `release_artifacts/` with:
    - `docker-compose.yml` (locked versions)
    - `poetry.lock` (exact deps)
    - `model_config.json` (exact model ID, temperature, etc.)
    - `methodology.md` (pre-registered protocol)
    - `results/` (all per-question outputs)
  - Verify package can reproduce results from clean environment

  **Must NOT do**:
  - Don't include API keys or secrets
  - Don't include raw eval datasets (link to source instead)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: File organization, not complex logic
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 9)
  - **Blocks**: Task 11 (documentation)
  - **Blocked By**: Task 7 (validation gate)

  **References**:
  - `docker-compose.yml` - Current docker config
  - `poetry.lock` - Current dependency lock
  - `server/config.py` - Model configuration

  **Acceptance Criteria**:
  ```bash
  ls release_artifacts/
  # Assert: Contains docker-compose.yml, poetry.lock, model_config.json, methodology.md
  
  # Verify completeness
  test -f release_artifacts/docker-compose.yml
  test -f release_artifacts/poetry.lock
  test -f release_artifacts/model_config.json
  test -f release_artifacts/methodology.md
  test -d release_artifacts/results/
  
  # Verify no secrets
  grep -r "API_KEY\|SECRET\|PASSWORD" release_artifacts/ && exit 1 || echo "No secrets found"
  ```

  **Commit**: YES
  - Message: `docs: add reproducibility package for v0.3 benchmarks`
  - Files: `release_artifacts/`

---

### Phase 4: Documentation (Wave 4)

- [x] 11. Update README with verified numbers

  **What to do**:
  - Replace current benchmark claims with verified multi-seed results
  - Add link to methodology and reproducibility package
  - Add "Benchmark Integrity" section explaining what was invalidated
  - Remove BEAM claims entirely (0 valid runs)

  **Must NOT do**:
  - Don't claim numbers higher than verified
  - Don't mention BEAM until valid runs exist
  - Don't include adjusted/audited numbers as primary metrics

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation update with precise language
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: NO (final task)
  - **Parallel Group**: Sequential (after Wave 3)
  - **Blocks**: None (final)
  - **Blocked By**: Tasks 8, 9, 10

  **References**:
  - `README.md` - Current benchmarks section (lines 45-60)
  - `release_artifacts/methodology.md` - Methodology to link
  - Calculated mean + CI from Tasks 8, 9

  **Acceptance Criteria**:
  ```bash
  # Verify README updated
  grep -q "PersonaMem.*±" README.md  # Has confidence interval
  grep -q "LongMemEval.*±" README.md  # Has confidence interval
  ! grep -q "BEAM" README.md  # No BEAM claims
  grep -q "methodology" README.md  # Links to methodology
  
  # Verify claims match actual results
  # (manual verification against summary.json files)
  ```

  **Commit**: YES
  - Message: `docs: update benchmarks with verified multi-seed results`
  - Files: `README.md`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `fix(tools): use user timezone in resolve_date_range (H1)` | memory.py, test_h1 | pytest test_h1 |
| 2 | `fix(tools): push date filtering into graph query (H2)` | memory.py, neo4j_vector.py, test_h2 | pytest test_h2 |
| 3 | `fix(context): normalize status comparison (H3)` | context.py, consolidation_service.py, test_h3 | pytest test_h3 |
| 4 | `fix(ingestion): remove 1000x microsecond multiplier (H4)` | ingestion_service.py, test_h4 | pytest test_h4 |
| 5 | `fix(persona): wire Retriever into run_agent (H5)` | persona_service.py, test_h5 | pytest test_h5 |
| 6 | (tag only) | v0.3-pre-eval | pytest tests/ |
| 10 | `docs: add reproducibility package` | release_artifacts/ | ls release_artifacts/ |
| 11 | `docs: update benchmarks with verified results` | README.md | grep verification |

---

## Success Criteria

### Verification Commands
```bash
# All regression tests pass
pytest tests/regression/ -v  # Expected: 5 tests, 0 failures

# PersonaMem multi-seed
cat results/paper_grade/personamem_*/summary.json | jq -s 'map(.accuracy) | add/length'
# Expected: > 0.68

# LongMemEval multi-seed
cat results/paper_grade/longmemeval_*/summary.json | jq -s 'map(.accuracy) | add/length'
# Expected: > 0.78

# Reproducibility package complete
ls release_artifacts/
# Expected: docker-compose.yml, poetry.lock, model_config.json, methodology.md, results/
```

### Final Checklist
- [ ] All H1-H5 bugs fixed with regression tests
- [ ] Validation gate passed (accuracy > 68%, generic_response_rate < 35%)
- [ ] PersonaMem: 5-seed mean > 68%, CI < 5%
- [ ] LongMemEval: 5-seed mean > 78%, CI < 5%
- [ ] Reproducibility package complete
- [ ] README updated with verified numbers only
- [ ] No BEAM claims anywhere
- [ ] All tests pass
