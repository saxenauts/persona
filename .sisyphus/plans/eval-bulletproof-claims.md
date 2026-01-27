# Work Plan: Bulletproof Evaluation Claims

## Context

### Goal
Build an evaluation framework that produces **statistically undeniable claims** of superiority over Honcho and Supermemory. NO RELEASE - just eval infrastructure and results.

### Constraints
- NO release, NO tagging, NO packaging changes
- NO LoCoMo (skip it)
- Focus: Honcho first, then Supermemory
- Must be bulletproof against ANY criticism

### Research Findings

**Academic Standards**: LoCoMo (ACL 2024), LongMemEval (ICLR 2025) = gold standard. Requirements: N >= 100, 95% CIs, p-values, effect sizes, 5+ random seeds.

**Current Eval Framework**: 4 adapters functional (Persona, Honcho, Mem0, Zep/Graphiti). BEAM, PersonaMem, LongMemEval available. Current accuracy: PersonaMem 63%, LongMemEval 64.1%, BEAM 67.6%. **CRITICAL GAPS**: No confidence intervals, no multi-run, no statistical tests.

**Competitors**: Honcho (demo.honcho.dev, adapter exists), Supermemory (MIT, 14.2K stars, needs adapter), Mem0 (adapter exists).

---

## Methodology Specification (FROZEN)

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Model | gpt-4o-mini |
| Temperature | 0.0 |
| Max tokens | 4096 |
| Timeout | 120s |

### Budget Parity (CRITICAL)
| Resource | Cap |
|----------|-----|
| LLM calls per ingest | 1 |
| LLM calls per query | 3 |
| Tool calls per turn | 5 |
| Retrieval top-K | 10 |

### Statistical Rigor
| Requirement | Value |
|-------------|-------|
| Sample size | N >= 100 per benchmark |
| Random seeds | [42, 123, 456, 789, 999] |
| Confidence level | 95% |
| Effect size | Cohen's d |
| Significance test | Paired t-test |

### Datasets
| Benchmark | Questions | Why |
|-----------|-----------|-----|
| PersonaMem | 519 | Apples-to-apples with competitors |
| LongMemEval | 500 | Academic credibility (ICLR 2025) |
| BEAM 100K | 100 | Comprehensive task coverage |

---

## Phase 1: Statistical Infrastructure

**Goal**: Add multi-run capability and statistical aggregation to the eval framework.
**Location**: `../memory-evals/mem_eval/`
**Estimated**: 1-2 sessions

---

### Task 1.1: Create stats module

- [ ] 1.1.1 Create `mem_eval/stats.py` with statistical functions

**What to do**:
- Create new file `mem_eval/stats.py`
- Implement `aggregate_results()`: mean, std, stderr, 95% CI
- Implement `compare_systems()`: paired t-test, p-value, Cohen's d
- Implement `format_ci()`: format confidence intervals for display

**Implementation sketch**:
```python
import numpy as np
from scipy import stats

def aggregate_results(accuracies: list[float]) -> dict:
    n = len(accuracies)
    mean = np.mean(accuracies)
    std = np.std(accuracies, ddof=1)
    stderr = std / np.sqrt(n)
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=stderr)
    return {"mean": mean, "std": std, "stderr": stderr, "ci_95": ci, "n": n}

def compare_systems(a_scores: list, b_scores: list) -> dict:
    t_stat, p_value = stats.ttest_rel(a_scores, b_scores)
    diff = np.array(a_scores) - np.array(b_scores)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    return {"t_stat": t_stat, "p_value": p_value, "cohens_d": cohens_d, "significant": p_value < 0.05}
```

**Acceptance Criteria**:
- [ ] `stats.py` exists with all functions
- [ ] Unit tests pass for statistical functions
- [ ] 95% CI calculated correctly

**Commit**: `feat(eval): add statistical aggregation module`

---

### Task 1.2: Add multi-seed support to runner

- [ ] 1.2.1 Modify `runner.py` to accept `--seeds` parameter
- [ ] 1.2.2 Implement `run_multi_seed()` function
- [ ] 1.2.3 Save per-seed results separately
- [ ] 1.2.4 Aggregate and report with CIs

**What to do**:
- Add `--seeds` CLI parameter (comma-separated list)
- Loop over seeds, set random seed before each run
- Save each run to `results/{adapter}_{benchmark}_seed{N}.json`
- After all runs, call `aggregate_results()` and save summary

**Must NOT do**:
- Don't change existing single-run behavior (default: seed=42)
- Don't break checkpoint/resume functionality

**Acceptance Criteria**:
- [ ] `mem-eval run --seeds 42,123,456` runs 3 times
- [ ] Per-run results saved separately
- [ ] Summary includes mean +/- 95% CI
- [ ] Backward compatible (no --seeds = single run)

**Commit**: `feat(eval): add multi-seed evaluation support`

---

### Task 1.3: Add comparison CLI command

- [ ] 1.3.1 Add `mem-eval compare` subcommand
- [ ] 1.3.2 Load results from two adapters
- [ ] 1.3.3 Compute paired statistics
- [ ] 1.3.4 Output comparison report

**What to do**:
- Add `compare` command to CLI
- Load results files for two systems
- Pair by question_id
- Call `compare_systems()` and format output

**Usage**:
```bash
mem-eval compare persona honcho --benchmark personamem
```

**Output format**:
```
Comparison: Persona vs Honcho (PersonaMem)

Persona: 72.3% [70.1, 74.5] (N=519, 5 seeds)
Honcho:  63.1% [60.8, 65.4] (N=519, 5 seeds)

Difference: +9.2 percentage points
p-value: < 0.001 (SIGNIFICANT)
Cohen's d: 0.42 (medium effect)
```

**Acceptance Criteria**:
- [ ] `mem-eval compare` command works
- [ ] Outputs p-value and effect size
- [ ] Works with multi-seed results

**Commit**: `feat(eval): add pairwise comparison command`

---

### Task 1.4: Create reporting template

- [ ] 1.4.1 Create `mem_eval/reporting.py`
- [ ] 1.4.2 Implement markdown table generator
- [ ] 1.4.3 Auto-generate interpretation text

**What to do**:
- Create function to generate standardized results table
- Include: system, mean, 95% CI, p-value, Cohen's d
- Add interpretation based on effect size thresholds

**Acceptance Criteria**:
- [ ] `mem-eval report` generates markdown
- [ ] All statistical elements included
- [ ] Interpretation auto-generated

**Commit**: `feat(eval): add statistical results reporting`

---

## Phase 2: Fairness Infrastructure

**Goal**: Ensure all comparisons are fair with documented methodology.
**Estimated**: 1-2 sessions

---

### Task 2.1: Implement LLM budget tracking

- [ ] 2.1.1 Create `mem_eval/budget.py` with BudgetTracker class
- [ ] 2.1.2 Integrate budget tracking into adapters
- [ ] 2.1.3 Log call counts per question
- [ ] 2.1.4 Add budget summary to results

**What to do**:
- Create BudgetTracker that counts LLM calls
- Inject into adapter query/ingest methods
- Log warnings if budget exceeded (don't abort)
- Include call counts in question logs

**Implementation**:
```python
class BudgetTracker:
    def __init__(self, max_ingest=1, max_query=3):
        self.max_ingest = max_ingest
        self.max_query = max_query
        self.counts = {"ingest": 0, "query": 0}
    
    def track(self, operation: str):
        self.counts[operation] += 1
        limit = getattr(self, f"max_{operation}")
        if self.counts[operation] > limit:
            logger.warning(f"Budget exceeded: {operation} {self.counts[operation]}/{limit}")
```

**Acceptance Criteria**:
- [ ] BudgetTracker logs all LLM calls
- [ ] Results include call counts
- [ ] Warnings logged for budget violations

**Commit**: `feat(eval): add LLM budget tracking`

---

### Task 2.2: Write methodology document

- [ ] 2.2.1 Create `docs/EVAL_METHODOLOGY.md` in persona repo
- [ ] 2.2.2 Document frozen parameters
- [ ] 2.2.3 Document statistical approach
- [ ] 2.2.4 Include reproducibility commands

**Content sections**:
1. Model Configuration (gpt-4o-mini, temp=0.0)
2. Budget Parity (1 ingest, 3 query calls)
3. Statistical Approach (5 seeds, 95% CI, paired t-test)
4. Datasets (PersonaMem, LongMemEval, BEAM)
5. Reproducibility (Docker command)

**Acceptance Criteria**:
- [ ] Document exists at `docs/EVAL_METHODOLOGY.md`
- [ ] All frozen parameters documented
- [ ] Version number included (v1.0)

**Commit**: `docs: add evaluation methodology specification v1.0`

---

### Task 2.3: Add Supermemory adapter (OPTIONAL)

- [ ] 2.3.1 Research Supermemory API/library
- [ ] 2.3.2 Create `mem_eval/adapters/supermemory_adapter.py`
- [ ] 2.3.3 Implement MemorySystem interface
- [ ] 2.3.4 Test with small dataset

**References**: https://github.com/supermemoryai/supermemory

**Note**: This is lower priority than Honcho comparison. Can be deferred if Supermemory setup is complex.

**Acceptance Criteria**:
- [ ] Adapter exists and implements interface
- [ ] Can run at least PersonaMem benchmark
- [ ] Budget tracking integrated

**Commit**: `feat(eval): add Supermemory adapter`

---

## Phase 3: Run Evaluations

**Goal**: Execute all benchmarks and generate comparison data.
**Estimated**: 2-3 sessions (mostly waiting for runs)

---

### Task 3.1: Run Persona baseline (5 seeds)

- [ ] 3.1.1 Run PersonaMem with 5 seeds
- [ ] 3.1.2 Run LongMemEval with 5 seeds  
- [ ] 3.1.3 Run BEAM 100K with 5 seeds
- [ ] 3.1.4 Verify results and save

**Commands**:
```bash
mem-eval run --benchmark personamem --adapter persona --seeds 42,123,456,789,999
mem-eval run --benchmark longmemeval --adapter persona --seeds 42,123,456,789,999
mem-eval run --benchmark beam --adapter persona --seeds 42,123,456,789,999
```

**Acceptance Criteria**:
- [ ] All 3 benchmarks completed
- [ ] Results include mean +/- 95% CI
- [ ] Results saved to versioned files

---

### Task 3.2: Run Honcho comparison (5 seeds)

- [ ] 3.2.1 Verify Honcho adapter works
- [ ] 3.2.2 Run PersonaMem with 5 seeds
- [ ] 3.2.3 Run LongMemEval with 5 seeds
- [ ] 3.2.4 Run BEAM 100K with 5 seeds

**Commands**:
```bash
mem-eval run --benchmark personamem --adapter honcho --seeds 42,123,456,789,999
mem-eval run --benchmark longmemeval --adapter honcho --seeds 42,123,456,789,999
mem-eval run --benchmark beam --adapter honcho --seeds 42,123,456,789,999
```

**Acceptance Criteria**:
- [ ] All 3 benchmarks completed for Honcho
- [ ] Budget caps verified in logs
- [ ] Results saved

---

### Task 3.3: Run Supermemory comparison (5 seeds) - OPTIONAL

- [ ] 3.3.1 Run PersonaMem with 5 seeds
- [ ] 3.3.2 Run LongMemEval with 5 seeds
- [ ] 3.3.3 Run BEAM 100K with 5 seeds

**Depends on**: Task 2.3 (Supermemory adapter)

---

### Task 3.4: Generate pairwise comparisons

- [ ] 3.4.1 Compare Persona vs Honcho (all benchmarks)
- [ ] 3.4.2 Compare Persona vs Supermemory (if available)
- [ ] 3.4.3 Generate summary report

**Commands**:
```bash
mem-eval compare persona honcho --benchmark personamem
mem-eval compare persona honcho --benchmark longmemeval
mem-eval compare persona honcho --benchmark beam
mem-eval report --output results/COMPARISON_SUMMARY.md
```

**Acceptance Criteria**:
- [ ] All comparisons generated
- [ ] p-values calculated
- [ ] Summary report with interpretation

---

## Phase 4: Reproducibility Package

**Goal**: Make all results reproducible by third parties.
**Estimated**: 1 session

---

### Task 4.1: Create Docker image

- [ ] 4.1.1 Create Dockerfile in memory-evals
- [ ] 4.1.2 Pin all dependencies
- [ ] 4.1.3 Include datasets
- [ ] 4.1.4 Test full eval in container

**Dockerfile**:
```dockerfile
FROM python:3.12-slim
WORKDIR /eval
COPY . .
RUN pip install -r requirements.lock.txt
COPY data/ data/
ENTRYPOINT ["python", "-m", "mem_eval"]
```

**Acceptance Criteria**:
- [ ] `docker build` succeeds
- [ ] `docker run persona-eval --help` works
- [ ] Full eval reproducible in container

**Commit**: `build: add Docker image for reproducible evaluation`

---

### Task 4.2: Create eval README

- [ ] 4.2.1 Write `README_EVAL.md` with reproduction steps
- [ ] 4.2.2 Document hardware requirements
- [ ] 4.2.3 Add troubleshooting section

**Acceptance Criteria**:
- [ ] README includes exact commands
- [ ] Third party could reproduce results
- [ ] Hardware requirements documented

**Commit**: `docs: add evaluation reproducibility instructions`

---

### Task 4.3: Generate final results summary

- [ ] 4.3.1 Create `results/EVALUATION_RESULTS.md`
- [ ] 4.3.2 Include all comparison tables
- [ ] 4.3.3 Include methodology summary
- [ ] 4.3.4 Include reproducibility command

**This is the artifact we can share/publish when ready.**

---

## Definition of Done

- [ ] Multi-run harness runs 5 seeds automatically
- [ ] Results include mean +/- 95% CI for each metric
- [ ] Pairwise comparisons include p-values and Cohen's d
- [ ] Methodology document is version-controlled
- [ ] All results reproducible with Docker
- [ ] Beat Honcho with p < 0.05 (or document honestly if not)

---

## Success Criteria

| Requirement | Threshold |
|-------------|-----------|
| Sample size | N >= 100 per benchmark |
| Confidence intervals | All results have 95% CI |
| Statistical significance | p < 0.05 for claimed wins |
| Effect size | Cohen's d reported |
| Multiple seeds | 5 seeds per benchmark |
| Budget parity | Same LLM calls verified |
| Reproducibility | Docker works |

---

## Risk Mitigation

**If Persona doesn't beat competitors**: Document honestly with CIs. Identify failure modes. Plan improvements. Credibility > hype.

**If adapter breaks**: Use API fallback. Skip that competitor (document why). Focus on available comparisons.

**If no statistical significance**: Report as "no significant difference". Still valuable finding.

---

## NOT IN SCOPE (Deferred to release plan)

- Packaging changes (pyproject.toml)
- Public API exports  
- CHANGELOG.md
- Release tagging
- GitHub release
- README updates for release

---

## Session Notes

**Date**: 2026-01-25
**Branch**: `refactor/v0.3-cognitive-memory`
**Goal**: Bulletproof eval claims, NOT release

**Competitors**:
1. Honcho (primary) - adapter exists
2. Supermemory (secondary) - needs adapter, optional
3. Mem0 (optional) - adapter exists

**NOT doing**: LoCoMo, release, packaging
