# Work Plan: v0.3 Full Release

## Context

### Original Request
Full v0.3 release this week (~5-7 days) with:
- Fair benchmarks against Honcho, Mem0 (later: Supermemory, Graphiti, OpenMemory)
- Good developer experience with "aha moment"
- Production-quality packaging

### Strategic Direction (Oracle Consultation)
**Option C: "Eval-first, developer-usable" release** gated on:
1. Clean packaging + stable public API
2. Reproducible, budget-capped LoCoMo methodology with filesystem baseline

**Key Insight**: Embrace filesystem as baseline, not threat.
> "Filesystem is a strong baseline because most 'memory' failures are actually agent/prompt failures. Persona v0.3 benchmarks include a filesystem baseline under the same harness so you can see the real incremental value."

---

## Work Objectives

### Core Objective
Ship v0.3 as a credible, installable, usable, and verifiable release with fair benchmark methodology.

### Concrete Deliverables
1. Clean `pyproject.toml` without competitor deps in core
2. Public API in `persona/__init__.py` 
3. 5-minute quickstart working
4. LoCoMo benchmark with filesystem baseline
5. Fairness contract document
6. Release notes with reproducibility claims

### Definition of Done
- [x] Branch renamed to `refactor/v0.3-cognitive-memory` ✅
- [ ] Competitor deps moved to optional extras
- [ ] `from persona import PersonaService` works
- [ ] Quickstart demo runs in <5 minutes
- [ ] LoCoMo benchmark runs with budget parity
- [ ] Filesystem baseline included
- [ ] Release notes published

### Must Have
- Clean core install (no competitor deps)
- Public API surface
- At least LoCoMo with filesystem baseline
- Reproducible methodology doc

### Must NOT Have (Guardrails)
- No unfair competitor comparisons (must control LLM budget)
- No claims without reproducible evidence
- No scope creep to v0.4 items

---

## Day-by-Day Schedule

### Day 1: Packaging + Public API
**Goal**: Clean install, usable imports

### Day 2: Fairness Contract + Eval Harness
**Goal**: Defined methodology, budget caps

### Day 3: Run LoCoMo
**Goal**: Filesystem baseline + Persona results

### Day 4-5: Stabilize + Document
**Goal**: Quickstart, docs, release notes

### Day 6-7: Release Gate
**Goal**: Final verification, tag, publish

---

## TODOs

### Day 1: Packaging (CRITICAL PATH)

- [ ] 1.1 Remove competitor deps from core pyproject.toml

  **What to do**:
  - Move `mem0ai`, `graphiti-core`, `honcho-ai` to `[project.optional-dependencies]`
  - Create extras: `persona[mem0]`, `persona[graphiti]`, `persona[honcho]`, `persona[evals]`
  - Verify core install works: `pip install -e .` without competitor libs

  **Must NOT do**:
  - Don't break eval framework (evals should work with `persona[evals]`)
  - Don't remove deps that persona/ code actually uses

  **Parallelizable**: YES (with 1.2)

  **References**:
  - `pyproject.toml` - current deps at lines 26-48
  - DSPy pattern: feature-based extras
  - Mem0 pattern: `[graph]`, `[vector_stores]`, `[llms]`

  **Acceptance Criteria**:
  - [ ] `pip install -e .` succeeds without mem0ai, graphiti-core, honcho-ai
  - [ ] `pip install -e ".[evals]"` includes competitor libs for evaluation
  - [ ] `poetry install` still works
  - [ ] All unit tests pass: `poetry run pytest tests/unit -v`

  **Commit**: YES
  - Message: `refactor(deps): move competitor libs to optional extras`
  - Files: `pyproject.toml`

---

- [ ] 1.2 Add public API exports to persona/__init__.py

  **What to do**:
  - Add `__all__` with key exports
  - Export: `PersonaService`, `PersonaAdapter`, `GraphOps`, `MemoryStore`
  - Export: `UserCard`, `Memeplex`, `Memory` types
  - Add version: `__version__ = "0.3.0"`

  **Must NOT do**:
  - Don't export internal implementation details
  - Don't break existing imports

  **Parallelizable**: YES (with 1.1)

  **References**:
  - `persona/__init__.py` - current state
  - `persona/services/persona_service.py` - PersonaService class
  - `persona/adapters/persona_adapter.py` - PersonaAdapter class

  **Acceptance Criteria**:
  - [ ] `from persona import PersonaService` works
  - [ ] `from persona import PersonaAdapter` works
  - [ ] `from persona import __version__` returns "0.3.0"
  - [ ] Unit tests still pass

  **Commit**: YES
  - Message: `feat(api): add public API exports to __init__.py`
  - Files: `persona/__init__.py`

---

- [ ] 1.3 Add CHANGELOG.md

  **What to do**:
  - Create CHANGELOG.md following Keep a Changelog format
  - Document v0.3.0 changes (prompt improvements, tool schema enhancements)
  - Include all commits since last release

  **Parallelizable**: YES (with 1.1, 1.2)

  **References**:
  - Keep a Changelog format: https://keepachangelog.com/
  - Previous commits: `git log --oneline origin/main..HEAD`

  **Acceptance Criteria**:
  - [ ] CHANGELOG.md exists at repo root
  - [ ] Contains v0.3.0 section with Added/Changed/Fixed

  **Commit**: YES
  - Message: `docs: add CHANGELOG.md for v0.3.0`
  - Files: `CHANGELOG.md`

---

### Day 1-2: Fairness Contract

- [ ] 2.1 Write evaluation fairness contract

  **What to do**:
  - Document exact methodology for fair comparison
  - Specify: model, temperature, max tokens, agent loop limits
  - Specify: prompt template, retrieval window, memory write policy
  - Make this versioned and tied to release tag

  **Location**: `docs/EVAL_METHODOLOGY.md`

  **Must Include**:
  - Model: gpt-4o-mini (same for all systems)
  - Temperature: 0.0 (deterministic)
  - Max tokens: 4096
  - Agent loop: max 5 tool calls per turn
  - Retrieval: top-10 results
  - Budget cap: same LLM calls across systems

  **Acceptance Criteria**:
  - [ ] Document exists at `docs/EVAL_METHODOLOGY.md`
  - [ ] Covers all fairness variables
  - [ ] Referenced in release notes

  **Commit**: YES
  - Message: `docs: add evaluation fairness methodology`
  - Files: `docs/EVAL_METHODOLOGY.md`

---

### Day 2: Eval Harness Improvements

- [ ] 2.2 Add filesystem baseline adapter

  **What to do**:
  - Create `mem_eval/adapters/filesystem_adapter.py`
  - Simple implementation: append to files, search with embeddings
  - Same interface as other adapters
  - This is the baseline Letta used for 74%

  **References**:
  - Letta blog: "Benchmarking AI Agent Memory: Is a Filesystem All You Need?"
  - `mem_eval/adapters/base.py` - adapter interface

  **Acceptance Criteria**:
  - [ ] `filesystem_adapter.py` implements MemorySystem protocol
  - [ ] Can run LoCoMo with filesystem adapter
  - [ ] Results comparable to Letta's 74% claim

  **Commit**: YES
  - Message: `feat(evals): add filesystem baseline adapter`
  - Files: `mem_eval/adapters/filesystem_adapter.py`

---

- [ ] 2.3 Add LLM budget caps to eval harness

  **What to do**:
  - Add `max_tool_calls` parameter to runner
  - Add token counting per evaluation
  - Log LLM calls per question
  - Ensure all adapters respect same caps

  **References**:
  - `mem_eval/runner.py` - current evaluation logic
  - Oracle guidance: "hard caps (turn/tool limits)"

  **Acceptance Criteria**:
  - [ ] Each eval run logs total LLM calls
  - [ ] Budget caps enforced across adapters
  - [ ] Token usage reported in results

  **Commit**: YES
  - Message: `feat(evals): add LLM budget caps and token counting`
  - Files: `mem_eval/runner.py`, `mem_eval/metrics/`

---

### Day 3: Run LoCoMo

- [ ] 3.1 Set up LoCoMo dataset

  **What to do**:
  - Download LoCoMo dataset from Snap Research
  - Create config YAML for LoCoMo benchmark
  - Integrate with existing eval framework

  **References**:
  - LoCoMo: https://snap-research.github.io/locomo/
  - Existing configs: `mem_eval/configs/*.yaml`

  **Acceptance Criteria**:
  - [ ] LoCoMo dataset available locally
  - [ ] Config YAML created: `mem_eval/configs/locomo.yaml`
  - [ ] Can run: `mem-eval run --benchmark locomo --adapter persona`

  **Commit**: YES
  - Message: `feat(evals): add LoCoMo benchmark configuration`
  - Files: `mem_eval/configs/locomo.yaml`, data files

---

- [ ] 3.2 Run baseline comparisons

  **What to do**:
  - Run LoCoMo with filesystem baseline
  - Run LoCoMo with Persona
  - Run LoCoMo with Honcho (if time permits)
  - Run LoCoMo with Mem0 (if time permits)
  - All under same budget caps

  **Expected Results**:
  - Filesystem baseline: ~70-74% (based on Letta's claim)
  - Persona: target >=70%
  - Document any deltas

  **Acceptance Criteria**:
  - [ ] Filesystem baseline results recorded
  - [ ] Persona results recorded
  - [ ] Results include: accuracy, latency, tokens, LLM calls
  - [ ] Methodology documented

  **Commit**: NO (results only)

---

### Day 4-5: Stabilize + Document

- [ ] 4.1 Create 5-minute quickstart

  **What to do**:
  - Add Python quickstart to README
  - Create `examples/quickstart.py`
  - Verify it works in <5 minutes from fresh clone

  **Target**:
  ```python
  from persona import PersonaService, GraphOps
  
  async def main():
      graph = GraphOps()
      await graph.initialize()
      service = PersonaService(graph)
      
      # Ingest a memory
      await service.ingest("user123", "I love hiking on weekends")
      
      # Query
      result = await service.run_agent("user123", "What do I like to do?")
      print(result["content"])  # "You love hiking on weekends"
  ```

  **Acceptance Criteria**:
  - [ ] `examples/quickstart.py` runs successfully
  - [ ] README updated with quickstart section
  - [ ] Time from clone to working: <5 minutes

  **Commit**: YES
  - Message: `docs: add 5-minute quickstart example`
  - Files: `README.md`, `examples/quickstart.py`

---

- [ ] 4.2 Update README with v0.3 improvements

  **What to do**:
  - Add benchmark results section
  - Add methodology link
  - Update installation instructions
  - Add "What's New in v0.3" section

  **Acceptance Criteria**:
  - [ ] README reflects v0.3 features
  - [ ] Benchmark methodology linked
  - [ ] Installation instructions work

  **Commit**: YES
  - Message: `docs: update README for v0.3 release`
  - Files: `README.md`

---

- [ ] 4.3 Write release notes

  **What to do**:
  - Summarize v0.3 improvements
  - Include benchmark results with caveats
  - Link to methodology doc
  - Highlight: "includes filesystem baseline for transparency"

  **Key Message**:
  > "Persona v0.3 introduces a reproducible evaluation harness with strict LLM budget parity. We include a filesystem baseline so you can see the real incremental value of structured memory."

  **Acceptance Criteria**:
  - [ ] Release notes drafted
  - [ ] Benchmark results included with methodology link
  - [ ] Filesystem baseline results highlighted

  **Commit**: NO (for GitHub release)

---

### Day 6-7: Release Gate

- [ ] 5.1 Final verification

  **What to do**:
  - Run all unit tests
  - Run integration tests
  - Verify quickstart works
  - Verify clean install works
  - Review all documentation

  **Acceptance Criteria**:
  - [ ] All tests pass
  - [ ] Clean install verified
  - [ ] Quickstart verified
  - [ ] Docs reviewed

---

- [ ] 5.2 Push branch and create PR

  **What to do**:
  - Push `refactor/v0.3-cognitive-memory` to origin
  - Create PR to main
  - Request review (if applicable)

  **Acceptance Criteria**:
  - [ ] Branch pushed
  - [ ] PR created with release notes

---

- [ ] 5.3 Merge and tag release

  **What to do**:
  - Merge PR to main
  - Create annotated tag: `git tag -a v0.3.0 -m "..."`
  - Push tag: `git push origin v0.3.0`
  - Create GitHub release with notes

  **Acceptance Criteria**:
  - [ ] PR merged to main
  - [ ] v0.3.0 tag created and pushed
  - [ ] GitHub release published

---

## Success Criteria

### Verification Commands
```bash
# Clean install
pip install -e .
python -c "from persona import PersonaService; print('OK')"

# Unit tests
poetry run pytest tests/unit -v

# Quickstart
python examples/quickstart.py

# LoCoMo baseline
cd ../memory-evals
mem-eval run --benchmark locomo --adapter filesystem
mem-eval run --benchmark locomo --adapter persona
```

### Final Checklist
- [ ] Competitor deps in optional extras (not core)
- [ ] Public API works: `from persona import PersonaService`
- [ ] Quickstart runs in <5 minutes
- [ ] LoCoMo with filesystem baseline completed
- [ ] Fairness methodology documented
- [ ] Release notes with honest claims
- [ ] v0.3.0 tagged and released

---

## Risk Mitigation

### If LoCoMo integration takes too long (>Day 4)
- Ship v0.3 with current benchmarks (PersonaMem, LongMemEval, BEAM)
- Defer LoCoMo to v0.3.1 or v0.4
- Still ship packaging + DX improvements

### If Persona underperforms filesystem baseline
- Document honestly: "X% vs filesystem Y%"
- Differentiate on features: structured semantics, lifecycle, multi-user, observability
- Iterate in v0.4

### If competitor adapters can't be fairly capped
- Defer competitor comparison to v0.4
- Ship filesystem baseline comparison only
- Focus on reproducibility narrative

---

## What to Claim vs. Defer

### Claim in v0.3
- "Reproducible eval harness with strict LLM budget parity"
- "LoCoMo results with filesystem baseline"
- "Current PersonaMem/LongMemEval/BEAM under pinned configs"
- "Packaging and DX improvements"

### Defer to v0.4
- "Definitive competitor leaderboard claims"
- Aggressive tuning and optimization
- Broader benchmark suite
- Additional backend adapters

---

## Session Notes

**Branch**: `refactor/v0.3-cognitive-memory`
**Previous work**: Prompt improvements, tool schema enhancements (6 commits)
**Previous tag**: Deleted (was orphaned)
