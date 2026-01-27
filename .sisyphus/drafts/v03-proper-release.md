# Draft: v0.3 Proper Release Plan

## User's Vision
- Not just a library - it's a PRODUCT
- Beat competitors in quality AND performance
- Fair, reproducible benchmarks (replicate competitor methodology exactly)
- Good dev experience, "aha moment"
- LLM-readable source code
- Simple setup, hidden complexity, good docs
- Everything tested and documented

---

## Research Complete (5 Agents)

### 1. Release Readiness
**Overall Score: 70% MVP-ready**

| Category | Score | Status |
|----------|-------|--------|
| Packaging | 85% | Remove unused deps, add `__all__` exports |
| Documentation | 80% | Need examples directory |
| Testing | 75% | Need coverage reporting |
| Entry Points | 60% | Need CLI and SDK |
| Dependencies | 50% | 3 unused: mem0ai, graphiti-core, honcho-ai |

**Critical Actions:**
1. Remove competitor deps from core (move to `[evals]` extra)
2. Add public API exports to `persona/__init__.py`
3. Create `examples/` directory
4. Add `pytest-cov` with 70% threshold

### 2. Eval Framework
**Fairness Gaps: CRITICAL**

| Gap | Severity | Fix |
|-----|----------|-----|
| Unequal LLM budgets | CRITICAL | Cap Graphiti to same calls as Persona |
| Uncontrolled prompts | CRITICAL | Baseline prompt for all systems |
| No extraction validation | HIGH | Verify 50 memories per system |
| No cost tracking | MEDIUM | Add token counting |
| No question-level logs | MEDIUM | Save retrieval traces |

**Current Results:**
- PersonaMem: 63%
- LongMemEval: 64.1% (temporal weak: 36.7%)
- BEAM: 67.6%

**To Beat Competitors Fairly:**
- Replicate EXACT methodology (same model, prompts, data)
- Control for LLM call budget
- Report confidence intervals

### 3. Developer Experience
**Overall Score: 5.6/10**

| Dimension | Score | Issue |
|-----------|-------|-------|
| API Surface | 6/10 | No public exports, deep imports |
| Configuration | 5/10 | 6+ env vars, no local-dev mode |
| Code Readability | 8/10 | Well-organized |
| Time to First Success | 4/10 | 30+ minutes (target: 5 min) |
| Abstraction Quality | 5/10 | 4-pillar model leaks, ToolContext exposed |

**Quick Wins:**
1. Add `__all__` to `persona/__init__.py`
2. Add Python quickstart to README
3. Add startup config validation

**Medium Effort:**
1. Create `PersonaClient` wrapper hiding complexity
2. Add local-dev mode (in-memory or SQLite)
3. Create CLI for common tasks

### 4. Library Best Practices
**Industry Standards (2024-2025):**

**Packaging:**
- pyproject.toml with feature-based extras
- Example: `persona[graph]`, `persona[azure]`, `persona[evals]`

**Documentation:**
- 10 lines to value (copy-pasteable quickstart)
- Prerequisites → Install → Demo → What's Next?
- Real-world examples (not toy)

**Observability:**
- OpenTelemetry from day 1
- Token counting, latency metrics
- Trace viewer for debugging

**Release Process:**
- Semantic versioning (MAJOR.MINOR.PATCH)
- CHANGELOG.md (Keep a Changelog format)
- GitHub Actions with semantic-release
- Migration guides for breaking changes

### 5. Competitor Benchmarks
**Key Insight: LoCoMo is the standard**

| System | LoCoMo Score | Notes |
|--------|--------------|-------|
| Mem0 | 66.9% | Claims 26% higher than OpenAI |
| Letta (filesystem) | 74.0% | Simple file storage, no memory library! |
| OpenAI Memory | ~50% | Baseline |

**Letta's Challenge:**
- Achieved 74% with gpt-4o-mini + filesystem
- Beats all "specialized memory libraries"
- Questions: Are benchmarks measuring the right thing?

**Fair Comparison Requirements:**
1. Same dataset (LoCoMo, version-specific)
2. Same model (gpt-4o-mini or gpt-4o)
3. Same prompts (document exactly)
4. Same evaluation (LLM-as-judge)
5. Same metrics (accuracy, latency, tokens)
6. Open-source evaluation code
7. Report confidence intervals

---

## CRITICAL BLOCKERS (Must Fix Before Release)

### 1. Competitor Dependencies in Core
**mem0ai, graphiti-core, honcho-ai are in pyproject.toml as core deps.**
- These are NOT used in persona/ code
- They bloat install size
- They're our competitors!
- **Fix:** Move to `[evals]` optional extra

### 2. Eval Fairness is Broken
- Graphiti gets 10-20x more LLM calls
- Cannot claim "we beat Graphiti" when they have 10x reasoning budget
- **Fix:** Control for LLM call budget or document the difference

### 3. No Public API
- `from persona import PersonaService` fails
- Developers must know internal structure
- **Fix:** Add `__all__` exports

---

## Scope Decisions (User Input Needed)

### Timeline
- [ ] Target date for v0.3 release?
- [ ] Is this a public release or internal milestone?

### Priority Competitors
- [ ] Focus on beating Mem0? (most direct competitor)
- [ ] Focus on beating Letta's filesystem baseline? (proves library value)
- [ ] Focus on Graphiti? (graph-based competitor)

### MVP vs Full Release
- [ ] MVP: Packaging + fair evals + basic quickstart?
- [ ] Full: All UX improvements + CLI + SDK + comprehensive docs?

### Eval Strategy
- [ ] Run LoCoMo to compare with Mem0/Letta directly?
- [ ] Continue with PersonaMem/LongMemEval/BEAM?
- [ ] Both?

---

## Open Questions (Need Answers)

1. **What's the target timeline for v0.3 release?**

2. **Which competitors are priority for comparison?**
   - Mem0 (direct competitor, 66.9% on LoCoMo)
   - Letta (74% with filesystem - need to address this)
   - Graphiti (graph-based, less rigorous benchmarks)

3. **What's the minimum viable "aha moment" demo?**
   - HTTP API only?
   - Python SDK with 10-line example?
   - Interactive playground?

4. **Should we focus on SDK or API-first?**
   - Currently API-first (FastAPI server)
   - SDK would improve DX significantly

5. **How do we address Letta's 74% filesystem result?**
   - This challenges memory library value proposition
   - Need compelling counter-narrative

---

## Proposed Work Streams

### Stream 1: Packaging & Dependencies (1-2 days)
- Remove unused deps (mem0ai, graphiti-core, honcho-ai)
- Move to `[evals]` extra
- Add public API exports
- Add CHANGELOG.md

### Stream 2: Developer Experience (2-3 days)
- Add `examples/` with 3-5 scripts
- Add Python quickstart to README
- Create `PersonaClient` wrapper (optional)
- Add startup config validation

### Stream 3: Fair Benchmarks (3-5 days)
- Implement LoCoMo evaluation (industry standard)
- Control for LLM call budget
- Document exact methodology
- Run fair comparisons against Mem0/Letta

### Stream 4: Documentation (2-3 days)
- Quickstart guide (5 min to first success)
- API reference with examples
- Migration guide
- Architecture overview

### Stream 5: Observability (1-2 days)
- Add token counting
- Add latency metrics
- OpenTelemetry integration (optional)

---

## Summary: What We Know Now

**The Good:**
- Code quality is solid (8/10 readability)
- Architecture is sound (4-pillar model, agent-native tools)
- Current results competitive (63-67% on various benchmarks)

**The Bad:**
- Competitor deps in core (blocker)
- Eval methodology has fairness gaps (can't claim superiority)
- Developer experience is poor (5.6/10)
- 30 minutes to first success (industry standard is 5 min)

**The Ugly:**
- Letta achieved 74% with just a filesystem
- Our value proposition needs to address this

**Next Step:** User confirms priorities, then we create detailed work plan.
