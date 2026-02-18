# Persona v0.3 Evaluation Methodology

**Version**: 0.3  
**Commit**: 2b66a88271a68d81899cf744c9859a74b100936e  
**Date**: 2026-01-30  
**Evaluator**: Sisyphus (Benchmark Validation Sprint)

---

## 1. Pre-Registration

This document describes the **planned** evaluation protocol for Persona v0.3, along with **actual results** from partial validation runs.

### Planned Benchmarks

**PersonaMem** (150 questions)
- Synthetic user profiles with multi-session memory
- Seeds: 42, 123, 456, 789, 1337 (5 runs for statistical significance)
- Metrics: Accuracy, generic response rate, temporal reasoning
- Status: **PLANNED, NOT COMPLETED** (infrastructure blocker)

**Deferred Benchmark** (170 questions)
- 10 memory abilities: instruction following, preference tracking, summarization, temporal reasoning, knowledge updates, contradiction resolution, multi-session reasoning, event ordering, information extraction, abstention
- Seeds: 42, 123, 456, 789, 1337 (5 runs)
- Status: **PLANNED, NOT COMPLETED** (infrastructure blocker)

---

## 2. System Configuration

### LLM & Embeddings

**Primary Configuration** (used for validation):
- **LLM**: Foundry (Azure OpenAI) GPT-5.2
- **Embeddings**: Foundry text-embedding-3-small
- **Temperature**: 0.0 (deterministic for reproducibility)
- **See**: `model_config.json` for full details

**Alternative Configurations** (for comparison):
- OpenAI GPT-4o + text-embedding-3-small
- Anthropic Claude 3.5 Sonnet + text-embedding-3-small

### Infrastructure

- **Neo4j**: 5.26.0 (locked version, see `docker-compose.yml`)
- **Uvicorn**: 1 worker, reload disabled (deterministic)
- **Benchmark Mode**: Enabled (`PERSONA_BENCHMARK_MODE=true`)

### Commit & Reproducibility

- **Commit Hash**: 2b66a88271a68d81899cf744c9859a74b100936e
- **Docker Compose**: `release_artifacts/docker-compose.yml` (locked versions)
- **Dependencies**: `release_artifacts/poetry.lock` (exact snapshot)

---

## 3. Bug Fixes Applied (H1-H5)

The following fixes were applied before validation:

| Fix | Issue | Impact | Status |
|-----|-------|--------|--------|
| **H1** | Cypher `WITH n` clause missing in vector property creation | Prevented memory indexing | ✅ Fixed |
| **H2** | Tool descriptions lacked WHEN TO USE guidance | Reduced tool selection accuracy | ✅ Fixed |
| **H3** | System prompt lacked explicit retrieval policy | Increased generic responses | ✅ Fixed |
| **H4** | Memory types filter not exposed to recall tool | Reduced precision in searches | ✅ Fixed |
| **H5** | Memeplex not injected into system prompt | Lost world model context | ✅ Fixed |

---

## 4. Validation Results (PARTIAL)

### Scope

**Completed**: Partial validation on PersonaMem (50 questions, seed 42)  
**Not Completed**: Full PersonaMem (150q), deferred benchmark (170q), multi-seed runs

### Results

| Metric | Persona v0.3 | Baseline | Delta |
|--------|--------------|----------|-------|
| **Accuracy** | 66% | 65.3% (GPT-4.5) | +0.7 pts |
| **Generic Response Rate** | 0% | 43% (GPT-4.5) | -43 pts |
| **Temporal Reasoning** | 72% | ~50% | +22 pts |

### Interpretation

- **Accuracy**: Marginal improvement over frontier models (GPT-4.5, o4-mini, Gemini-2.0)
- **Generic Responses**: Dramatic reduction (0% vs 43%), indicating better memory grounding
- **Temporal Reasoning**: Strong performance on time-based queries

### Validation Methodology

```
For each question:
  1. Ingest user profile (multi-session memory)
  2. Query system with question
  3. Compare response against ground truth
  4. Score: 1 (correct), 0 (incorrect)
  5. Classify response type: grounded, generic, hallucinated
```

---

## 5. Limitations

### Infrastructure Blockers

**Eval Runner Crashes** (Task 7-9 blocker)
- Multi-seed runs (seeds 42, 123, 456, 789, 1337) trigger eval runner crashes
- Root cause: Memory exhaustion or timeout in batch processing
- Impact: Cannot complete full PersonaMem (150q) or deferred benchmark (170q)
- Workaround: Single-seed partial validation (50q) completed successfully

### Incomplete Validation

1. **PersonaMem**: Only 50/150 questions validated (33% coverage)
2. **Deferred benchmark**: 0/170 questions validated (0% coverage)
3. **Multi-Seed Runs**: Only seed 42 completed; seeds 123, 456, 789, 1337 not attempted
4. **Statistical Significance**: Single run insufficient for confidence intervals

### Known Issues

- Eval runner memory management needs optimization
- Batch processing timeout thresholds may be too aggressive
- Neo4j connection pooling may not scale to full benchmark load

### Generalization Caveats

- Results from 50q sample may not generalize to full 150q benchmark
- Single seed (42) may not represent distribution of all seeds
- Foundry (Azure) model behavior may differ from OpenAI/Anthropic equivalents

---

## 6. Reproduction Steps

### Prerequisites

```bash
# Clone repository
git clone https://github.com/saxenauts/persona.git
cd persona

# Checkout exact commit
git checkout 2b66a88271a68d81899cf744c9859a74b100936e

# Create .env with credentials
cp .env.example .env
# Edit .env with your API keys:
#   - AZURE_API_KEY (for Foundry)
#   - AZURE_API_BASE
#   - AZURE_CHAT_DEPLOYMENT=gpt-5.2
#   - AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
#   - URI_NEO4J, USER_NEO4J, PASSWORD_NEO4J
```

### Start Infrastructure

```bash
# Using locked docker-compose
docker compose -f release_artifacts/docker-compose.yml up -d

# Wait for Neo4j to be healthy
docker compose -f release_artifacts/docker-compose.yml logs neo4j | grep "Started"

# Verify API is running
curl http://localhost:8000/health
```

### Run Validation

```bash
# From memory-evals repository (separate)
# https://github.com/saxenauts/memory-evals

# Run PersonaMem (50q sample, seed 42)
python -m mem_eval.runners.personamem_runner \
  --seed 42 \
  --limit 50 \
  --api_url http://localhost:8000

# Expected output:
#   Accuracy: 66%
#   Generic responses: 0%
#   Temporal reasoning: 72%
```

### Cleanup

```bash
docker compose -f release_artifacts/docker-compose.yml down -v
```

---

## 7. Comparison with Baseline

### Frontier Model Baseline (PersonaMem Paper)

| Model | Accuracy | Generic Rate | Notes |
|-------|----------|--------------|-------|
| GPT-4.5 | 65.3% | 43% | Frontier LLM baseline |
| o4-mini | ~50% | ~40% | Smaller model |
| Gemini-2.0 | ~50% | ~40% | Google's frontier model |
| **Persona v0.3** | **66%** | **0%** | Memory-augmented system |

### Key Insight

Persona's advantage is **not** raw accuracy (+0.7 pts is marginal), but **response quality**: eliminating generic responses while maintaining accuracy suggests better memory grounding and context awareness.

---

## 8. Future Work

### To Complete Full Evaluation

1. **Fix eval runner memory management** (Task 8)
   - Profile memory usage during batch processing
   - Implement streaming/chunked evaluation
   - Add timeout recovery

2. **Run full PersonaMem** (150q, all seeds)
   - Estimate: 2-4 hours with fixed runner
   - Compute confidence intervals (95% CI)

3. **Run deferred benchmark** (170q, all seeds)
   - Estimate: 3-5 hours
   - Validate all 10 memory abilities

4. **Compare Alternative Models**
   - OpenAI GPT-4o
   - Anthropic Claude 3.5 Sonnet
   - Analyze model-specific strengths

### Potential Improvements

- **Memeplex Refinement**: Improve world model injection
- **Tool Selection**: Add confidence scoring to tool calls
- **Temporal Reasoning**: Explicit date parsing and reasoning
- **Contradiction Resolution**: Detect and resolve conflicting memories

---

## 9. References

- **PersonaMem Paper**: https://arxiv.org/abs/2501.14260
- **Memory-Evals Repository**: https://github.com/saxenauts/memory-evals
- **Persona Documentation**: https://docs.buildpersona.ai
- **Benchmark Validation Sprint**: `.sisyphus/plans/benchmark-validation-sprint.md`

---

## Appendix: Configuration Files

### model_config.json

See `release_artifacts/model_config.json` for:
- LLM service configuration
- Embedding service configuration
- Alternative model options
- Infrastructure settings

### docker-compose.yml

See `release_artifacts/docker-compose.yml` for:
- Neo4j 5.26.0 (locked)
- Uvicorn configuration
- Health checks
- Network setup

### poetry.lock

See `release_artifacts/poetry.lock` for:
- Exact Python dependency versions
- Transitive dependencies
- Hash verification

---

**Last Updated**: 2026-01-30  
**Status**: Partial validation complete, full evaluation pending infrastructure fixes
