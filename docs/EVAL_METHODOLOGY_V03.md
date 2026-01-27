# Persona v0.3 Evaluation Methodology

**Version**: v0.3  
**Date**: January 26, 2026

## Purpose

This document defines the exact methodology for evaluating Persona's memory capabilities. Results are only valid if produced following this methodology.

---

## Benchmarks

### PersonaMem (Primary)

PersonaMem tests personal memory recall using multiple-choice questions. The benchmark simulates a user sharing information across conversations, then asking questions about that information.

**Question Types** (5 categories):
1. **Single-session**: Facts shared within one conversation
2. **Multi-session**: Facts requiring cross-conversation recall
3. **Temporal**: Time-based questions ("When did X happen?")
4. **Preference**: User preferences and opinions
5. **Evolution**: Facts that change over time

**Baseline**: Random chance = 25% (4-choice MCQ)

### LongMemEval (Secondary)

LongMemEval tests memory over extended conversation sequences with more complex retrieval requirements.

### BEAM (Planned - Not Yet Run)

BEAM (Beyond a Million Tokens) tests memory at extreme scale with 100K+ token conversations. Currently not included in v0.3 evaluation.

---

## Official v0.3 Configuration

### PersonaMem Stratified Run

```bash
cd memory-evals

# Official configuration
uv run mem-eval run \
  --benchmark personamem \
  --adapter persona \
  --samples 50 \
  --seeds 42,123,456

# This produces:
# - 50 samples per seed
# - 3 seeds = 150 total questions
# - Stratified across all question types
```

**Requirements**:
- Persona server running (`docker compose up -d`)
- Clean database state (no pre-existing memories)
- Same LLM for all runs (gpt-4 or gpt-4o)

### LongMemEval Run

```bash
uv run mem-eval run \
  --benchmark longmemeval \
  --adapter persona \
  --samples 20 \
  --seeds 42
```

---

## Environment Requirements

### Server Configuration

```env
# Required in persona/.env
LLM_SERVICE=foundry/gpt-4o  # or openai/gpt-4o
EMBEDDING_SERVICE=foundry/text-embedding-3-small

URI_NEO4J=bolt://neo4j:7687
USER_NEO4J=neo4j
PASSWORD_NEO4J=<secure_password>
```

### Eval Framework

```bash
cd memory-evals

# Install dependencies
uv sync

# Verify adapter connection
curl http://localhost:8000/health
```

---

## Result Validity Criteria

Results are **valid** only if:

1. **Reproducible**: Same seeds produce same question order
2. **Complete**: All samples completed without errors
3. **Fair**: Same LLM and configuration across comparisons
4. **Documented**: Run ID, timestamp, and configuration recorded

Results are **invalid** if:

- Run terminated early with errors
- Database had pre-existing memories
- Different LLM versions used in comparison
- Configuration deviates from this document

---

## Reporting Format

### Required Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Correct / Total (percentage) |
| Samples | Total questions evaluated |
| Seeds | List of random seeds used |
| LLM | Model identifier |
| Run ID | Unique evaluation run identifier |

### Optional Metrics

| Metric | Description |
|--------|-------------|
| Per-type breakdown | Accuracy by question type |
| Confidence interval | 95% CI for accuracy |
| Cohen's h | Effect size vs baseline |
| p-value | Statistical significance |

---

## Baseline Comparisons

### Random Baseline

For 4-choice MCQ: **25% expected accuracy**

### Honcho Comparison (If Applicable)

When comparing to Honcho:
1. Use identical question sets (same seeds)
2. Use identical LLM configuration
3. Run sequentially to avoid resource contention
4. Document both adapters' configurations

---

## Known Limitations

1. **Synthetic data**: PersonaMem uses generated conversations, not real user data
2. **MCQ format**: Multiple-choice may not reflect real-world free-form queries
3. **LLM variance**: Results may vary with LLM provider/version
4. **Sample size**: 150 samples provides ~8% margin of error

---

## Changelog

- **2026-01-26**: Initial v0.3 methodology document
- Removed invalid BEAM references (benchmark not yet run)
- Established stratified sampling as official configuration
