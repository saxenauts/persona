# Persona Evaluation Methodology v1.0

> This document specifies the frozen evaluation parameters used for all Persona benchmarks.
> Any claims made from evaluation results MUST use these exact parameters.

---

## Model Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | gpt-4o-mini | Cost-effective, consistent |
| Temperature | 0.0 | Deterministic outputs |
| Max tokens | 4096 | Sufficient for all tasks |
| Timeout | 120s | Per-question limit |

---

## Budget Parity

Fair comparison requires equivalent resource usage across all systems.

| Resource | Cap | Enforcement |
|----------|-----|-------------|
| LLM calls per ingest | 1 | Adapter-level tracking |
| LLM calls per query | 3 | Agent loop limit |
| Tool calls per turn | 5 | Tool runner limit |
| Retrieval top-K | 10 | Consistent across systems |

**Rationale**: Some systems (e.g., Graphiti) use 40-60 LLM calls per ingest. Without budget caps, comparisons are meaningless.

---

## Statistical Rigor

| Requirement | Value |
|-------------|-------|
| Sample size | N >= 100 per benchmark |
| Random seeds | [42, 123, 456, 789, 999] |
| Confidence level | 95% |
| Effect size metric | Cohen's d |
| Significance test | Paired t-test |

### Effect Size Interpretation

| Cohen's d | Interpretation |
|-----------|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

---

## Datasets

| Benchmark | Questions | Source | Purpose |
|-----------|-----------|--------|---------|
| PersonaMem | 519 | Local | Apples-to-apples competitor comparison |
| LongMemEval | 500 | ICLR 2025 | Academic credibility |
| BEAM 100K | 100 | HuggingFace | Comprehensive task coverage |

---

## Reproducibility

All evaluations can be reproduced with:

```bash
# Clone repositories
git clone https://github.com/saxenauts/persona.git
git clone https://github.com/saxenauts/memory-evals.git

# Run evaluation
cd memory-evals
docker run persona-eval:v1.0 \
  --benchmark personamem \
  --adapter persona \
  --seeds 42,123,456,789,999

# Compare systems
mem-eval compare persona honcho --benchmark personamem
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-01-25 | Initial frozen specification |

---

## Anti-Criticism Checklist

This methodology is designed to withstand the following criticisms:

| Potential Criticism | Defense |
|---------------------|---------|
| "Small sample size" | N >= 100 with power analysis |
| "No confidence intervals" | 95% CI on all claims |
| "Not reproducible" | Docker + public code |
| "Unfair comparison" | Budget caps + this methodology doc |
| "Cherry-picked results" | All benchmarks, all seeds shown |
| "Single run variance" | 5 seeds with aggregation |
| "P-hacking" | Pre-registered methodology (this document) |

---

## How to Cite

When referencing evaluation results, include:

```
Evaluated using Persona Evaluation Methodology v1.0
https://github.com/saxenauts/persona/blob/main/docs/EVAL_METHODOLOGY.md
```
