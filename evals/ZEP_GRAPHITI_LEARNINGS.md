# Zep/Graphiti Benchmark Learnings

## Benchmark Results (Dec 18, 2024)

| Metric | Value |
|--------|-------|
| **Accuracy** | 55% (22/40 correct) |
| **Data Integrity** | 100% (0 errors) |
| **Total LLM Calls** | 89,524 |
| **Runtime** | ~8 hours |
| **Avg Query Latency** | 1.97s |

---

## The 45x LLM Call Multiplier

**Key Finding**: Graphiti makes ~45 LLM calls per session ingested.

For each `add_episode()` call, Graphiti runs:
1. Entity extraction (1-2 calls)
2. Entity resolution (1-3 calls)
3. Relation extraction (1-2 calls)
4. Edge deduplication (1-2 calls)
5. Community detection (1 call)
6. Summary generation (1 call)
7. Multiple embedding calls

**Impact**: 40 questions × 50 sessions × 45 calls = **90,000 LLM calls**

---

## Rate Limiting Patterns

### The "Thundering Herd"

With 75 concurrent workers hitting Azure's 750k TPM quota:

```
📊 Est. TPM: 1,440,000/750,000 (192.0%)  ← Burst exceeds quota
📊 Est. TPM: 2,000/750,000 (0.3%)         ← All 75 workers in backoff sleep
📊 Est. TPM: 614,000/750,000 (82%)        ← Recovery
```

**Why**: All workers hit rate limit simultaneously → all enter 2s exponential backoff → quota drops to 0.3% → all wake up → repeat.

### Why `time.sleep()` over `asyncio.sleep()`

We chose `time.sleep()` for the global rate limiter because:
- **Cross-Thread Safety**: `asyncio.sleep()` only works within one event loop. Each question runs in its own thread with its own loop.
- **Global Enforcement**: `time.sleep()` blocks the OS thread, ensuring the 50 RPS limit applies across ALL concurrent questions.
- **Trade-off**: This causes "stuttering" where workers pause together, but guarantees quota safety.

---

## Optimizations Applied

| Setting | Before | After |
|---------|--------|-------|
| Concurrency | 25 workers | 75 workers |
| Rate Limit | 18 RPS | 50 RPS |
| `await` bug (line 345) | ❌ Crashed | ✅ Fixed |
| Missing deps | ❌ pycurl, aiohttp | ✅ Installed |
| Eval method | 1-5 grading | Binary (LongMemEval) |

## Not Applied (Future Work)

- **Token-based limiting**: We limit requests, Azure limits tokens
- **Model swapping**: Use cheaper model for extraction phase
- **Session sampling**: Ingest only relevant sessions, not all

---

## Files Reference

- **Benchmark Code**: `evals/benchmark_runner.py`
- **Zep Adapter**: `evals/adapters/zep_adapter.py`
- **Analysis Examples**: `evals/analysis_examples.json`
- **Results**: `evals/results/zep_graphiti_binary_eval_final.jsonl`
