# Score Improvement Execution Report (2026-02-07)

Reference plan: `docs/SCORE_IMPROVEMENT_PLAN_20260207.md`

## Scope executed

Implemented end-to-end plan items in code, then ran matched PersonaMem slice evaluation against local Persona API.

## Implemented changes

### Retrieval budget enforcement

- Added budget guardrails in `persona/tools/runner.py`:
  - `max_total_tool_calls`
  - `max_recall_calls`
  - `max_expand_calls`
  - `max_no_novel_recalls`
- Added budget telemetry in iteration stats:
  - `blocked_tool_calls`
  - `recall_no_novel_count`
- Added blocked tool-call feedback messages to keep loop behavior deterministic under guardrails.

### Persona runtime defaults

- Set conservative defaults in `persona/services/persona_service.py`:
  - max turns = 6
  - max tool calls = 6
  - max recall calls = 2
  - max expand calls = 2
  - max non-novel recalls = 1

### Prompt/schema contract alignment

- Updated `persona/llm/prompts.py` to align with low-call policy and explicit non-novel stop behavior.
- Updated `persona/tools/schemas.py` recall stop condition and iteration protocol to avoid deep recall loops.

### Deterministic temporal interval math

- Added new `date_diff` tool schema in `persona/tools/schemas.py`.
- Implemented `date_diff_handler` in `persona/tools/memory.py`.
- Registered handler in tool registry.
- Updated prompt guidance to use `date_diff()` for interval questions.

### Context pressure reduction

- Reduced default working-memory item caps in `persona/models/memory.py`:
  - `max_episodes`: 6 -> 4
  - `max_psyche`: 6 -> 4
  - `max_active_notes`: 6 -> 4

### Temporal integrity guardrails

- Added merge safety checks in `persona/tools/integration.py`:
  - MERGE is restricted to entity memories only
  - MERGE is blocked when temporal attributes conflict (birthday/date/deadline-like keys)
- Strengthened ingestion dedup in `persona/services/ingestion_service.py`:
  - no merge across different `entity_type`
  - no merge when temporal attributes conflict
  - preserve both entities via deterministic alternate dedup key
- Added integration prompt guidance in `persona/services/integration_agent.py` to avoid merging temporal conflicts.

## Tests and validation

### Unit tests added/updated

- `tests/unit/test_tools_runner.py` (budget blocking behavior)
- `tests/unit/test_memory_tools.py` (`date_diff` handler)
- `tests/unit/test_services.py` (persona service passes guard params)
- `tests/unit/test_ingestion_dedup.py` (entity-type and temporal conflict dedup guards)
- `tests/unit/test_integration_tools.py` (merge guardrails)

### Verification commands

- `poetry run pytest tests/unit/test_tools_runner.py tests/unit/test_memory_tools.py tests/unit/test_services.py tests/unit/test_prompts.py tests/unit/test_ingestion_dedup.py tests/unit/test_integration_tools.py -q`
  - Result: 34 passed
- `poetry build`
  - Result: success
- `lsp_diagnostics` run on all modified source/test files
  - Result: zero errors

## Matched eval slice runs

Run commands:

- `uv run mem-eval run --benchmark personamem --adapter persona --samples 10 --seed 42 --workers 1 --output results/score_plan_20260207 --log-prefix score-plan-s10 --log-details`
- `uv run mem-eval run --benchmark personamem --adapter persona --samples 10 --seed 123 --workers 1 --output results/score_plan_20260207 --log-prefix score-plan-s10-seed123 --log-details`

Result files:

- `../memory-evals/results/score_plan_20260207/run_20260207_193016/summary.json`
- `../memory-evals/results/score_plan_20260207/run_20260207_193016/deep_logs.jsonl`
- `../memory-evals/results/score_plan_20260207/run_20260207_202510/summary.json`
- `../memory-evals/results/score_plan_20260207/run_20260207_202510/deep_logs.jsonl`

Observed metrics:

- Seed 42: 50.00% (5/10)
- Seed 123: 80.00% (8/10)
- Two-seed mean accuracy: 65.00%
- Average retrieval context size:
  - Seed 42: 13,583.4 tokens
  - Seed 123: 13,429.4 tokens
- Average `persona_context.tool_calls_made`:
  - Seed 42: 1.0
  - Seed 123: 1.0

Interpretation:

- Retrieval-loop over-exploration was successfully constrained in this slice (single tool call per question).
- Accuracy variance remains high across seeds (+30 points on 10-question slices), consistent with prior analysis warning about small-sample instability.
- Next gains are likely in answer selection quality and ingestion/integration fidelity, not additional tool depth.

## Notes

- A smaller 2-sample smoke run was also completed at `../memory-evals/results/score_plan_20260207/run_20260207_191902/summary.json` (2/2 correct), used only as sanity check.
- A resumed 3-sample attempt (`run_20260207_190208`) included a previously crashed item skip and is not used for primary comparison.
