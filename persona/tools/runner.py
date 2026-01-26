"""Agent runner with static registry, context injection, and bounded parallel execution."""

import json
import asyncio
import time
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Dict, Any, Optional, Callable, Awaitable, Literal

from persona.llm.providers.base import (
    BaseLLMClient,
    ChatMessage,
    ToolCall,
    ToolResult,
)
from persona.tools.schemas import MEMORY_TOOLS
from persona.tools.context import ToolContext
from persona.tools.memory import TOOL_HANDLERS
from server.logging_config import get_logger

logger = get_logger(__name__)

AgentStatus = Literal["completed", "paused", "max_turns", "timeout", "error"]


@dataclass
class ToolExecutionResult:
    tool_call_id: str
    name: str
    ok: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class BatchExecutionResult:
    results: List[ToolExecutionResult] = field(default_factory=list)
    total_ms: float = 0.0
    succeeded: int = 0
    failed: int = 0


class ToolRegistry:
    """
    Static tool registry with context injection at execution time.

    Tool schemas are static (never change). Context (user_id, store, etc.)
    is injected per-request via execute(tool_call, ctx).
    """

    def __init__(
        self, handlers: Optional[Dict[str, Callable[..., Awaitable[Any]]]] = None
    ):
        self.handlers = handlers or TOOL_HANDLERS

    async def execute(self, tool_call: ToolCall, ctx: ToolContext) -> ToolResult:
        handler = self.handlers.get(tool_call.name)
        if not handler:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": f"Unknown tool: {tool_call.name}"}),
            )

        try:
            args = json.loads(tool_call.arguments)
            logger.info(f"Tool call: {tool_call.name}({args})")
            result = await handler(ctx, **args)
            logger.info(f"Tool result: {tool_call.name} -> {result}")

            def serialize_value(v: Any) -> Any:
                if is_dataclass(v) and not isinstance(v, type):
                    return asdict(v)
                elif isinstance(v, list):
                    return [serialize_value(item) for item in v]
                elif isinstance(v, dict):
                    return {k: serialize_value(val) for k, val in v.items()}
                return v

            if is_dataclass(result) and not isinstance(result, type):
                content = json.dumps(serialize_value(result), default=str)
            elif isinstance(result, (dict, list)):
                content = json.dumps(serialize_value(result), default=str)
            else:
                content = str(result)

            return ToolResult(tool_call_id=tool_call.id, content=content)
        except json.JSONDecodeError as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": f"Invalid arguments JSON: {e}"}),
            )
        except Exception as e:
            logger.error(f"Tool execution error for {tool_call.name}: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": str(e)}),
            )


# Global static registry instance
REGISTRY = ToolRegistry()


async def execute_tools_bounded(
    tool_calls: List[ToolCall],
    ctx: ToolContext,
    registry: Optional[ToolRegistry] = None,
    max_concurrency: int = 8,
    timeout_s: float = 30.0,
) -> BatchExecutionResult:
    """
    Execute multiple tool calls with bounded concurrency and timeouts.

    - Partial failures are captured, not raised
    - Each tool gets its own timeout
    - Concurrency is limited via semaphore
    - Results include timing for observability
    """
    registry = registry or REGISTRY
    sem = asyncio.Semaphore(max_concurrency)
    start_time = time.time()

    async def execute_one(call: ToolCall) -> ToolExecutionResult:
        call_start = time.time()
        subtask_idx = ctx.track_subtask(call.name, "running")

        async with sem:
            try:
                result = await asyncio.wait_for(
                    registry.execute(call, ctx),
                    timeout=timeout_s,
                )
                duration_ms = (time.time() - call_start) * 1000

                try:
                    output = json.loads(result.content)
                    ok = "error" not in output
                except json.JSONDecodeError:
                    output = result.content
                    ok = True

                ctx.complete_subtask(subtask_idx, "done" if ok else "error")
                return ToolExecutionResult(
                    tool_call_id=call.id,
                    name=call.name,
                    ok=ok,
                    output=output,
                    duration_ms=duration_ms,
                )
            except asyncio.TimeoutError:
                duration_ms = (time.time() - call_start) * 1000
                ctx.complete_subtask(subtask_idx, "timeout")
                return ToolExecutionResult(
                    tool_call_id=call.id,
                    name=call.name,
                    ok=False,
                    error=f"Timeout after {timeout_s}s",
                    duration_ms=duration_ms,
                )
            except Exception as e:
                duration_ms = (time.time() - call_start) * 1000
                ctx.complete_subtask(subtask_idx, "error")
                return ToolExecutionResult(
                    tool_call_id=call.id,
                    name=call.name,
                    ok=False,
                    error=str(e),
                    duration_ms=duration_ms,
                )

    results = await asyncio.gather(*(execute_one(c) for c in tool_calls))
    total_ms = (time.time() - start_time) * 1000

    return BatchExecutionResult(
        results=list(results),
        total_ms=total_ms,
        succeeded=sum(1 for r in results if r.ok),
        failed=sum(1 for r in results if not r.ok),
    )


@dataclass
class AgentResult:
    content: str
    status: AgentStatus = "completed"
    tool_calls_made: int = 0
    turns: int = 0
    usage: Optional[Dict[str, Any]] = None
    state: Optional[str] = None
    subtask_summary: Optional[Dict[str, int]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    iteration_stats: Optional[Dict[str, Any]] = None

    @property
    def can_resume(self) -> bool:
        return (
            self.status in ("max_turns", "timeout", "paused") and self.state is not None
        )

    @property
    def is_complete(self) -> bool:
        return self.status == "completed"


class AgentRunner:
    """
    Agent runner with static registry and context injection.

    Tool schemas are passed once at construction (static).
    Tool context is passed per-run for user/session-specific data.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        tools: Optional[List[Dict[str, Any]]] = None,
        registry: Optional[ToolRegistry] = None,
    ):
        self.llm = llm
        self.tools = tools or MEMORY_TOOLS
        self.registry = registry or REGISTRY

    async def run(
        self,
        messages: List[ChatMessage],
        ctx: ToolContext,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = 10,
        timeout: Optional[float] = None,
        tool_timeout: float = 30.0,
        max_tool_concurrency: int = 8,
        auto_recall_first: bool = False,
        tool_choice: Optional[str] = None,
    ) -> AgentResult:
        total_tool_calls = 0
        total_usage: Dict[str, int] = {}
        current_messages = list(messages)
        start_time = time.time()
        turns = 0
        all_tool_results: List[Dict[str, Any]] = []
        forced_recall_done = False

        # Iteration tracking for observability
        iteration_stats: Dict[str, Any] = {
            "recall_count": 0,
            "browse_count": 0,
            "expand_count": 0,
            "unique_queries": [],
        }

        while True:
            if timeout and (time.time() - start_time) > timeout:
                return AgentResult(
                    content=self._get_last_content(current_messages),
                    status="timeout",
                    tool_calls_made=total_tool_calls,
                    turns=turns,
                    usage=total_usage if total_usage else None,
                    state=self._serialize_state(current_messages),
                    subtask_summary=ctx.subtask_summary,
                    iteration_stats=iteration_stats,
                )

            if max_turns and turns >= max_turns:
                return AgentResult(
                    content=self._get_last_content(current_messages),
                    status="max_turns",
                    tool_calls_made=total_tool_calls,
                    turns=turns,
                    usage=total_usage if total_usage else None,
                    state=self._serialize_state(current_messages),
                    subtask_summary=ctx.subtask_summary,
                    iteration_stats=iteration_stats,
                )

            try:
                # Use tool_choice on first turn only, then auto
                effective_tool_choice = (
                    tool_choice if (turns == 0 and tool_choice) else None
                )

                response = await self.llm.chat(
                    messages=current_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=self.tools,
                    tool_choice=effective_tool_choice,
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return AgentResult(
                    content=str(e),
                    status="error",
                    tool_calls_made=total_tool_calls,
                    turns=turns,
                    usage=total_usage if total_usage else None,
                    state=self._serialize_state(current_messages),
                    subtask_summary=ctx.subtask_summary,
                    iteration_stats=iteration_stats,
                )

            turns += 1

            if response.usage:
                for key, val in response.usage.items():
                    if isinstance(val, (int, float)):
                        total_usage[key] = total_usage.get(key, 0) + int(val)

            if response.stop_reason != "tool_use" or not response.tool_calls:
                # Auto-recall: if model skipped tools on first turn and we haven't forced recall yet
                if auto_recall_first and turns == 1 and not forced_recall_done:
                    forced_recall_done = True
                    user_query = self._extract_user_query(messages)
                    if user_query:
                        logger.info(f"Auto-recall triggered for: {user_query[:100]}")
                        recall_result = await self._execute_auto_recall(ctx, user_query)
                        if recall_result:
                            current_messages.append(
                                ChatMessage(
                                    role="assistant",
                                    content="Let me check my memories first.",
                                    tool_calls=[recall_result["tool_call"]],
                                )
                            )
                            current_messages.append(
                                ChatMessage(
                                    role="tool",
                                    content=recall_result["content"],
                                    tool_call_id=recall_result["tool_call"].id,
                                )
                            )
                            all_tool_results.append(recall_result["result_data"])
                            total_tool_calls += 1
                            continue

                return AgentResult(
                    content=response.content or "",
                    status="completed",
                    tool_calls_made=total_tool_calls,
                    turns=turns,
                    usage=total_usage if total_usage else None,
                    subtask_summary=ctx.subtask_summary,
                    tool_results=all_tool_results if all_tool_results else None,
                    iteration_stats=iteration_stats,
                )

            assistant_msg = ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            )
            current_messages.append(assistant_msg)

            batch_result = await execute_tools_bounded(
                response.tool_calls,
                ctx,
                self.registry,
                max_concurrency=max_tool_concurrency,
                timeout_s=tool_timeout,
            )
            total_tool_calls += len(batch_result.results)

            tool_args_by_id: Dict[str, Any] = {}
            for tc in response.tool_calls:
                try:
                    tool_args_by_id[tc.id] = json.loads(tc.arguments)
                except Exception:
                    tool_args_by_id[tc.id] = tc.arguments

            for exec_result in batch_result.results:
                if exec_result.name == "recall":
                    iteration_stats["recall_count"] += 1
                    args = tool_args_by_id.get(exec_result.tool_call_id, {})
                    if args.get("query"):
                        iteration_stats["unique_queries"].append(args["query"])
                elif exec_result.name == "browse":
                    iteration_stats["browse_count"] += 1
                elif exec_result.name in ("expand_neighbors", "follow_relationship"):
                    iteration_stats["expand_count"] += 1

                all_tool_results.append(
                    {
                        "tool": exec_result.name,
                        "ok": exec_result.ok,
                        "duration_ms": exec_result.duration_ms,
                        "args": tool_args_by_id.get(exec_result.tool_call_id),
                        "output": exec_result.output
                        if exec_result.ok
                        else {"error": exec_result.error},
                    }
                )
                content = (
                    json.dumps(exec_result.output, default=str)
                    if exec_result.ok
                    else json.dumps({"error": exec_result.error})
                )
                logger.info(f"Tool message to LLM: {content[:500]}")
                current_messages.append(
                    ChatMessage(
                        role="tool",
                        content=content,
                        tool_call_id=exec_result.tool_call_id,
                    )
                )

    async def resume(
        self,
        state: str,
        ctx: ToolContext,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> AgentResult:
        messages = self._deserialize_state(state)
        return await self.run(
            messages=messages,
            ctx=ctx,
            temperature=temperature,
            max_tokens=max_tokens,
            max_turns=max_turns,
            timeout=timeout,
        )

    def _serialize_state(self, messages: List[ChatMessage]) -> str:
        serializable = []
        for msg in messages:
            item = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                item["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            serializable.append(item)
        return json.dumps(serializable)

    def _deserialize_state(self, state: str) -> List[ChatMessage]:
        data = json.loads(state)
        messages = []
        for item in data:
            tool_calls = None
            if "tool_calls" in item:
                tool_calls = [
                    ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                    for tc in item["tool_calls"]
                ]
            messages.append(
                ChatMessage(
                    role=item["role"],
                    content=item.get("content"),
                    tool_calls=tool_calls,
                    tool_call_id=item.get("tool_call_id"),
                )
            )
        return messages

    def _get_last_content(self, messages: List[ChatMessage]) -> str:
        for msg in reversed(messages):
            if msg.role == "assistant" and msg.content:
                return msg.content
        return ""

    def _extract_user_query(self, messages: List[ChatMessage]) -> Optional[str]:
        for msg in messages:
            if msg.role == "user" and msg.content:
                return msg.content
        return None

    async def _execute_auto_recall(
        self, ctx: ToolContext, query: str
    ) -> Optional[Dict[str, Any]]:
        from uuid import uuid4

        tool_call = ToolCall(
            id=f"auto_{uuid4().hex[:8]}",
            name="recall",
            arguments=json.dumps({"query": query, "limit": 10}),
        )

        result = await self.registry.execute(tool_call, ctx)

        try:
            output = json.loads(result.content)
            ok = "error" not in output
        except json.JSONDecodeError:
            output = result.content
            ok = True

        if not ok:
            return None

        return {
            "tool_call": tool_call,
            "content": result.content,
            "result_data": {
                "tool": "recall",
                "ok": True,
                "duration_ms": 0,
                "args": {"query": query, "limit": 10},
                "output": output,
            },
        }
