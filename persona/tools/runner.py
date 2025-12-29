"""Agent runner with static registry, context injection, and bounded parallel execution."""

import json
import asyncio
import time
from dataclasses import dataclass, field
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
            result = await handler(ctx, **args)

            if hasattr(result, "__dict__"):
                content = json.dumps(result.__dict__, default=str)
            elif isinstance(result, (dict, list)):
                content = json.dumps(result, default=str)
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
        max_turns: Optional[int] = None,
        timeout: Optional[float] = None,
        tool_timeout: float = 30.0,
        max_tool_concurrency: int = 8,
    ) -> AgentResult:
        total_tool_calls = 0
        total_usage: Dict[str, int] = {}
        current_messages = list(messages)
        start_time = time.time()
        turns = 0

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
                )

            try:
                response = await self.llm.chat(
                    messages=current_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=self.tools,
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
                )

            turns += 1

            if response.usage:
                for key, val in response.usage.items():
                    if isinstance(val, (int, float)):
                        total_usage[key] = total_usage.get(key, 0) + int(val)

            if response.stop_reason != "tool_use" or not response.tool_calls:
                return AgentResult(
                    content=response.content or "",
                    status="completed",
                    tool_calls_made=total_tool_calls,
                    turns=turns,
                    usage=total_usage if total_usage else None,
                    subtask_summary=ctx.subtask_summary,
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

            for exec_result in batch_result.results:
                content = (
                    json.dumps(exec_result.output, default=str)
                    if exec_result.ok
                    else json.dumps({"error": exec_result.error})
                )
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
