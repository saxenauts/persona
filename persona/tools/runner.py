"""Agent runner with hybrid termination: model-first + configurable limits + pause/resume."""

import json
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Awaitable, Literal

from persona.llm.providers.base import (
    BaseLLMClient,
    ChatMessage,
    ChatResponse,
    ToolCall,
    ToolResult,
)
from persona.tools.schemas import MEMORY_TOOLS
from persona.tools.memory import recall, record, expand_neighbors, follow_relationship
from server.logging_config import get_logger

logger = get_logger(__name__)

AgentStatus = Literal["completed", "paused", "max_turns", "timeout", "error"]


# TODO this is done in a very basic way as a standard today. CHeck for this and update.
@dataclass
class ToolRegistry:
    handlers: Dict[str, Callable[..., Awaitable[Any]]] = field(default_factory=dict)

    def register(self, name: str, handler: Callable[..., Awaitable[Any]]) -> None:
        self.handlers[name] = handler

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        handler = self.handlers.get(tool_call.name)
        if not handler:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": f"Unknown tool: {tool_call.name}"}),
            )

        try:
            args = json.loads(tool_call.arguments)
            result = await handler(**args)
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


@dataclass
class AgentResult:
    content: str
    status: AgentStatus = "completed"
    tool_calls_made: int = 0
    turns: int = 0
    usage: Optional[Dict[str, Any]] = None
    state: Optional[str] = None

    @property
    def can_resume(self) -> bool:
        return (
            self.status in ("max_turns", "timeout", "paused") and self.state is not None
        )

    @property
    def is_complete(self) -> bool:
        return self.status == "completed"


# TODO: WHere exactly in this is the agent making new queries better for each retrieval loop. Prompt asks for natural language.
# HOw does that get converted to querying, like "three months ago, etc." and then the iteration has to mean something.
# TODO This is a common agent, so it may or may not be used in a conversation, sometimes it may be used as a one off API
# ex. weekly update on current state of content watching topics etc. That doesnt need a conversation, it needs tool calls in sequence.
# But what kind of tool calls? We need the agent to be able to smartly fetch content from the graph, based on time, vector queries,
# and other metrics that make sense, like active, related, etc. it has to be like a smart hybrid of vector and graph crawl
# with LLM intelligence applied on top for emergence.
class AgentRunner:
    def __init__(
        self,
        llm: BaseLLMClient,
        tools: Optional[List[Dict[str, Any]]] = None,
        registry: Optional[ToolRegistry] = None,
    ):
        self.llm = llm
        self.tools = tools or MEMORY_TOOLS
        self.registry = registry or ToolRegistry()

    async def run(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
        timeout: Optional[float] = None,
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
                )

            if max_turns and turns >= max_turns:
                return AgentResult(
                    content=self._get_last_content(current_messages),
                    status="max_turns",
                    tool_calls_made=total_tool_calls,
                    turns=turns,
                    usage=total_usage if total_usage else None,
                    state=self._serialize_state(current_messages),
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
                )

            assistant_msg = ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            )
            current_messages.append(assistant_msg)

            tool_results = await self._execute_tools_parallel(response.tool_calls)
            total_tool_calls += len(tool_results)

            for result in tool_results:
                current_messages.append(
                    ChatMessage(
                        role="tool",
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                    )
                )

    async def resume(
        self,
        state: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> AgentResult:
        messages = self._deserialize_state(state)
        return await self.run(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_turns=max_turns,
            timeout=timeout,
        )

    async def _execute_tools_parallel(
        self, tool_calls: List[ToolCall]
    ) -> List[ToolResult]:
        tasks = [self.registry.execute(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)

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


def create_memory_tool_registry(
    user_id: str,
    graph_ops: Any,
    store: Any,
    user_card: Optional[Any] = None,
    user_timezone: str = "UTC",
    session_id: Optional[str] = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    async def bound_recall(query: str):
        return await recall(
            user_id=user_id,
            query=query,
            graph_ops=graph_ops,
            store=store,
        )

    async def bound_record(text: str):
        return await record(
            user_id=user_id,
            text=text,
            session_id=session_id,
        )

    async def bound_expand_neighbors(
        memory_id: str,
        relationship_types: Optional[List[str]] = None,
    ):
        return await expand_neighbors(
            memory_id=memory_id,
            user_id=user_id,
            store=store,
            relationship_types=relationship_types,
        )

    async def bound_follow_relationship(
        source_id: str,
        relation_type: str,
        limit: int = 5,
    ):
        return await follow_relationship(
            source_id=source_id,
            relation_type=relation_type,
            user_id=user_id,
            store=store,
            limit=limit,
        )

    registry.register("recall", bound_recall)
    registry.register("record", bound_record)
    registry.register("expand_neighbors", bound_expand_neighbors)
    registry.register("follow_relationship", bound_follow_relationship)

    return registry
