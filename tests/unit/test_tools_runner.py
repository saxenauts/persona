"""Tests for AgentRunner tool execution loop."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from persona.llm.providers.base import ChatMessage, ChatResponse, ToolCall, ToolResult
from persona.tools.runner import AgentRunner, ToolRegistry, AgentResult
from persona.tools.context import ToolContext


def make_mock_ctx() -> ToolContext:
    """Create a mock ToolContext for testing."""
    mock_graph_ops = MagicMock()
    mock_store = MagicMock()
    return ToolContext(
        user_id="test_user",
        graph_ops=mock_graph_ops,
        store=mock_store,
        user_timezone="UTC",
    )


class TestToolRegistry:
    @pytest.mark.asyncio
    async def test_register_and_execute_tool(self):
        async def mock_tool(ctx: ToolContext, query: str, limit: int = 10):
            return {"results": [query], "count": limit}

        registry = ToolRegistry(handlers={"search": mock_tool})
        ctx = make_mock_ctx()

        tool_call = ToolCall(
            id="call_123",
            name="search",
            arguments=json.dumps({"query": "test", "limit": 5}),
        )

        result = await registry.execute(tool_call, ctx)

        assert result.tool_call_id == "call_123"
        parsed = json.loads(result.content)
        assert parsed["results"] == ["test"]
        assert parsed["count"] == 5

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self):
        registry = ToolRegistry(handlers={})
        ctx = make_mock_ctx()

        tool_call = ToolCall(
            id="call_456",
            name="nonexistent",
            arguments="{}",
        )

        result = await registry.execute(tool_call, ctx)

        assert result.tool_call_id == "call_456"
        parsed = json.loads(result.content)
        assert "error" in parsed
        assert "Unknown tool" in parsed["error"]

    @pytest.mark.asyncio
    async def test_execute_with_invalid_json_arguments(self):
        async def mock_tool(ctx: ToolContext):
            return "ok"

        registry = ToolRegistry(handlers={"simple": mock_tool})
        ctx = make_mock_ctx()

        tool_call = ToolCall(
            id="call_789",
            name="simple",
            arguments="not valid json",
        )

        result = await registry.execute(tool_call, ctx)

        parsed = json.loads(result.content)
        assert "error" in parsed
        assert "Invalid arguments JSON" in parsed["error"]

    @pytest.mark.asyncio
    async def test_execute_with_exception_returns_error(self):
        async def failing_tool(ctx: ToolContext):
            raise ValueError("Something went wrong")

        registry = ToolRegistry(handlers={"failing": failing_tool})
        ctx = make_mock_ctx()

        tool_call = ToolCall(
            id="call_fail",
            name="failing",
            arguments="{}",
        )

        result = await registry.execute(tool_call, ctx)

        parsed = json.loads(result.content)
        assert "error" in parsed
        assert "Something went wrong" in parsed["error"]


class TestAgentRunner:
    @pytest.mark.asyncio
    async def test_simple_response_no_tools(self):
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = ChatResponse(
            content="Hello, world!",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            stop_reason="end_turn",
        )

        runner = AgentRunner(llm=mock_llm)
        messages = [ChatMessage(role="user", content="Hi")]
        ctx = make_mock_ctx()

        result = await runner.run(messages, ctx=ctx)

        assert result.content == "Hello, world!"
        assert result.status == "completed"
        assert result.tool_calls_made == 0
        assert result.turns == 1

    @pytest.mark.asyncio
    async def test_single_tool_call_loop(self):
        mock_llm = AsyncMock()

        first_response = ChatResponse(
            content=None,
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="memory_query",
                    arguments=json.dumps({"query": "test"}),
                )
            ],
            stop_reason="tool_use",
        )

        second_response = ChatResponse(
            content="Based on your memory, here is the answer.",
            model="test-model",
            usage={"prompt_tokens": 20, "completion_tokens": 15},
            stop_reason="end_turn",
        )

        mock_llm.chat.side_effect = [first_response, second_response]

        async def mock_memory_query(ctx: ToolContext, query: str):
            return {"hits": [{"content": "test memory"}]}

        registry = ToolRegistry(handlers={"memory_query": mock_memory_query})

        runner = AgentRunner(llm=mock_llm, registry=registry)
        messages = [ChatMessage(role="user", content="What do you remember?")]
        ctx = make_mock_ctx()

        result = await runner.run(messages, ctx=ctx)

        assert result.content == "Based on your memory, here is the answer."
        assert result.status == "completed"
        assert result.tool_calls_made == 1
        assert result.turns == 2
        assert mock_llm.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        mock_llm = AsyncMock()

        tool_response = ChatResponse(
            content="Still working...",
            model="test-model",
            tool_calls=[ToolCall(id="call_loop", name="looping", arguments="{}")],
            stop_reason="tool_use",
        )

        mock_llm.chat.return_value = tool_response

        async def looping_tool(ctx: ToolContext):
            return "loop"

        registry = ToolRegistry(handlers={"looping": looping_tool})

        runner = AgentRunner(llm=mock_llm, registry=registry)
        messages = [ChatMessage(role="user", content="Loop forever")]
        ctx = make_mock_ctx()

        result = await runner.run(messages, ctx=ctx, max_turns=5)

        assert result.status == "max_turns"
        assert result.turns == 5
        assert result.can_resume is True
        assert result.state is not None

    @pytest.mark.asyncio
    async def test_no_limit_runs_until_model_stops(self):
        mock_llm = AsyncMock()

        tool_responses = [
            ChatResponse(
                content=None,
                model="test-model",
                tool_calls=[ToolCall(id=f"call_{i}", name="work", arguments="{}")],
                stop_reason="tool_use",
            )
            for i in range(20)
        ]
        final_response = ChatResponse(
            content="Finally done after 20 tool calls",
            model="test-model",
            stop_reason="end_turn",
        )
        mock_llm.chat.side_effect = tool_responses + [final_response]

        async def work_tool(ctx: ToolContext):
            return "done"

        registry = ToolRegistry(handlers={"work": work_tool})

        runner = AgentRunner(llm=mock_llm, registry=registry)
        messages = [ChatMessage(role="user", content="Do lots of work")]
        ctx = make_mock_ctx()

        result = await runner.run(messages, ctx=ctx)

        assert result.status == "completed"
        assert result.turns == 21
        assert result.tool_calls_made == 20

    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self):
        mock_llm = AsyncMock()

        first_response = ChatResponse(
            content=None,
            model="test-model",
            tool_calls=[
                ToolCall(id="call_a", name="tool_a", arguments="{}"),
                ToolCall(id="call_b", name="tool_b", arguments="{}"),
                ToolCall(id="call_c", name="tool_c", arguments="{}"),
            ],
            stop_reason="tool_use",
        )

        second_response = ChatResponse(
            content="Got all three results",
            model="test-model",
            stop_reason="end_turn",
        )

        mock_llm.chat.side_effect = [first_response, second_response]

        call_order = []

        async def tool_a(ctx: ToolContext):
            call_order.append("a")
            return "result_a"

        async def tool_b(ctx: ToolContext):
            call_order.append("b")
            return "result_b"

        async def tool_c(ctx: ToolContext):
            call_order.append("c")
            return "result_c"

        registry = ToolRegistry(
            handlers={"tool_a": tool_a, "tool_b": tool_b, "tool_c": tool_c}
        )

        runner = AgentRunner(llm=mock_llm, registry=registry)
        messages = [ChatMessage(role="user", content="Call all tools")]
        ctx = make_mock_ctx()

        result = await runner.run(messages, ctx=ctx)

        assert result.tool_calls_made == 3
        assert len(call_order) == 3

    @pytest.mark.asyncio
    async def test_usage_accumulation(self):
        mock_llm = AsyncMock()

        responses = [
            ChatResponse(
                content=None,
                model="test-model",
                usage={"prompt_tokens": 100, "completion_tokens": 50},
                tool_calls=[ToolCall(id="c1", name="t", arguments="{}")],
                stop_reason="tool_use",
            ),
            ChatResponse(
                content=None,
                model="test-model",
                usage={"prompt_tokens": 150, "completion_tokens": 75},
                tool_calls=[ToolCall(id="c2", name="t", arguments="{}")],
                stop_reason="tool_use",
            ),
            ChatResponse(
                content="Done",
                model="test-model",
                usage={"prompt_tokens": 200, "completion_tokens": 100},
                stop_reason="end_turn",
            ),
        ]

        mock_llm.chat.side_effect = responses

        async def dummy_tool(ctx: ToolContext):
            return "ok"

        registry = ToolRegistry(handlers={"t": dummy_tool})

        runner = AgentRunner(llm=mock_llm, registry=registry)
        messages = [ChatMessage(role="user", content="test")]
        ctx = make_mock_ctx()

        result = await runner.run(messages, ctx=ctx)

        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 450
        assert result.usage["completion_tokens"] == 225

    @pytest.mark.asyncio
    async def test_resume_from_state(self):
        mock_llm = AsyncMock()

        tool_response = ChatResponse(
            content="Working...",
            model="test-model",
            tool_calls=[ToolCall(id="call_1", name="work", arguments="{}")],
            stop_reason="tool_use",
        )

        mock_llm.chat.return_value = tool_response

        async def work_tool(ctx: ToolContext):
            return "done"

        registry = ToolRegistry(handlers={"work": work_tool})

        runner = AgentRunner(llm=mock_llm, registry=registry)
        messages = [ChatMessage(role="user", content="Start work")]
        ctx = make_mock_ctx()

        result1 = await runner.run(messages, ctx=ctx, max_turns=2)

        assert result1.status == "max_turns"
        assert result1.can_resume is True

        final_response = ChatResponse(
            content="All done!",
            model="test-model",
            stop_reason="end_turn",
        )
        mock_llm.chat.return_value = final_response

        assert result1.state is not None
        result2 = await runner.resume(result1.state, ctx=ctx, max_turns=5)

        assert result2.status == "completed"
        assert result2.content == "All done!"

    @pytest.mark.asyncio
    async def test_agent_result_properties(self):
        completed = AgentResult(content="done", status="completed")
        assert completed.is_complete is True
        assert completed.can_resume is False

        max_turns = AgentResult(content="partial", status="max_turns", state="{}")
        assert max_turns.is_complete is False
        assert max_turns.can_resume is True

        timeout = AgentResult(content="partial", status="timeout", state="{}")
        assert timeout.is_complete is False
        assert timeout.can_resume is True

        error = AgentResult(content="error", status="error")
        assert error.is_complete is False
        assert error.can_resume is False
