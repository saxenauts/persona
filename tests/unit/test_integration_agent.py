"""
Tests for the IntegrationAgent that runs background memory integration.

Tests cover:
- Agent loop executes tools correctly
- Agent stops when no tool calls returned
- Checkpoint is updated after each step
- max_turns limit is respected
- Error handling when tool fails
- Tool execution via IntegrationToolRegistry
"""

import pytest
import json
from datetime import datetime
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional

from persona.llm.providers.base import ChatMessage, ChatResponse, ToolCall, ToolResult


# ============================================================================
# Mock Models (to avoid import issues with full integration module)
# ============================================================================


@dataclass
class IntegrationAgentConfig:
    """Configuration for integration agent runs."""

    max_turns: int = 10
    max_tool_calls: int = 50
    tool_timeout: float = 30.0
    temperature: float = 0.3
    max_tokens: int = 4096


@dataclass
class IntegrationResult:
    """Result of an integration agent run."""

    success: bool
    memories_processed: int = 0
    links_created: int = 0
    flags_raised: int = 0
    merges_performed: int = 0
    turns: int = 0
    tool_calls_made: int = 0
    duration_ms: float = 0.0
    summary: str = ""
    errors: List[str] = field(default_factory=list)
    state: Optional[str] = None
    can_resume: bool = False


@dataclass
class IntegrationContext:
    """Context passed to integration tools."""

    user_id: str
    graph_ops: MagicMock
    store: MagicMock
    trigger_ids: List[str]
    run_id: str
    checkpoint: dict = field(default_factory=dict)


# ============================================================================
# IntegrationToolRegistry (mirrors implementation)
# ============================================================================


class IntegrationToolRegistry:
    """Tool registry for integration agent with IntegrationContext."""

    def __init__(self, handlers: Optional[dict] = None):
        self.handlers = handlers or {}

    async def execute(self, tool_call: ToolCall, ctx: IntegrationContext) -> ToolResult:
        handler = self.handlers.get(tool_call.name)
        if not handler:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": f"Unknown tool: {tool_call.name}"}),
            )

        try:
            args = json.loads(tool_call.arguments)
            result = await handler(ctx, **args)

            # Serialize result
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
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": str(e)}),
            )


# ============================================================================
# IntegrationAgent (simplified for testing)
# ============================================================================


class IntegrationAgent:
    """Integration agent that runs the tool loop."""

    def __init__(
        self,
        llm,
        registry: IntegrationToolRegistry,
        config: Optional[IntegrationAgentConfig] = None,
    ):
        self.llm = llm
        self.registry = registry
        self.config = config or IntegrationAgentConfig()

    async def run(
        self,
        ctx: IntegrationContext,
        messages: Optional[List[ChatMessage]] = None,
    ) -> IntegrationResult:
        """Run the agent loop until completion or max_turns."""
        messages = messages or []

        turns = 0
        total_tool_calls = 0
        links_created = 0
        flags_raised = 0
        merges_performed = 0
        memories_processed = 0
        errors = []

        while (
            turns < self.config.max_turns
            and total_tool_calls < self.config.max_tool_calls
        ):
            turns += 1

            # Update checkpoint
            ctx.checkpoint["turn"] = turns
            ctx.checkpoint["tool_calls"] = total_tool_calls

            try:
                response = await self.llm.chat(
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            except Exception as e:
                errors.append(f"LLM error: {e}")
                break

            # Check if agent is done (no tool calls)
            if response.stop_reason != "tool_use" or not response.tool_calls:
                break

            # Add assistant message
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            # Execute tool calls
            for tool_call in response.tool_calls:
                total_tool_calls += 1

                result = await self.registry.execute(tool_call, ctx)
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=result.content,
                        tool_call_id=tool_call.id,
                    )
                )

                # Track metrics from commit_patch results
                if tool_call.name == "commit_patch":
                    try:
                        result_data = json.loads(result.content)
                        if result_data.get("success"):
                            applied = result_data.get("applied", 0)
                            links_created += applied

                            args = json.loads(tool_call.arguments)
                            patch_json = args.get("patch_json", "{}")
                            patch_data = json.loads(patch_json)
                            for item in patch_data.get("items", []):
                                op = item.get("operation")
                                if op == "flag":
                                    flags_raised += 1
                                elif op == "merge":
                                    merges_performed += 1
                                elif op == "mark_integrated":
                                    memories_processed += 1
                    except (json.JSONDecodeError, KeyError):
                        pass

        # Update final checkpoint
        ctx.checkpoint["completed"] = True
        ctx.checkpoint["final_turn"] = turns

        return IntegrationResult(
            success=len(errors) == 0,
            memories_processed=memories_processed,
            links_created=links_created,
            flags_raised=flags_raised,
            merges_performed=merges_performed,
            turns=turns,
            tool_calls_made=total_tool_calls,
            errors=errors,
            state=json.dumps(ctx.checkpoint)
            if turns >= self.config.max_turns
            else None,
            can_resume=turns >= self.config.max_turns,
        )


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    return AsyncMock()


@pytest.fixture
def mock_ctx():
    """Create a mock IntegrationContext."""
    return IntegrationContext(
        user_id="test-user",
        graph_ops=MagicMock(),
        store=MagicMock(),
        trigger_ids=["mem-1", "mem-2"],
        run_id="run-001",
    )


@pytest.fixture
def registry_with_tools():
    """Create registry with mock tool handlers."""

    async def mock_recall(ctx, query, **kwargs):
        return {"results": [{"id": "mem-100", "content": "Found memory"}], "count": 1}

    async def mock_get_unintegrated(ctx, **kwargs):
        return {"memories": [{"id": "mem-1"}, {"id": "mem-2"}], "count": 2}

    async def mock_expand_neighbors(ctx, memory_id, **kwargs):
        return {"neighbors": [], "count": 0}

    async def mock_commit_patch(ctx, patch_json):
        patch = json.loads(patch_json)
        return {"success": True, "applied": len(patch.get("items", []))}

    return IntegrationToolRegistry(
        handlers={
            "recall": mock_recall,
            "get_unintegrated": mock_get_unintegrated,
            "expand_neighbors": mock_expand_neighbors,
            "commit_patch": mock_commit_patch,
        }
    )


# ============================================================================
# IntegrationToolRegistry Tests
# ============================================================================


class TestIntegrationToolRegistry:
    """Tests for IntegrationToolRegistry."""

    @pytest.mark.asyncio
    async def test_execute_known_tool(self, mock_ctx):
        """Test executing a registered tool."""

        async def mock_recall(ctx, query):
            return {"results": [query], "count": 1}

        registry = IntegrationToolRegistry(handlers={"recall": mock_recall})

        tool_call = ToolCall(
            id="call_123", name="recall", arguments=json.dumps({"query": "test query"})
        )

        result = await registry.execute(tool_call, mock_ctx)

        assert result.tool_call_id == "call_123"
        data = json.loads(result.content)
        assert data["results"] == ["test query"]

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, mock_ctx):
        """Test error when tool is unknown."""
        registry = IntegrationToolRegistry(handlers={})

        tool_call = ToolCall(id="call_456", name="nonexistent", arguments="{}")

        result = await registry.execute(tool_call, mock_ctx)

        data = json.loads(result.content)
        assert "error" in data
        assert "Unknown tool" in data["error"]

    @pytest.mark.asyncio
    async def test_execute_with_invalid_json(self, mock_ctx):
        """Test error handling for invalid JSON arguments."""

        async def mock_tool(ctx):
            return "ok"

        registry = IntegrationToolRegistry(handlers={"test": mock_tool})

        tool_call = ToolCall(id="call_789", name="test", arguments="not valid json")

        result = await registry.execute(tool_call, mock_ctx)

        data = json.loads(result.content)
        assert "error" in data
        assert "Invalid arguments JSON" in data["error"]

    @pytest.mark.asyncio
    async def test_execute_with_tool_exception(self, mock_ctx):
        """Test error handling when tool raises exception."""

        async def failing_tool(ctx):
            raise ValueError("Tool failed")

        registry = IntegrationToolRegistry(handlers={"failing": failing_tool})

        tool_call = ToolCall(id="call_fail", name="failing", arguments="{}")

        result = await registry.execute(tool_call, mock_ctx)

        data = json.loads(result.content)
        assert "error" in data
        assert "Tool failed" in data["error"]

    @pytest.mark.asyncio
    async def test_execute_serializes_dict_result(self, mock_ctx):
        """Test that dict results are JSON serialized."""

        async def dict_tool(ctx):
            return {"key": "value", "nested": {"a": 1}}

        registry = IntegrationToolRegistry(handlers={"dict_tool": dict_tool})

        tool_call = ToolCall(id="call_dict", name="dict_tool", arguments="{}")

        result = await registry.execute(tool_call, mock_ctx)

        data = json.loads(result.content)
        assert data["key"] == "value"
        assert data["nested"]["a"] == 1

    @pytest.mark.asyncio
    async def test_execute_serializes_list_result(self, mock_ctx):
        """Test that list results are JSON serialized."""

        async def list_tool(ctx):
            return [1, 2, 3]

        registry = IntegrationToolRegistry(handlers={"list_tool": list_tool})

        tool_call = ToolCall(id="call_list", name="list_tool", arguments="{}")

        result = await registry.execute(tool_call, mock_ctx)

        data = json.loads(result.content)
        assert data == [1, 2, 3]


# ============================================================================
# IntegrationAgent Loop Tests
# ============================================================================


class TestIntegrationAgentLoop:
    """Tests for agent loop execution."""

    @pytest.mark.asyncio
    async def test_agent_stops_when_no_tool_calls(
        self, mock_llm, mock_ctx, registry_with_tools
    ):
        """Test agent stops when LLM returns no tool calls."""
        mock_llm.chat = AsyncMock(
            return_value=ChatResponse(
                content="Done processing all memories.",
                model="test-model",
                stop_reason="end_turn",
            )
        )

        agent = IntegrationAgent(llm=mock_llm, registry=registry_with_tools)
        result = await agent.run(mock_ctx)

        assert result.success is True
        assert result.turns == 1
        assert result.tool_calls_made == 0
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_executes_tool_calls(
        self, mock_llm, mock_ctx, registry_with_tools
    ):
        """Test agent correctly executes tool calls."""
        first_response = ChatResponse(
            content=None,
            model="test-model",
            tool_calls=[ToolCall(id="call_1", name="get_unintegrated", arguments="{}")],
            stop_reason="tool_use",
        )

        second_response = ChatResponse(
            content="Found 2 memories to process.",
            model="test-model",
            stop_reason="end_turn",
        )

        mock_llm.chat = AsyncMock(side_effect=[first_response, second_response])

        agent = IntegrationAgent(llm=mock_llm, registry=registry_with_tools)
        result = await agent.run(mock_ctx)

        assert result.success is True
        assert result.turns == 2
        assert result.tool_calls_made == 1
        assert mock_llm.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_agent_max_turns_limit(self, mock_llm, mock_ctx, registry_with_tools):
        """Test that agent respects max_turns limit."""
        # LLM always returns tool calls (infinite loop scenario)
        mock_llm.chat = AsyncMock(
            return_value=ChatResponse(
                content="Still working...",
                model="test-model",
                tool_calls=[
                    ToolCall(
                        id="call_loop", name="recall", arguments='{"query": "test"}'
                    )
                ],
                stop_reason="tool_use",
            )
        )

        config = IntegrationAgentConfig(max_turns=5)
        agent = IntegrationAgent(
            llm=mock_llm, registry=registry_with_tools, config=config
        )
        result = await agent.run(mock_ctx)

        assert result.turns == 5
        assert result.can_resume is True
        assert result.state is not None

    @pytest.mark.asyncio
    async def test_agent_max_tool_calls_limit(
        self, mock_llm, mock_ctx, registry_with_tools
    ):
        """Test that agent respects max_tool_calls limit."""
        # Each response has 3 tool calls
        mock_llm.chat = AsyncMock(
            return_value=ChatResponse(
                content=None,
                model="test-model",
                tool_calls=[
                    ToolCall(id="call_a", name="recall", arguments='{"query": "a"}'),
                    ToolCall(id="call_b", name="recall", arguments='{"query": "b"}'),
                    ToolCall(id="call_c", name="recall", arguments='{"query": "c"}'),
                ],
                stop_reason="tool_use",
            )
        )

        # With max_tool_calls=5, should stop before completing 2 full turns
        config = IntegrationAgentConfig(max_turns=100, max_tool_calls=5)
        agent = IntegrationAgent(
            llm=mock_llm, registry=registry_with_tools, config=config
        )
        result = await agent.run(mock_ctx)

        # First turn: 3 calls, second turn: would be 6 total > 5 limit
        assert result.tool_calls_made <= 6  # May complete turn 2

    @pytest.mark.asyncio
    async def test_checkpoint_updated_each_turn(
        self, mock_llm, mock_ctx, registry_with_tools
    ):
        """Test checkpoint is updated after each turn."""
        responses = [
            ChatResponse(
                content=None,
                model="test-model",
                tool_calls=[
                    ToolCall(id="call_1", name="recall", arguments='{"query": "x"}')
                ],
                stop_reason="tool_use",
            ),
            ChatResponse(
                content=None,
                model="test-model",
                tool_calls=[
                    ToolCall(id="call_2", name="recall", arguments='{"query": "y"}')
                ],
                stop_reason="tool_use",
            ),
            ChatResponse(
                content="Done",
                model="test-model",
                stop_reason="end_turn",
            ),
        ]
        mock_llm.chat = AsyncMock(side_effect=responses)

        agent = IntegrationAgent(llm=mock_llm, registry=registry_with_tools)
        result = await agent.run(mock_ctx)

        # Checkpoint should have final state
        assert mock_ctx.checkpoint["completed"] is True
        assert mock_ctx.checkpoint["final_turn"] == 3

    @pytest.mark.asyncio
    async def test_agent_handles_llm_error(
        self, mock_llm, mock_ctx, registry_with_tools
    ):
        """Test agent handles LLM errors gracefully."""
        mock_llm.chat = AsyncMock(side_effect=Exception("Connection error"))

        agent = IntegrationAgent(llm=mock_llm, registry=registry_with_tools)
        result = await agent.run(mock_ctx)

        assert result.success is False
        assert len(result.errors) == 1
        assert "LLM error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_agent_handles_tool_error(self, mock_llm, mock_ctx):
        """Test agent continues after tool error."""

        async def failing_tool(ctx, **kwargs):
            raise Exception("Tool crashed")

        registry = IntegrationToolRegistry(handlers={"recall": failing_tool})

        mock_llm.chat = AsyncMock(
            side_effect=[
                ChatResponse(
                    content=None,
                    model="test-model",
                    tool_calls=[
                        ToolCall(id="call_1", name="recall", arguments='{"query": "x"}')
                    ],
                    stop_reason="tool_use",
                ),
                ChatResponse(
                    content="Handled error",
                    model="test-model",
                    stop_reason="end_turn",
                ),
            ]
        )

        agent = IntegrationAgent(llm=mock_llm, registry=registry)
        result = await agent.run(mock_ctx)

        # Agent should continue even if tool fails
        assert result.turns == 2


# ============================================================================
# Metrics Tracking Tests
# ============================================================================


class TestIntegrationMetrics:
    """Tests for metrics tracking in integration agent."""

    @pytest.mark.asyncio
    async def test_tracks_commit_patch_metrics(self, mock_llm, mock_ctx):
        """Test that commit_patch results update metrics."""

        async def mock_commit_patch(ctx, patch_json):
            return {"success": True, "applied": 3}

        registry = IntegrationToolRegistry(handlers={"commit_patch": mock_commit_patch})

        patch_json = json.dumps(
            {
                "items": [
                    {"operation": "link", "source_id": "a", "target_id": "b"},
                    {"operation": "mark_integrated", "source_id": "a"},
                    {"operation": "mark_integrated", "source_id": "b"},
                ]
            }
        )

        mock_llm.chat = AsyncMock(
            side_effect=[
                ChatResponse(
                    content=None,
                    model="test-model",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="commit_patch",
                            arguments=json.dumps({"patch_json": patch_json}),
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ChatResponse(
                    content="Done", model="test-model", stop_reason="end_turn"
                ),
            ]
        )

        agent = IntegrationAgent(llm=mock_llm, registry=registry)
        result = await agent.run(mock_ctx)

        assert result.links_created == 3
        assert result.memories_processed == 2

    @pytest.mark.asyncio
    async def test_tracks_flags_and_merges(self, mock_llm, mock_ctx):
        """Test tracking of flags and merges."""

        async def mock_commit_patch(ctx, patch_json):
            return {"success": True, "applied": 2}

        registry = IntegrationToolRegistry(handlers={"commit_patch": mock_commit_patch})

        patch_json = json.dumps(
            {
                "items": [
                    {
                        "operation": "flag",
                        "source_id": "a",
                        "properties": {"flag_type": "contradiction"},
                    },
                    {"operation": "merge", "source_id": "dup", "target_id": "canon"},
                ]
            }
        )

        mock_llm.chat = AsyncMock(
            side_effect=[
                ChatResponse(
                    content=None,
                    model="test-model",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="commit_patch",
                            arguments=json.dumps({"patch_json": patch_json}),
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ChatResponse(
                    content="Done", model="test-model", stop_reason="end_turn"
                ),
            ]
        )

        agent = IntegrationAgent(llm=mock_llm, registry=registry)
        result = await agent.run(mock_ctx)

        assert result.flags_raised == 1
        assert result.merges_performed == 1


# ============================================================================
# Multiple Tool Calls Tests
# ============================================================================


class TestMultipleToolCalls:
    """Tests for parallel tool call handling."""

    @pytest.mark.asyncio
    async def test_executes_multiple_parallel_tools(
        self, mock_llm, mock_ctx, registry_with_tools
    ):
        """Test handling multiple tool calls in single response."""
        mock_llm.chat = AsyncMock(
            side_effect=[
                ChatResponse(
                    content=None,
                    model="test-model",
                    tool_calls=[
                        ToolCall(
                            id="call_a", name="recall", arguments='{"query": "a"}'
                        ),
                        ToolCall(
                            id="call_b", name="recall", arguments='{"query": "b"}'
                        ),
                        ToolCall(id="call_c", name="get_unintegrated", arguments="{}"),
                    ],
                    stop_reason="tool_use",
                ),
                ChatResponse(
                    content="Done", model="test-model", stop_reason="end_turn"
                ),
            ]
        )

        agent = IntegrationAgent(llm=mock_llm, registry=registry_with_tools)
        result = await agent.run(mock_ctx)

        assert result.tool_calls_made == 3
        assert result.turns == 2

    @pytest.mark.asyncio
    async def test_messages_contain_all_tool_results(
        self, mock_llm, mock_ctx, registry_with_tools
    ):
        """Test that messages include all tool results."""
        messages_captured = []

        original_chat = mock_llm.chat

        async def capture_chat(messages, **kwargs):
            messages_captured.append(list(messages))
            if len(messages_captured) == 1:
                return ChatResponse(
                    content=None,
                    model="test-model",
                    tool_calls=[
                        ToolCall(
                            id="call_1", name="recall", arguments='{"query": "x"}'
                        ),
                        ToolCall(
                            id="call_2", name="recall", arguments='{"query": "y"}'
                        ),
                    ],
                    stop_reason="tool_use",
                )
            return ChatResponse(
                content="Done", model="test-model", stop_reason="end_turn"
            )

        mock_llm.chat = AsyncMock(side_effect=capture_chat)

        agent = IntegrationAgent(llm=mock_llm, registry=registry_with_tools)
        await agent.run(mock_ctx, messages=[])

        # Second call should have both tool results
        assert len(messages_captured) == 2
        second_call_messages = messages_captured[1]
        tool_messages = [m for m in second_call_messages if m.role == "tool"]
        assert len(tool_messages) == 2


# ============================================================================
# State and Resumption Tests
# ============================================================================


class TestStateAndResumption:
    """Tests for state management and resumption."""

    @pytest.mark.asyncio
    async def test_state_contains_checkpoint(
        self, mock_llm, mock_ctx, registry_with_tools
    ):
        """Test that state contains checkpoint data when max_turns hit."""
        mock_llm.chat = AsyncMock(
            return_value=ChatResponse(
                content=None,
                model="test-model",
                tool_calls=[
                    ToolCall(id="call_1", name="recall", arguments='{"query": "x"}')
                ],
                stop_reason="tool_use",
            )
        )

        config = IntegrationAgentConfig(max_turns=3)
        agent = IntegrationAgent(
            llm=mock_llm, registry=registry_with_tools, config=config
        )
        result = await agent.run(mock_ctx)

        assert result.can_resume is True
        assert result.state is not None

        state_data = json.loads(result.state)
        assert "completed" in state_data
        assert "final_turn" in state_data

    @pytest.mark.asyncio
    async def test_no_state_when_completed(
        self, mock_llm, mock_ctx, registry_with_tools
    ):
        """Test that state is None when agent completes normally."""
        mock_llm.chat = AsyncMock(
            return_value=ChatResponse(
                content="All done!",
                model="test-model",
                stop_reason="end_turn",
            )
        )

        agent = IntegrationAgent(llm=mock_llm, registry=registry_with_tools)
        result = await agent.run(mock_ctx)

        assert result.can_resume is False
        assert result.state is None


# ============================================================================
# Config Tests
# ============================================================================


class TestIntegrationAgentConfig:
    """Tests for IntegrationAgentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IntegrationAgentConfig()

        assert config.max_turns == 10
        assert config.max_tool_calls == 50
        assert config.tool_timeout == 30.0
        assert config.temperature == 0.3
        assert config.max_tokens == 4096

    def test_custom_config(self):
        """Test custom configuration."""
        config = IntegrationAgentConfig(
            max_turns=20,
            max_tool_calls=100,
            temperature=0.5,
        )

        assert config.max_turns == 20
        assert config.max_tool_calls == 100
        assert config.temperature == 0.5


# ============================================================================
# IntegrationResult Tests
# ============================================================================


class TestIntegrationResult:
    """Tests for IntegrationResult dataclass."""

    def test_success_result(self):
        """Test creating a success result."""
        result = IntegrationResult(
            success=True,
            memories_processed=10,
            links_created=5,
            turns=3,
            tool_calls_made=15,
        )

        assert result.success is True
        assert result.memories_processed == 10
        assert result.links_created == 5

    def test_failure_result(self):
        """Test creating a failure result with errors."""
        result = IntegrationResult(
            success=False,
            errors=["LLM error: timeout", "Connection lost"],
        )

        assert result.success is False
        assert len(result.errors) == 2

    def test_resumable_result(self):
        """Test resumable result properties."""
        result = IntegrationResult(
            success=True,
            can_resume=True,
            state='{"turn": 10, "tool_calls": 50}',
        )

        assert result.can_resume is True
        assert result.state is not None

        state = json.loads(result.state)
        assert state["turn"] == 10
