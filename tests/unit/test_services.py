import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from persona.services.user_service import UserService
from persona.services.persona_service import PersonaService
from persona.core.graph_ops import GraphOps
from persona.models.schema import AskRequest
from persona.tools.runner import AgentResult


@pytest.fixture
def mock_graph_ops():
    mock = AsyncMock(spec=GraphOps)
    mock.__aenter__.return_value = mock
    mock.__aexit__.return_value = None
    mock.graph_db = MagicMock()
    mock.vector_store = MagicMock()
    return mock


@pytest.mark.asyncio
async def test_create_user_success(mock_graph_ops):
    mock_graph_ops.create_user = AsyncMock()
    mock_graph_ops.user_exists = AsyncMock(return_value=False)

    result = await UserService.create_user("test_user", mock_graph_ops)
    assert result["message"] == "User test_user created successfully"
    assert result["status"] == "created"


@pytest.mark.asyncio
async def test_delete_user_success(mock_graph_ops):
    mock_graph_ops.delete_user = AsyncMock()

    result = await UserService.delete_user("test_user", mock_graph_ops)
    assert result["message"] == "User test_user deleted successfully"
    mock_graph_ops.delete_user.assert_called_once_with("test_user")


@pytest.mark.asyncio
async def test_persona_service_run_agent_success(mock_graph_ops):
    service = PersonaService(mock_graph_ops)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(
        return_value=AgentResult(
            content="Agent response based on memory",
            tool_calls_made=1,
            turns=2,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
    )

    with patch.object(service, "_get_user_card", return_value=None):
        with patch.object(service, "_get_memeplex", return_value=None):
            with patch("persona.services.persona_service.AgentRunner") as MockRunner:
                MockRunner.return_value = mock_runner

                result = await service.run_agent(
                    user_id="test_user",
                    query="What do you remember about me?",
                    include_stats=True,
                )

                assert result["answer"] == "Agent response based on memory"
                assert result["stats"]["tool_calls_made"] == 1
                assert result["stats"]["turns"] == 2


@pytest.mark.asyncio
async def test_persona_service_ask_success(mock_graph_ops):
    service = PersonaService(mock_graph_ops)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(
        return_value=AgentResult(
            content="User likes coffee based on their preferences.",
            tool_calls_made=1,
            turns=2,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
    )

    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(
        return_value=MagicMock(content='{"preferences": ["coffee"]}')
    )
    mock_llm.supports_json_mode = MagicMock(return_value=True)

    with patch.object(service, "_get_user_card", return_value=None):
        with patch.object(service, "_get_memeplex", return_value=None):
            with patch("persona.services.persona_service.AgentRunner") as MockRunner:
                MockRunner.return_value = mock_runner
                with patch(
                    "persona.services.persona_service.get_chat_client",
                    return_value=mock_llm,
                ):
                    result = await service.ask(
                        user_id="test_user",
                        query="What are user preferences?",
                        output_schema={"preferences": []},
                    )

                    assert "result" in result
