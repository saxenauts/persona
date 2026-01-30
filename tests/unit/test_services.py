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
async def test_persona_service_run_agent_includes_working_memory(mock_graph_ops):
    """Test that run_agent calls Retriever and includes working_memory_chars in stats."""
    service = PersonaService(mock_graph_ops)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(
        return_value=AgentResult(
            content="Agent response with working memory context",
            tool_calls_made=2,
            turns=1,
            usage={"prompt_tokens": 150, "completion_tokens": 75},
        )
    )

    mock_retriever = AsyncMock()
    mock_retriever.get_working_memory = AsyncMock(
        return_value=(
            "Recent episode: User went for a run yesterday",
            {
                "episode_count": 1,
                "psyche_count": 0,
                "note_count": 0,
                "link_count": 0,
                "working_memory_chars": 45,
                "config": {"episode_window_days": 7, "psyche_window_days": 30},
            },
        )
    )

    with patch.object(service, "_get_user_card", return_value=None):
        with patch.object(service, "_get_memeplex", return_value=None):
            with patch("persona.services.persona_service.Retriever") as MockRetriever:
                MockRetriever.return_value = mock_retriever
                with patch(
                    "persona.services.persona_service.AgentRunner"
                ) as MockRunner:
                    MockRunner.return_value = mock_runner

                    result = await service.run_agent(
                        user_id="test_user",
                        query="What did I do yesterday?",
                        include_stats=True,
                    )

                    assert (
                        result["answer"] == "Agent response with working memory context"
                    )
                    assert "stats" in result
                    stats = result["stats"]

                    assert "working_memory_chars" in stats
                    assert stats["working_memory_chars"] == 45

                    assert "retriever" in stats
                    assert stats["retriever"]["episode_count"] == 1

                    mock_retriever.get_working_memory.assert_called_once()


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

    mock_retriever = AsyncMock()
    mock_retriever.get_working_memory = AsyncMock(
        return_value=(
            "",
            {
                "episode_count": 0,
                "psyche_count": 0,
                "note_count": 0,
                "link_count": 0,
                "working_memory_chars": 0,
                "config": {},
            },
        )
    )

    with patch.object(service, "_get_user_card", return_value=None):
        with patch.object(service, "_get_memeplex", return_value=None):
            with patch("persona.services.persona_service.Retriever") as MockRetriever:
                MockRetriever.return_value = mock_retriever
                with patch(
                    "persona.services.persona_service.AgentRunner"
                ) as MockRunner:
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
