import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from server.routers.graph_api import router, ChatRequest, ChatResponse


@pytest.fixture
def mock_graph_ops():
    mock = MagicMock()
    mock.user_exists = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def app_with_mocks(mock_graph_ops):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.graph_ops = mock_graph_ops
    return app


@pytest.fixture
def client(app_with_mocks):
    with TestClient(app_with_mocks) as client:
        yield client


class TestChatEndpoint:
    def test_chat_returns_response_structure(self, client, mock_graph_ops):
        with patch("server.routers.graph_api.PersonaService") as MockService:
            mock_instance = MockService.return_value
            mock_instance.run_agent = AsyncMock(
                return_value={
                    "answer": "Hello! How can I help you?",
                    "status": "completed",
                }
            )

            response = client.post(
                "/api/v1/users/test-user/chat",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "status" in data
            assert "session_id" in data
            assert data["response"] == "Hello! How can I help you?"
            assert data["status"] == "completed"

    def test_chat_generates_session_id_when_not_provided(self, client, mock_graph_ops):
        with patch("server.routers.graph_api.PersonaService") as MockService:
            mock_instance = MockService.return_value
            mock_instance.run_agent = AsyncMock(
                return_value={"answer": "Hi", "status": "completed"}
            )

            response = client.post(
                "/api/v1/users/test-user/chat",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

            assert response.status_code == 200
            session_id = response.json()["session_id"]
            assert session_id is not None
            assert len(session_id) > 0

    def test_chat_uses_provided_session_id(self, client, mock_graph_ops):
        with patch("server.routers.graph_api.PersonaService") as MockService:
            mock_instance = MockService.return_value
            mock_instance.run_agent = AsyncMock(
                return_value={"answer": "Hi", "status": "completed"}
            )
            provided_session_id = "my-custom-session-123"

            response = client.post(
                "/api/v1/users/test-user/chat",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "session_id": provided_session_id,
                },
            )

            assert response.status_code == 200
            assert response.json()["session_id"] == f"persona:{provided_session_id}"

    def test_chat_passes_parameters_to_service(self, client, mock_graph_ops):
        with patch("server.routers.graph_api.PersonaService") as MockService:
            mock_instance = MockService.return_value
            mock_instance.run_agent = AsyncMock(
                return_value={"answer": "Hi", "status": "completed"}
            )

            response = client.post(
                "/api/v1/users/test-user/chat",
                json={
                    "messages": [{"role": "user", "content": "What's my schedule?"}],
                    "user_timezone": "America/New_York",
                    "max_turns": 5,
                    "timeout": 30.0,
                    "include_stats": True,
                },
            )

            assert response.status_code == 200
            mock_instance.run_agent.assert_called_once()
            call_kwargs = mock_instance.run_agent.call_args.kwargs
            assert call_kwargs["user_id"] == "test-user"
            assert call_kwargs["query"] == "What's my schedule?"
            assert call_kwargs["user_timezone"] == "America/New_York"
            assert call_kwargs["max_turns"] == 5
            assert call_kwargs["timeout"] == 30.0
            assert call_kwargs["include_stats"] == True

    def test_chat_extracts_last_user_message(self, client, mock_graph_ops):
        with patch("server.routers.graph_api.PersonaService") as MockService:
            mock_instance = MockService.return_value
            mock_instance.run_agent = AsyncMock(
                return_value={"answer": "Response", "status": "completed"}
            )

            response = client.post(
                "/api/v1/users/test-user/chat",
                json={
                    "messages": [
                        {"role": "user", "content": "First message"},
                        {"role": "assistant", "content": "First response"},
                        {"role": "user", "content": "Second message"},
                    ]
                },
            )

            assert response.status_code == 200
            call_kwargs = mock_instance.run_agent.call_args.kwargs
            assert call_kwargs["query"] == "Second message"

    def test_chat_returns_404_for_nonexistent_user(self, client):
        with patch("server.routers.graph_api.PersonaService"):
            client.app.state.graph_ops.user_exists = AsyncMock(return_value=False)

            response = client.post(
                "/api/v1/users/nonexistent-user/chat",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_chat_returns_400_for_empty_messages(self, client, mock_graph_ops):
        response = client.post("/api/v1/users/test-user/chat", json={"messages": []})

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_chat_returns_400_for_no_user_message(self, client, mock_graph_ops):
        response = client.post(
            "/api/v1/users/test-user/chat",
            json={"messages": [{"role": "assistant", "content": "Just assistant"}]},
        )

        assert response.status_code == 400
        assert "user message" in response.json()["detail"].lower()

    def test_chat_includes_stats_when_requested(self, client, mock_graph_ops):
        with patch("server.routers.graph_api.PersonaService") as MockService:
            mock_instance = MockService.return_value
            mock_instance.run_agent = AsyncMock(
                return_value={
                    "answer": "Hi",
                    "status": "completed",
                    "stats": {"tool_calls_made": 2, "turns": 3},
                }
            )

            response = client.post(
                "/api/v1/users/test-user/chat",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "include_stats": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["stats"] is not None
            assert data["stats"]["tool_calls_made"] == 2

    def test_chat_includes_state_when_resumable(self, client, mock_graph_ops):
        with patch("server.routers.graph_api.PersonaService") as MockService:
            mock_instance = MockService.return_value
            mock_instance.run_agent = AsyncMock(
                return_value={
                    "answer": "Partial response",
                    "status": "max_turns_reached",
                    "state": "base64encodedstate==",
                }
            )

            response = client.post(
                "/api/v1/users/test-user/chat",
                json={"messages": [{"role": "user", "content": "Complex query"}]},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["state"] == "base64encodedstate=="
