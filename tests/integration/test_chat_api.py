import pytest


@pytest.mark.integration
class TestChatAPIIntegration:
    @pytest.mark.asyncio
    async def test_chat_end_to_end(self, test_client, isolated_graph_ops):
        graph_ops, user_id = isolated_graph_ops

        response = test_client.post(
            f"/api/v1/users/{user_id}/chat",
            json={
                "messages": [{"role": "user", "content": "What do you know about me?"}]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["completed", "max_turns_reached"]
        assert len(data["response"]) > 0
        assert "session_id" in data

    @pytest.mark.asyncio
    async def test_chat_with_stats(self, test_client, isolated_graph_ops):
        graph_ops, user_id = isolated_graph_ops

        response = test_client.post(
            f"/api/v1/users/{user_id}/chat",
            json={
                "messages": [{"role": "user", "content": "Tell me something"}],
                "include_stats": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "stats" in data

    @pytest.mark.asyncio
    async def test_chat_with_timezone(self, test_client, isolated_graph_ops):
        graph_ops, user_id = isolated_graph_ops

        response = test_client.post(
            f"/api/v1/users/{user_id}/chat",
            json={
                "messages": [{"role": "user", "content": "What time is it?"}],
                "user_timezone": "America/Los_Angeles",
            },
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_nonexistent_user_returns_404(self, test_client):
        response = test_client.post(
            "/api/v1/users/definitely-not-a-real-user-12345/chat",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 404
