import pytest
import uuid

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

async def test_data_isolation(isolated_graph_ops, test_client):
    """
    Tests that data ingested by one user is not visible to another.
    1. Creates user_A and user_B.
    2. user_A ingests data about "dogs".
    3. user_B ingests data about "cats".
    4. Asserts that user_A's search for "cats" yields no results.
    5. Asserts that user_B's graph only contains "cat" nodes.
    """
    async for graph_ops, user_a_id in isolated_graph_ops:
        # Manually create a second user for this test
        user_b_id = f"test-user-{uuid.uuid4()}"
        await graph_ops.create_user(user_b_id)

        try:
            # User A ingests data about dogs
            user_a_data = {"content": "My favorite pets are dogs. I love golden retrievers."}
            response_a = test_client.post(f"/api/v1/users/{user_a_id}/ingest", json=user_a_data)
            assert response_a.status_code == 201

            # User B ingests data about cats
            user_b_data = {"content": "My favorite pets are cats. I love siamese cats."}
            response_b = test_client.post(f"/api/v1/users/{user_b_id}/ingest", json=user_b_data)
            assert response_b.status_code == 201
            
            # Give a moment for async operations and indexing to complete
            import asyncio
            await asyncio.sleep(1)

            # 1. Assert that user_A searching for "cats" gets no results
            # We need to bypass the response mock for a true similarity search test
            search_results_a = await graph_ops.text_similarity_search(query="cats", user_id=user_a_id)
            assert len(search_results_a['results']) == 0, "User A should not see User B's cat data"

            # 2. Assert that user_B's graph only contains cat-related nodes
            nodes_b = await graph_ops.get_all_nodes(user_id=user_b_id)
            node_names_b = {node.name.lower() for node in nodes_b}
            
            assert any("cat" in name for name in node_names_b)
            assert not any("dog" in name for name in node_names_b), "User B's graph should not contain dog data"

        finally:
            # Clean up the manually created user
            await graph_ops.delete_user(user_b_id)

async def test_deletion_isolation(isolated_graph_ops, test_client):
    """
    Tests that deleting one user does not affect another user's data.
    1. Creates user_A and user_B.
    2. Both users ingest data.
    3. Deletes user_A.
    4. Asserts that user_B's data is still intact.
    5. Asserts that user_A no longer exists.
    """
    async for graph_ops, user_a_id in isolated_graph_ops:
        # Manually create a second user
        user_b_id = f"test-user-{uuid.uuid4()}"
        await graph_ops.create_user(user_b_id)

        try:
            # Ingest data for both users
            client = test_client
            client.post(f"/api/v1/users/{user_a_id}/ingest", json={"content": "User A data"})
            client.post(f"/api/v1/users/{user_b_id}/ingest", json={"content": "User B data"})

            # Get User B's nodes before deletion
            nodes_b_before = await graph_ops.get_all_nodes(user_id=user_b_id)
            assert len(nodes_b_before) > 0, "User B should have data before deletion"

            # Delete User A
            delete_response = client.delete(f"/api/v1/users/{user_a_id}")
            assert delete_response.status_code == 200

            # Assert User B's data is still intact
            nodes_b_after = await graph_ops.get_all_nodes(user_id=user_b_id)
            assert len(nodes_b_after) == len(nodes_b_before), "User B's data should not be affected"
            
            # Assert User A no longer exists
            user_a_exists = await graph_ops.user_exists(user_a_id)
            assert not user_a_exists, "User A should be deleted"

        finally:
            # Clean up the manually created user
            await graph_ops.delete_user(user_b_id) 