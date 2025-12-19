import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
from persona.llm.providers.azure_openai_client import AzureOpenAIClient, AzureEndpointState
from persona.llm.providers.base import ChatMessage

@pytest.mark.asyncio
async def test_azure_endpoint_state_parsing():
    state = AzureEndpointState("https://test.url")
    
    # Test seconds parsing
    state.update_from_headers({"x-ratelimit-remaining-tokens": "1000", "x-ratelimit-reset-tokens": "5s"})
    assert state.remaining_tokens == 1000
    assert state.reset_at > time.time() + 4
    
    # Test ms parsing
    state.update_from_headers({"x-ratelimit-reset-tokens": "500ms"})
    assert state.reset_at <= time.time() + 0.6

@pytest.mark.asyncio
async def test_azure_client_load_balancing():
    client = AzureOpenAIClient(
        api_key="test-key",
        api_base="https://endpoint1, https://endpoint2"
    )
    
    assert len(client.endpoints) == 2
    
    # Round robin check
    e1 = client._get_next_endpoint()
    e2 = client._get_next_endpoint()
    e3 = client._get_next_endpoint()
    
    assert e1 != e2
    assert e1 == e3

@pytest.mark.asyncio
async def test_azure_client_429_handling():
    client = AzureOpenAIClient(
        api_key="test-key",
        api_base="https://endpoint1, https://endpoint2"
    )
    
    # Mark endpoint 1 as 429
    client.endpoint_states["https://endpoint1"].mark_429(retry_after=10)
    
    # Next endpoint should be endpoint 2 even if it's endpoint 1's turn
    for _ in range(5):
        assert client._get_next_endpoint() == "https://endpoint2"

@pytest.mark.asyncio
async def test_azure_client_pacing_trigger():
    client = AzureOpenAIClient(
        api_key="test-key",
        api_base="https://endpoint1"
    )
    client.endpoint_states["https://endpoint1"].remaining_tokens = 100 # Low tokens
    
    start_time = time.time()
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await client._pace("https://endpoint1")
        mock_sleep.assert_called_once()
