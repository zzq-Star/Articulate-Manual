from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from articulate_core.config.settings import ArticulateConfig
from articulate_core.exceptions import LLMError
from articulate_core.llm.client import ClaudeClient


@pytest.fixture
def config():
    return ArticulateConfig(anthropic_api_key="test-key", llm_max_retries=2, llm_retry_base_delay=0.01)


@pytest.fixture
def client(config):
    return ClaudeClient(config)


@pytest.mark.asyncio
async def test_complete_success(client):
    """Successful completion returns content and usage info."""
    mock_response = AsyncMock()
    mock_response.content = [AsyncMock(text="Hello, world!")]
    mock_response.usage = AsyncMock(input_tokens=10, output_tokens=5)
    mock_response.model = "test-model"

    with patch.object(client._client.messages, "create", AsyncMock(return_value=mock_response)):
        response = await client.complete(system="Be helpful.", messages=[{"role": "user", "content": "Hi"}])
        assert response.content == "Hello, world!"
        assert response.usage["input_tokens"] == 10


@pytest.mark.asyncio
async def test_complete_retry_on_rate_limit(client):
    """RateLimitError triggers retry."""
    from anthropic import RateLimitError

    call_count = 0

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            resp = AsyncMock(status_code=429)
            resp.headers = {}
            raise RateLimitError("rate limited", response=resp, body="")
        mock_resp = AsyncMock()
        mock_resp.content = [AsyncMock(text="Success after retry")]
        mock_resp.usage = None
        mock_resp.model = "test"
        return mock_resp

    with patch.object(client._client.messages, "create", mock_create):
        response = await client.complete(system="", messages=[{"role": "user", "content": "Hi"}])
        assert response.content == "Success after retry"
        assert call_count == 2


@pytest.mark.asyncio
async def test_complete_fails_after_max_retries(client):
    """Raises LLMError after exhausting retries."""
    from anthropic import APIConnectionError

    with patch.object(
        client._client.messages, "create",
        AsyncMock(side_effect=APIConnectionError(message="connection refused", request=AsyncMock())),
    ):
        with pytest.raises(LLMError, match="Connection error after"):
            await client.complete(system="", messages=[])


@pytest.mark.asyncio
async def test_complete_structured_success(client):
    """complete_structured returns parsed Pydantic model."""

    @dataclass
    class TestModel:
        name: str
        value: int

    # We'll use pydantic BaseModel for proper schema generation
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        value: int

    mock_response = AsyncMock()
    mock_response.content = [AsyncMock(text='{"name": "test", "value": 42}')]
    mock_response.usage = None
    mock_response.model = "test"

    with patch.object(client._client.messages, "create", AsyncMock(return_value=mock_response)):
        result = await client.complete_structured(
            system="", messages=[{"role": "user", "content": "test"}],
            output_model=TestSchema,
        )
        assert result.name == "test"
        assert result.value == 42
