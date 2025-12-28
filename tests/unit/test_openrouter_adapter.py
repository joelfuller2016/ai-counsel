"""Tests for OpenRouter adapter with rate limit handling."""

import asyncio
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from adapters.openrouter import OpenRouterAdapter, RateLimitError


class TestOpenRouterAdapter:
    """Tests for OpenRouterAdapter."""

    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1", api_key="sk-test-123", timeout=90
        )
        assert adapter.base_url == "https://openrouter.ai/api/v1"
        assert adapter.api_key == "sk-test-123"
        assert adapter.timeout == 90

    def test_adapter_initialization_without_api_key(self):
        """Test adapter can be initialized without API key (will fail on request)."""
        adapter = OpenRouterAdapter(base_url="https://openrouter.ai/api/v1", timeout=60)
        assert adapter.base_url == "https://openrouter.ai/api/v1"
        assert adapter.api_key is None

    def test_adapter_initialization_with_rate_limit_settings(self):
        """Test adapter initializes with custom rate limit settings."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",
            rate_limit_retries=5,
            base_retry_delay=2.0,
        )
        assert adapter.rate_limit_retries == 5
        assert adapter.base_retry_delay == 2.0

    def test_adapter_initialization_default_rate_limit_settings(self):
        """Test adapter uses default rate limit settings when not specified."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1", api_key="sk-test"
        )
        assert (
            adapter.rate_limit_retries == OpenRouterAdapter.DEFAULT_RATE_LIMIT_RETRIES
        )
        assert adapter.base_retry_delay == OpenRouterAdapter.DEFAULT_BASE_RETRY_DELAY

    def test_build_request_structure(self):
        """Test build_request returns correct OpenAI-compatible structure with auth."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1", api_key="sk-test-key-123"
        )

        endpoint, headers, body = adapter.build_request(
            model="anthropic/claude-3.5-sonnet", prompt="What is 2+2?"
        )

        assert endpoint == "/chat/completions"
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer sk-test-key-123"
        assert body["model"] == "anthropic/claude-3.5-sonnet"
        assert body["messages"] == [{"role": "user", "content": "What is 2+2?"}]
        assert body["stream"] is False

    def test_build_request_without_api_key_still_includes_header(self):
        """Test build_request includes Authorization header even if api_key is None."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1", api_key=None
        )

        endpoint, headers, body = adapter.build_request(
            model="test-model", prompt="test"
        )

        # Should include header with "Bearer None" - will fail at API level
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer None"

    def test_build_request_with_long_prompt(self):
        """Test build_request handles long prompts."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1", api_key="sk-test"
        )

        long_prompt = "A" * 5000
        endpoint, headers, body = adapter.build_request(
            model="test-model", prompt=long_prompt
        )

        assert body["messages"][0]["content"] == long_prompt

    def test_parse_response_extracts_content(self):
        """Test parse_response extracts message content from OpenAI format."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1", api_key="sk-test"
        )

        response_json = {
            "id": "gen-123",
            "model": "anthropic/claude-3.5-sonnet",
            "created": 1234567890,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "The answer is 4."},
                    "finish_reason": "stop",
                }
            ],
        }

        result = adapter.parse_response(response_json)
        assert result == "The answer is 4."

    def test_parse_response_handles_missing_choices(self):
        """Test parse_response raises error if choices missing."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1", api_key="sk-test"
        )

        response_json = {"id": "gen-123", "model": "test-model"}

        with pytest.raises(KeyError) as exc_info:
            adapter.parse_response(response_json)

        assert "choices" in str(exc_info.value).lower()

    def test_parse_response_handles_empty_choices(self):
        """Test parse_response raises error if choices is empty."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1", api_key="sk-test"
        )

        response_json = {"choices": []}

        with pytest.raises(IndexError):
            adapter.parse_response(response_json)

    def test_parse_response_handles_missing_message(self):
        """Test parse_response raises error if message missing."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1", api_key="sk-test"
        )

        response_json = {"choices": [{"index": 0, "finish_reason": "stop"}]}

        with pytest.raises(KeyError) as exc_info:
            adapter.parse_response(response_json)

        assert "message" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        """Test successful invocation with mocked HTTP client."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response from OpenRouter"}}]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            adapter = OpenRouterAdapter(
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-test-key",
                timeout=60,
            )
            result = await adapter.invoke(
                prompt="Say hello", model="anthropic/claude-3.5-sonnet"
            )

            assert result == "Test response from OpenRouter"
            mock_client.post.assert_called_once()

            # Verify the request was built correctly
            call_args = mock_client.post.call_args
            assert "/chat/completions" in call_args[0][0]
            assert call_args[1]["headers"]["Authorization"] == "Bearer sk-test-key"
            assert call_args[1]["json"]["messages"][0]["content"] == "Say hello"

    @pytest.mark.asyncio
    async def test_invoke_with_context(self):
        """Test invocation with context prepends context to prompt."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response with context"}}]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            adapter = OpenRouterAdapter(
                base_url="https://openrouter.ai/api/v1", api_key="sk-test"
            )
            await adapter.invoke(
                prompt="Current question",
                model="test-model",
                context="Previous context",
            )

            # Verify context was prepended
            call_args = mock_client.post.call_args
            message_content = call_args[1]["json"]["messages"][0]["content"]
            assert "Previous context" in message_content
            assert "Current question" in message_content

    def test_environment_variable_in_api_key(self):
        """Test that environment variables can be used for API keys."""
        # This test verifies the pattern - actual env var substitution
        # happens in HTTPAdapterConfig validator
        test_key = "sk-or-test-env-123"
        os.environ["TEST_OPENROUTER_KEY"] = test_key

        # Simulating what would happen after config loading
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1",
            api_key=test_key,  # In real usage, this would be resolved from ${TEST_OPENROUTER_KEY}
            timeout=60,
        )

        assert adapter.api_key == test_key

        # Cleanup
        del os.environ["TEST_OPENROUTER_KEY"]


class TestRateLimitHandling:
    """Tests for rate limit handling with exponential backoff."""

    def test_parse_retry_after_numeric(self):
        """Test parsing numeric Retry-After header."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1", api_key="sk-test"
        )

        mock_response = Mock()
        mock_response.headers = {"Retry-After": "60"}

        result = adapter._parse_retry_after(mock_response)
        assert result == 60.0

    def test_parse_retry_after_missing(self):
        """Test parsing when Retry-After header is missing."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1", api_key="sk-test"
        )

        mock_response = Mock()
        mock_response.headers = {}

        result = adapter._parse_retry_after(mock_response)
        assert result is None

    def test_parse_retry_after_invalid(self):
        """Test parsing invalid Retry-After value returns None."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1", api_key="sk-test"
        )

        mock_response = Mock()
        mock_response.headers = {"Retry-After": "invalid"}

        result = adapter._parse_retry_after(mock_response)
        assert result is None

    def test_calculate_backoff_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",
            base_retry_delay=1.0,
        )

        # Attempt 0: base_delay * 2^0 = 1.0 + jitter
        delay_0 = adapter._calculate_backoff_delay(0)
        assert 1.0 <= delay_0 <= 2.0  # 1.0 + 0-1 jitter

        # Attempt 1: base_delay * 2^1 = 2.0 + jitter
        delay_1 = adapter._calculate_backoff_delay(1)
        assert 2.0 <= delay_1 <= 3.0

        # Attempt 2: base_delay * 2^2 = 4.0 + jitter
        delay_2 = adapter._calculate_backoff_delay(2)
        assert 4.0 <= delay_2 <= 5.0

    def test_calculate_backoff_delay_with_retry_after(self):
        """Test backoff respects Retry-After header."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",
            base_retry_delay=1.0,
        )

        # With Retry-After, should use that value + small jitter
        delay = adapter._calculate_backoff_delay(0, retry_after=30.0)
        assert 30.0 <= delay <= 31.0  # 30 + 0-1 jitter

    def test_calculate_backoff_delay_capped_at_max(self):
        """Test backoff delay is capped at MAX_RETRY_DELAY."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",
            base_retry_delay=1.0,
        )

        # Attempt 10: 2^10 = 1024, should be capped
        delay = adapter._calculate_backoff_delay(10)
        assert delay <= OpenRouterAdapter.MAX_RETRY_DELAY + 1.0  # Max + jitter

    @pytest.mark.asyncio
    async def test_rate_limit_retry_success(self):
        """Test rate limit retry succeeds after initial 429."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",
            rate_limit_retries=3,
            base_retry_delay=0.01,  # Fast retries for test
        )

        # First call returns 429, second succeeds
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}
        mock_response_429.json.return_value = {"error": "Rate limit exceeded"}

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "choices": [{"message": {"content": "Success after retry"}}]
        }
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[mock_response_429, mock_response_success]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):  # Don't actually sleep
                result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Success after retry"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_retries_exhausted(self):
        """Test RateLimitError raised when retries exhausted."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",
            rate_limit_retries=2,
            base_retry_delay=0.01,
        )

        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {}
        mock_response_429.json.return_value = {"error": "Rate limit exceeded"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_429)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RateLimitError) as exc_info:
                    await adapter.invoke(prompt="test", model="test-model")

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.message
        # Should have tried 3 times (initial + 2 retries)
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_non_429_errors_not_retried(self):
        """Test that non-429 HTTP errors are not retried."""
        adapter = OpenRouterAdapter(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",
            rate_limit_retries=3,
        )

        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        mock_response_400.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "Bad Request",
                request=Mock(url="https://test.com"),
                response=mock_response_400,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_400)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                await adapter.invoke(prompt="test", model="test-model")

        # Should only be called once (no retries for 400)
        assert mock_client.post.call_count == 1


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_rate_limit_error_attributes(self):
        """Test RateLimitError stores all attributes."""
        error = RateLimitError(
            message="Rate limit exceeded",
            retry_after=60.0,
            status_code=429,
        )

        assert error.message == "Rate limit exceeded"
        assert error.retry_after == 60.0
        assert error.status_code == 429
        assert str(error) == "Rate limit exceeded"

    def test_rate_limit_error_defaults(self):
        """Test RateLimitError default values."""
        error = RateLimitError(message="Test error")

        assert error.retry_after is None
        assert error.status_code == 429


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi"}}]}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            adapter = OpenRouterAdapter(
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-test",
            )
            result = await adapter.health_check(model="test-model")

        assert result.available is True
        assert result.model == "test-model"
        assert result.adapter == "openrouter"
        assert result.latency_ms is not None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_health_check_rate_limited(self):
        """Test health check handles rate limit gracefully."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "30"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            adapter = OpenRouterAdapter(
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-test",
            )
            result = await adapter.health_check(model="test-model")

        assert result.available is False
        assert result.model == "test-model"
        assert "Rate limited" in result.error
        assert "30" in result.error

    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        """Test health check handles timeout."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            adapter = OpenRouterAdapter(
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-test",
            )
            result = await adapter.health_check(model="test-model", timeout=5.0)

        assert result.available is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_health_check_http_error(self):
        """Test health check handles HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Invalid API key"}
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "Unauthorized",
                request=Mock(url="https://test.com"),
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            adapter = OpenRouterAdapter(
                base_url="https://openrouter.ai/api/v1",
                api_key="invalid",
            )
            result = await adapter.health_check(model="test-model")

        assert result.available is False
        assert "401" in result.error
        assert result.latency_ms is not None
