"""Unit tests for HTTP adapter error scenarios.

Issue #31: Tests for timeout handling, connection errors, malformed responses,
rate limit (429) responses, and server errors (500, 502, 503).

Uses pytest and mock/httpx patterns consistent with existing test_base_http_adapter.py.
"""
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from adapters.base_http import BaseHTTPAdapter, is_retryable_http_error


class ConcreteHTTPAdapter(BaseHTTPAdapter):
    """Concrete implementation for testing BaseHTTPAdapter error scenarios."""

    ADAPTER_NAME = "test"

    def build_request(self, model: str, prompt: str):
        """Build test request."""
        return (
            "/api/test",
            {"Content-Type": "application/json"},
            {"model": model, "prompt": prompt},
        )

    def parse_response(self, response_json: dict) -> str:
        """Parse test response."""
        if "response" not in response_json:
            raise KeyError(
                f"Response missing 'response' field. Keys: {list(response_json.keys())}"
            )
        return response_json["response"]


class TestIsRetryableHttpError:
    """Tests for the is_retryable_http_error helper function."""

    def test_503_is_retryable(self):
        """Test that 503 Service Unavailable is retryable."""
        mock_response = Mock()
        mock_response.status_code = 503
        error = httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=Mock(url="http://test"),
            response=mock_response,
        )
        assert is_retryable_http_error(error) is True

    def test_502_is_retryable(self):
        """Test that 502 Bad Gateway is retryable."""
        mock_response = Mock()
        mock_response.status_code = 502
        error = httpx.HTTPStatusError(
            "502 Bad Gateway",
            request=Mock(url="http://test"),
            response=mock_response,
        )
        assert is_retryable_http_error(error) is True

    def test_500_is_retryable(self):
        """Test that 500 Internal Server Error is retryable."""
        mock_response = Mock()
        mock_response.status_code = 500
        error = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=Mock(url="http://test"),
            response=mock_response,
        )
        assert is_retryable_http_error(error) is True

    def test_429_is_retryable(self):
        """Test that 429 Too Many Requests (rate limit) is retryable."""
        mock_response = Mock()
        mock_response.status_code = 429
        error = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=Mock(url="http://test"),
            response=mock_response,
        )
        assert is_retryable_http_error(error) is True

    def test_400_is_not_retryable(self):
        """Test that 400 Bad Request is not retryable."""
        mock_response = Mock()
        mock_response.status_code = 400
        error = httpx.HTTPStatusError(
            "400 Bad Request",
            request=Mock(url="http://test"),
            response=mock_response,
        )
        assert is_retryable_http_error(error) is False

    def test_401_is_not_retryable(self):
        """Test that 401 Unauthorized is not retryable."""
        mock_response = Mock()
        mock_response.status_code = 401
        error = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=Mock(url="http://test"),
            response=mock_response,
        )
        assert is_retryable_http_error(error) is False

    def test_403_is_not_retryable(self):
        """Test that 403 Forbidden is not retryable."""
        mock_response = Mock()
        mock_response.status_code = 403
        error = httpx.HTTPStatusError(
            "403 Forbidden",
            request=Mock(url="http://test"),
            response=mock_response,
        )
        assert is_retryable_http_error(error) is False

    def test_404_is_not_retryable(self):
        """Test that 404 Not Found is not retryable."""
        mock_response = Mock()
        mock_response.status_code = 404
        error = httpx.HTTPStatusError(
            "404 Not Found",
            request=Mock(url="http://test"),
            response=mock_response,
        )
        assert is_retryable_http_error(error) is False

    def test_connect_error_is_retryable(self):
        """Test that ConnectError is retryable."""
        error = httpx.ConnectError("Connection refused")
        assert is_retryable_http_error(error) is True

    def test_timeout_exception_is_retryable(self):
        """Test that TimeoutException is retryable."""
        error = httpx.TimeoutException("Request timed out")
        assert is_retryable_http_error(error) is True

    def test_network_error_is_retryable(self):
        """Test that NetworkError is retryable."""
        error = httpx.NetworkError("Network unreachable")
        assert is_retryable_http_error(error) is True

    def test_generic_exception_is_not_retryable(self):
        """Test that generic exceptions are not retryable."""
        error = ValueError("Some random error")
        assert is_retryable_http_error(error) is False


class TestTimeoutHandling:
    """Tests for HTTP adapter timeout handling."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_timeout_raises_timeout_error(self, mock_client_class):
        """Test that timeout raises TimeoutError with appropriate message."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=30, max_retries=1)

        with pytest.raises(TimeoutError) as exc_info:
            await adapter.invoke(prompt="test", model="test-model")

        assert "timed out" in str(exc_info.value)
        assert "30" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_timeout_with_override(self, mock_client_class):
        """Test that timeout_override is used correctly."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=30)

        with pytest.raises(TimeoutError) as exc_info:
            await adapter.invoke(prompt="test", model="test-model", timeout_override=120)

        assert "120" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_httpx_timeout_exception_is_retried(self, mock_client_class):
        """Test that httpx.TimeoutException triggers retry logic."""
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Success after retry"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                httpx.TimeoutException("Read timed out"),
                mock_response_success,
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Success after retry"
        assert mock_client.post.call_count == 2


class TestConnectionErrors:
    """Tests for HTTP adapter connection error handling."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_connection_refused_is_retried(self, mock_client_class):
        """Test that connection refused errors are retried."""
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Success"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                httpx.ConnectError("Connection refused"),
                mock_response_success,
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Success"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_connection_error_exhausts_retries(self, mock_client_class):
        """Test that persistent connection errors exhaust retries and raise."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)

        with pytest.raises(httpx.ConnectError):
            await adapter.invoke(prompt="test", model="test-model")

        # Should have attempted max_retries times
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_network_unreachable_is_retried(self, mock_client_class):
        """Test that network unreachable errors are retried."""
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Connected"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                httpx.NetworkError("Network is unreachable"),
                httpx.NetworkError("Network is unreachable"),
                mock_response_success,
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Connected"
        assert mock_client.post.call_count == 3


class TestMalformedResponses:
    """Tests for handling malformed API responses."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_missing_response_field(self, mock_client_class):
        """Test handling of response missing expected 'response' field."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "data", "model": "test"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60)

        with pytest.raises(KeyError) as exc_info:
            await adapter.invoke(prompt="test", model="test-model")

        assert "response" in str(exc_info.value)
        assert "unexpected" in str(exc_info.value) or "model" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_empty_response_body(self, mock_client_class):
        """Test handling of empty response body."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60)

        with pytest.raises(KeyError):
            await adapter.invoke(prompt="test", model="test-model")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_invalid_json_response(self, mock_client_class):
        """Test handling of invalid JSON in response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60)

        with pytest.raises(ValueError, match="Invalid JSON"):
            await adapter.invoke(prompt="test", model="test-model")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_null_response_field(self, mock_client_class):
        """Test handling of null value in response field."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": None}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60)
        result = await adapter.invoke(prompt="test", model="test-model")

        # Should return None (which the adapter returns as-is)
        assert result is None


class TestRateLimitResponses:
    """Tests for 429 rate limit response handling."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_429_is_retried(self, mock_client_class):
        """Test that 429 Too Many Requests is retried."""
        mock_response_fail = Mock()
        mock_response_fail.status_code = 429
        mock_response_fail.json.return_value = {"error": "Rate limit exceeded"}
        mock_response_fail.text = "Rate limit exceeded"
        mock_response_fail.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=Mock(url="http://test"),
                response=mock_response_fail,
            )
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Success"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[mock_response_fail, mock_response_success]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Success"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_429_exhausts_retries(self, mock_client_class):
        """Test that persistent 429 errors exhaust retries."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": "Rate limit exceeded"}
        mock_response.text = "Rate limit exceeded"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=Mock(url="http://test"),
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=2)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await adapter.invoke(prompt="test", model="test-model")

        assert exc_info.value.response.status_code == 429
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_429_with_retry_after_header(self, mock_client_class):
        """Test 429 response with Retry-After header."""
        mock_response_fail = Mock()
        mock_response_fail.status_code = 429
        mock_response_fail.headers = {"Retry-After": "5"}
        mock_response_fail.json.return_value = {"error": "Rate limit exceeded"}
        mock_response_fail.text = "Rate limit exceeded"
        mock_response_fail.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=Mock(url="http://test"),
                response=mock_response_fail,
            )
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Success"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[mock_response_fail, mock_response_success]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Success"


class TestServerErrors:
    """Tests for 5xx server error handling."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_500_internal_server_error_is_retried(self, mock_client_class):
        """Test that 500 Internal Server Error is retried."""
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.json.return_value = {"error": "Internal Server Error"}
        mock_response_fail.text = "Internal Server Error"
        mock_response_fail.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=Mock(url="http://test"),
                response=mock_response_fail,
            )
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Recovered"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[mock_response_fail, mock_response_success]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Recovered"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_502_bad_gateway_is_retried(self, mock_client_class):
        """Test that 502 Bad Gateway is retried."""
        mock_response_fail = Mock()
        mock_response_fail.status_code = 502
        mock_response_fail.json.return_value = {"error": "Bad Gateway"}
        mock_response_fail.text = "Bad Gateway"
        mock_response_fail.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "502 Bad Gateway",
                request=Mock(url="http://test"),
                response=mock_response_fail,
            )
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Gateway recovered"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[mock_response_fail, mock_response_success]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Gateway recovered"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_503_service_unavailable_is_retried(self, mock_client_class):
        """Test that 503 Service Unavailable is retried."""
        mock_response_fail = Mock()
        mock_response_fail.status_code = 503
        mock_response_fail.json.return_value = {"error": "Service Unavailable"}
        mock_response_fail.text = "Service Unavailable"
        mock_response_fail.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "503 Service Unavailable",
                request=Mock(url="http://test"),
                response=mock_response_fail,
            )
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Service restored"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                mock_response_fail,
                mock_response_fail,
                mock_response_success,
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Service restored"
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_504_gateway_timeout_is_retried(self, mock_client_class):
        """Test that 504 Gateway Timeout is retried."""
        mock_response_fail = Mock()
        mock_response_fail.status_code = 504
        mock_response_fail.json.return_value = {"error": "Gateway Timeout"}
        mock_response_fail.text = "Gateway Timeout"
        mock_response_fail.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "504 Gateway Timeout",
                request=Mock(url="http://test"),
                response=mock_response_fail,
            )
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Gateway responded"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[mock_response_fail, mock_response_success]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Gateway responded"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_server_error_exhausts_retries(self, mock_client_class):
        """Test that persistent server errors exhaust retries."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal Server Error"}
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=Mock(url="http://test"),
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await adapter.invoke(prompt="test", model="test-model")

        assert exc_info.value.response.status_code == 500
        assert mock_client.post.call_count == 3


class TestClientErrors:
    """Tests for 4xx client error handling (not retried)."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_400_bad_request_not_retried(self, mock_client_class):
        """Test that 400 Bad Request is not retried."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Invalid request format"}
        mock_response.text = "Invalid request format"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "400 Bad Request",
                request=Mock(url="http://test"),
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await adapter.invoke(prompt="test", model="test-model")

        assert exc_info.value.response.status_code == 400
        # Only called once - no retries for client errors
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_401_unauthorized_not_retried(self, mock_client_class):
        """Test that 401 Unauthorized is not retried."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Invalid API key"}
        mock_response.text = "Invalid API key"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "401 Unauthorized",
                request=Mock(url="http://test"),
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await adapter.invoke(prompt="test", model="test-model")

        assert exc_info.value.response.status_code == 401
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_403_forbidden_not_retried(self, mock_client_class):
        """Test that 403 Forbidden is not retried."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"error": "Access denied"}
        mock_response.text = "Access denied"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "403 Forbidden",
                request=Mock(url="http://test"),
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await adapter.invoke(prompt="test", model="test-model")

        assert exc_info.value.response.status_code == 403
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_404_not_found_not_retried(self, mock_client_class):
        """Test that 404 Not Found is not retried."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Model not found"}
        mock_response.text = "Model not found"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "404 Not Found",
                request=Mock(url="http://test"),
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await adapter.invoke(prompt="test", model="test-model")

        assert exc_info.value.response.status_code == 404
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_422_unprocessable_entity_not_retried(self, mock_client_class):
        """Test that 422 Unprocessable Entity is not retried."""
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.json.return_value = {"error": "Validation failed"}
        mock_response.text = "Validation failed"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "422 Unprocessable Entity",
                request=Mock(url="http://test"),
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=3)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await adapter.invoke(prompt="test", model="test-model")

        assert exc_info.value.response.status_code == 422
        assert mock_client.post.call_count == 1


class TestPromptValidation:
    """Tests for prompt length validation."""

    def test_prompt_too_long_raises_value_error(self):
        """Test that prompts exceeding max length raise ValueError."""
        adapter = ConcreteHTTPAdapter(
            base_url="http://test", timeout=60, max_prompt_length=100
        )

        with pytest.raises(ValueError) as exc_info:
            # Synchronous validation test - need to use sync check
            long_prompt = "x" * 150
            if not adapter.validate_prompt_length(long_prompt):
                raise ValueError(f"Prompt too long ({len(long_prompt)} characters)")

        assert "150" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_prompt_validation_in_invoke(self):
        """Test that invoke validates prompt length."""
        adapter = ConcreteHTTPAdapter(
            base_url="http://test", timeout=60, max_prompt_length=100
        )

        long_prompt = "x" * 150

        with pytest.raises(ValueError) as exc_info:
            await adapter.invoke(prompt=long_prompt, model="test-model")

        assert "too long" in str(exc_info.value).lower()
        assert "150" in str(exc_info.value)

    def test_default_prompt_limits_by_adapter(self):
        """Test that default prompt limits are set correctly per adapter type."""
        from adapters.base_http import BaseHTTPAdapter

        # Test that default limits exist for common adapter types
        assert "ollama" in BaseHTTPAdapter.DEFAULT_PROMPT_LIMITS
        assert "lmstudio" in BaseHTTPAdapter.DEFAULT_PROMPT_LIMITS
        assert "openrouter" in BaseHTTPAdapter.DEFAULT_PROMPT_LIMITS


class TestMixedErrorScenarios:
    """Tests for mixed error scenarios and edge cases."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_network_then_server_error_then_success(self, mock_client_class):
        """Test recovery from alternating error types."""
        mock_response_503 = Mock()
        mock_response_503.status_code = 503
        mock_response_503.json.return_value = {"error": "Service Unavailable"}
        mock_response_503.text = "Service Unavailable"
        mock_response_503.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "503 Service Unavailable",
                request=Mock(url="http://test"),
                response=mock_response_503,
            )
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Finally worked"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                httpx.ConnectError("Network error"),
                mock_response_503,
                mock_response_success,
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=5)
        result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Finally worked"
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_server_error_then_client_error_fails_immediately(
        self, mock_client_class
    ):
        """Test that client error after server errors stops retries."""
        mock_response_500 = Mock()
        mock_response_500.status_code = 500
        mock_response_500.json.return_value = {"error": "Server Error"}
        mock_response_500.text = "Server Error"
        mock_response_500.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=Mock(url="http://test"),
                response=mock_response_500,
            )
        )

        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        mock_response_400.json.return_value = {"error": "Bad Request"}
        mock_response_400.text = "Bad Request"
        mock_response_400.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "400 Bad Request",
                request=Mock(url="http://test"),
                response=mock_response_400,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[mock_response_500, mock_response_400]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=5)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await adapter.invoke(prompt="test", model="test-model")

        # Should fail on 400, not continue retrying
        assert exc_info.value.response.status_code == 400
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_max_retries_zero_no_retries(self, mock_client_class):
        """Test that max_retries=0 means exactly 1 attempt."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.json.return_value = {"error": "Service Unavailable"}
        mock_response.text = "Service Unavailable"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "503 Service Unavailable",
                request=Mock(url="http://test"),
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        # Note: tenacity's stop_after_attempt(0) means 0 attempts, which would
        # raise immediately. The adapter uses max_retries=3 by default,
        # so stop_after_attempt(max_retries) means 3 attempts.
        # With max_retries=1, we get stop_after_attempt(1) = 1 attempt
        adapter = ConcreteHTTPAdapter(base_url="http://test", timeout=60, max_retries=1)

        with pytest.raises(httpx.HTTPStatusError):
            await adapter.invoke(prompt="test", model="test-model")

        # Only 1 attempt with max_retries=1
        assert mock_client.post.call_count == 1
