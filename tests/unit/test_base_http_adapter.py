"""Unit tests for BaseHTTPAdapter."""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest


# Concrete test implementation for testing abstract base
class ConcreteHTTPAdapter:
    """Concrete implementation for testing BaseHTTPAdapter."""

    def __init__(
        self,
        base_url: str,
        timeout: int = 60,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key
        self.default_headers = headers or {}

    def build_request(self, model: str, prompt: str):
        """Test implementation."""
        return (
            "/test",
            {"Authorization": "Bearer test", "Content-Type": "application/json"},
            {"model": model, "prompt": prompt},
        )

    def parse_response(self, response_json: dict) -> str:
        """Test implementation."""
        return response_json.get("response", "")


class TestBaseHTTPAdapter:
    """Tests for BaseHTTPAdapter abstract class behavior."""

    def test_cannot_instantiate_base_adapter(self):
        """Test that BaseHTTPAdapter cannot be instantiated directly."""
        from adapters.base_http import BaseHTTPAdapter

        # Abstract classes should raise TypeError when instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseHTTPAdapter(base_url="http://test", timeout=60)

    def test_subclass_must_implement_build_request(self):
        """Test that subclass must implement build_request method."""
        from adapters.base_http import BaseHTTPAdapter

        class IncompleteAdapter(BaseHTTPAdapter):
            def parse_response(self, response_json):
                return str(response_json)

        # Should raise TypeError because build_request is not implemented
        with pytest.raises(TypeError, match="abstract"):
            IncompleteAdapter(base_url="http://test", timeout=60)

    def test_subclass_must_implement_parse_response(self):
        """Test that subclass must implement parse_response method."""
        from adapters.base_http import BaseHTTPAdapter

        class IncompleteAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/test", {}, {})

        # Should raise TypeError because parse_response is not implemented
        with pytest.raises(TypeError, match="abstract"):
            IncompleteAdapter(base_url="http://test", timeout=60)

    def test_complete_subclass_can_be_instantiated(self):
        """Test that complete subclass can be instantiated."""
        from adapters.base_http import BaseHTTPAdapter

        class CompleteAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json.get("text", "")

        # Should work without errors
        adapter = CompleteAdapter(base_url="http://localhost:8080", timeout=60)
        assert adapter.base_url == "http://localhost:8080"
        assert adapter.timeout == 60

    def test_base_url_trailing_slash_removed(self):
        """Test that trailing slash is removed from base_url."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/test", {}, {})

            def parse_response(self, response_json):
                return str(response_json)

        adapter = TestAdapter(base_url="http://localhost:8080/", timeout=60)
        assert adapter.base_url == "http://localhost:8080"


class TestHTTPAdapterInvoke:
    """Tests for BaseHTTPAdapter invoke method."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_invoke_success(self, mock_client_class):
        """Test successful HTTP request."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return (
                    "/api/test",
                    {"Content-Type": "application/json"},
                    {"model": model, "prompt": prompt},
                )

            def parse_response(self, response_json):
                return response_json["response"]

        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test response"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60)
        result = await adapter.invoke(prompt="test prompt", model="test-model")

        assert result == "Test response"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_invoke_with_context(self, mock_client_class):
        """Test invoke with context prepended to prompt."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test response"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60)
        await adapter.invoke(
            prompt="test prompt", model="test-model", context="Previous context"
        )

        # Check that post was called with context prepended
        call_args = mock_client.post.call_args
        body = call_args.kwargs["json"]
        assert "Previous context" in body["prompt"]
        assert "test prompt" in body["prompt"]

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_invoke_retries_on_503(self, mock_client_class):
        """Test retry logic on 503 Service Unavailable."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        # Setup mock to fail twice with 503, then succeed
        mock_response_fail = Mock()
        mock_response_fail.status_code = 503
        mock_response_fail.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "503 Service Unavailable",
                request=Mock(url="http://test"),
                response=mock_response_fail,
            )
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Success after retry"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[mock_response_fail, mock_response_fail, mock_response_success]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Success after retry"
        assert mock_client.post.call_count == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_invoke_no_retry_on_400(self, mock_client_class):
        """Test that 4xx errors are not retried."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_response.text = '{"error": "Bad request"}'
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

        adapter = TestAdapter(base_url="http://test", timeout=60, max_retries=3)

        # Should raise immediately without retries
        with pytest.raises(httpx.HTTPStatusError, match="400"):
            await adapter.invoke(prompt="test", model="test-model")

        # Should only be called once (no retries for 4xx)
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_invoke_retries_on_network_error(self, mock_client_class):
        """Test retry logic on network errors."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        # Simulate network error then success
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "response": "Success after network error"
        }
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[httpx.ConnectError("Connection failed"), mock_response_success]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke(prompt="test", model="test-model")

        assert result == "Success after network error"
        assert mock_client.post.call_count == 2  # 1 failure + 1 success

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_invoke_timeout_error(self, mock_client_class):
        """Test that timeout raises TimeoutError."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=1, max_retries=1)

        with pytest.raises(TimeoutError, match="timed out"):
            await adapter.invoke(prompt="test", model="test-model")


class TestHTTPAdapterFallback:
    """Tests for BaseHTTPAdapter invoke_with_fallback method."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fallback_primary_succeeds(self, mock_client_class):
        """Test that fallback is not used when primary model succeeds."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Primary success"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60)
        result = await adapter.invoke_with_fallback(
            prompt="test prompt",
            model="primary-model",
            fallback_models=["fallback-1", "fallback-2"],
        )

        assert result == "Primary success"
        # Only primary model should be called
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fallback_on_timeout(self, mock_client_class):
        """Test fallback triggers on timeout error."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        # Primary times out, first fallback succeeds
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Fallback success"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[asyncio.TimeoutError(), mock_response_success]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60, max_retries=1)
        result = await adapter.invoke_with_fallback(
            prompt="test prompt",
            model="primary-model",
            fallback_models=["fallback-1"],
        )

        assert result == "Fallback success"
        # Primary + 1 fallback
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fallback_on_429_rate_limit(self, mock_client_class):
        """Test fallback triggers on 429 rate limit error."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        # Create 429 response with proper json() and text mocking
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.json.return_value = {"error": "Rate limit exceeded"}
        mock_response_429.text = '{"error": "Rate limit exceeded"}'
        mock_response_429.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=Mock(url="http://test"),
                response=mock_response_429,
            )
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Fallback success"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        # Primary fails with 429 (retries exhausted), fallback succeeds
        mock_client.post = AsyncMock(
            side_effect=[
                mock_response_429,
                mock_response_429,
                mock_response_429,  # 3 retries for primary
                mock_response_success,  # fallback succeeds
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke_with_fallback(
            prompt="test prompt",
            model="primary-model",
            fallback_models=["fallback-1"],
        )

        assert result == "Fallback success"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fallback_on_500_server_error(self, mock_client_class):
        """Test fallback triggers on 500 server error."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        mock_response_500 = Mock()
        mock_response_500.status_code = 500
        mock_response_500.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=Mock(url="http://test"),
                response=mock_response_500,
            )
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Fallback success"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                mock_response_500,
                mock_response_500,
                mock_response_500,  # 3 retries for primary
                mock_response_success,  # fallback succeeds
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60, max_retries=3)
        result = await adapter.invoke_with_fallback(
            prompt="test prompt",
            model="primary-model",
            fallback_models=["fallback-1"],
        )

        assert result == "Fallback success"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fallback_chain_multiple_failures(self, mock_client_class):
        """Test multiple fallbacks are tried in order."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        # Track which models are called
        models_called = []

        def track_model_calls(*args, **kwargs):
            model = kwargs.get("json", {}).get("model", "unknown")
            models_called.append(model)
            if model == "third-fallback":
                mock_success = Mock()
                mock_success.status_code = 200
                mock_success.json.return_value = {"response": "Third fallback success"}
                mock_success.raise_for_status = Mock()
                return mock_success
            else:
                raise httpx.ConnectError("Connection failed")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=track_model_calls)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60, max_retries=1)
        result = await adapter.invoke_with_fallback(
            prompt="test prompt",
            model="primary-model",
            fallback_models=["first-fallback", "second-fallback", "third-fallback"],
        )

        assert result == "Third fallback success"
        assert models_called == [
            "primary-model",
            "first-fallback",
            "second-fallback",
            "third-fallback",
        ]

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fallback_all_fail(self, mock_client_class):
        """Test that exception is raised when all models fail."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection failed")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60, max_retries=1)

        with pytest.raises(httpx.ConnectError):
            await adapter.invoke_with_fallback(
                prompt="test prompt",
                model="primary-model",
                fallback_models=["fallback-1", "fallback-2"],
            )

        # All models should be tried: primary + 2 fallbacks
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fallback_empty_list(self, mock_client_class):
        """Test with empty fallback list behaves like regular invoke."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Primary success"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60)
        result = await adapter.invoke_with_fallback(
            prompt="test prompt",
            model="primary-model",
            fallback_models=[],  # Empty fallback list
        )

        assert result == "Primary success"
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fallback_no_retry_on_400(self, mock_client_class):
        """Test that 400 errors trigger fallback (client error may be model-specific)."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        mock_response_400.json.return_value = {"error": "Bad request"}
        mock_response_400.text = '{"error": "Bad request"}'
        mock_response_400.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "400 Bad Request",
                request=Mock(url="http://test"),
                response=mock_response_400,
            )
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Fallback success"}
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        # 400 errors on primary trigger fallback (e.g., model-specific issues)
        mock_client.post = AsyncMock(
            side_effect=[mock_response_400, mock_response_success]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60, max_retries=1)

        # 400 errors DO trigger fallback since they may be model-specific
        result = await adapter.invoke_with_fallback(
            prompt="test prompt",
            model="primary-model",
            fallback_models=["fallback-1"],
        )

        # Fallback should have succeeded
        assert result == "Fallback success"
        # Both models should be called
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fallback_no_retry_on_valueerror(self, mock_client_class):
        """Test that ValueError (e.g., prompt too long) doesn't trigger fallback."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            ADAPTER_NAME = "test"

            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        adapter = TestAdapter(base_url="http://test", timeout=60, max_prompt_length=10)

        # Long prompt should raise ValueError without trying fallbacks
        with pytest.raises(ValueError, match="Prompt too long"):
            await adapter.invoke_with_fallback(
                prompt="This is a very long prompt that exceeds the limit",
                model="primary-model",
                fallback_models=["fallback-1"],
            )

        # No HTTP calls should be made
        mock_client_class.return_value.post.assert_not_called()


class TestHTTPAdapterHealthCheck:
    """Tests for BaseHTTPAdapter health_check method."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_health_check_success(self, mock_client_class):
        """Test successful health check."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            ADAPTER_NAME = "test"

            def build_request(self, model, prompt):
                return (
                    "/api/test",
                    {"Content-Type": "application/json"},
                    {"model": model, "prompt": prompt, "max_tokens": 100},
                )

            def parse_response(self, response_json):
                return response_json["response"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hi"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60)
        result = await adapter.health_check(model="test-model")

        assert result.available is True
        assert result.model == "test-model"
        assert result.adapter == "test"
        assert result.latency_ms is not None
        assert result.error is None

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_health_check_timeout(self, mock_client_class):
        """Test health check handles timeout."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            ADAPTER_NAME = "test"

            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60)
        result = await adapter.health_check(model="test-model", timeout=5.0)

        assert result.available is False
        assert result.model == "test-model"
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_health_check_http_error(self, mock_client_class):
        """Test health check handles HTTP errors."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            ADAPTER_NAME = "test"

            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.json.return_value = {"error": "Service unavailable"}
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

        adapter = TestAdapter(base_url="http://test", timeout=60)
        result = await adapter.health_check(model="test-model")

        assert result.available is False
        assert "503" in result.error
        assert result.latency_ms is not None

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_health_check_connection_error(self, mock_client_class):
        """Test health check handles connection errors."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            ADAPTER_NAME = "test"

            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60)
        result = await adapter.health_check(model="test-model")

        assert result.available is False
        assert "ConnectError" in result.error

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_health_check_custom_timeout(self, mock_client_class):
        """Test health check uses custom timeout."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            ADAPTER_NAME = "test"

            def build_request(self, model, prompt):
                return ("/api/test", {}, {"model": model, "prompt": prompt})

            def parse_response(self, response_json):
                return response_json["response"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hi"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60)
        result = await adapter.health_check(model="test-model", timeout=5.0)

        assert result.available is True
        # Verify the custom timeout was used
        mock_client_class.assert_called_with(timeout=5.0)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_health_check_minimizes_max_tokens(self, mock_client_class):
        """Test health check sets max_tokens to 1 for minimal response."""
        from adapters.base_http import BaseHTTPAdapter

        class TestAdapter(BaseHTTPAdapter):
            ADAPTER_NAME = "test"

            def build_request(self, model, prompt):
                return (
                    "/api/test",
                    {},
                    {"model": model, "prompt": prompt, "max_tokens": 1000},
                )

            def parse_response(self, response_json):
                return response_json["response"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "H"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        adapter = TestAdapter(base_url="http://test", timeout=60)
        await adapter.health_check(model="test-model")

        # Verify max_tokens was set to 1 for minimal response
        call_args = mock_client.post.call_args
        body = call_args.kwargs["json"]
        assert body["max_tokens"] == 1
