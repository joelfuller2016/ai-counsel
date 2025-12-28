"""Tests for error normalizer."""

import pytest
import httpx

from error_normalizer import (
    ErrorNormalizer,
    ErrorPattern,
    GENERIC_PATTERNS,
    OPENROUTER_PATTERNS,
    OLLAMA_PATTERNS,
    CLAUDE_PATTERNS,
    GEMINI_PATTERNS,
    LMSTUDIO_PATTERNS,
    normalize_error,
    get_error_response,
    is_retryable_error,
)
from exceptions import (
    AICouncilError,
    AdapterConnectionError,
    AdapterNetworkError,
    AdapterTimeoutError,
    ContentFilteredError,
    EmptyResponseError,
    ErrorCode,
    ErrorDetails,
    InvalidAPIKeyError,
    InvalidModelError,
    InvalidResponseError,
    ModelUnavailableError,
    PermissionDeniedError,
    PromptTooLongError,
    QuotaExceededError,
    RateLimitError,
    ResponseParseError,
    ServiceUnavailableError,
)


class TestErrorPattern:
    """Tests for ErrorPattern class."""

    def test_pattern_matches(self):
        """Test that patterns correctly match error messages."""
        pattern = ErrorPattern(
            pattern=r"rate.?limit",
            exception_class=RateLimitError,
            message_template="Rate limit exceeded",
        )

        assert pattern.matches("Rate limit exceeded, please wait")
        assert pattern.matches("429 rate-limit error")
        assert pattern.matches("RATELIMIT")
        assert not pattern.matches("Connection refused")

    def test_pattern_creates_exception(self):
        """Test that patterns create correct exceptions."""
        pattern = ErrorPattern(
            pattern=r"timeout",
            exception_class=AdapterTimeoutError,
            message_template="Request timed out",
        )

        details = ErrorDetails(provider="test")
        exc = pattern.create_exception("timeout error", details)

        assert isinstance(exc, AdapterTimeoutError)
        assert exc.message == "Request timed out"
        assert exc.details.provider == "test"

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive by default."""
        pattern = ErrorPattern(
            pattern=r"connection.*refused",
            exception_class=AdapterConnectionError,
            message_template="Connection refused",
        )

        assert pattern.matches("Connection refused")
        assert pattern.matches("CONNECTION REFUSED")
        assert pattern.matches("connection was refused by server")


class TestProviderPatterns:
    """Tests for provider-specific error patterns."""

    def test_openrouter_patterns_exist(self):
        """Test that OpenRouter patterns are defined."""
        assert len(OPENROUTER_PATTERNS) > 0

        # Test specific patterns exist
        error_types = set()
        for pattern in OPENROUTER_PATTERNS:
            error_types.add(pattern.exception_class)

        assert InvalidAPIKeyError in error_types
        assert RateLimitError in error_types
        assert ServiceUnavailableError in error_types
        assert InvalidModelError in error_types

    def test_ollama_patterns_exist(self):
        """Test that Ollama patterns are defined."""
        assert len(OLLAMA_PATTERNS) > 0

        error_types = set()
        for pattern in OLLAMA_PATTERNS:
            error_types.add(pattern.exception_class)

        assert AdapterConnectionError in error_types
        assert InvalidModelError in error_types
        assert AdapterTimeoutError in error_types

    def test_claude_patterns_exist(self):
        """Test that Claude patterns are defined."""
        assert len(CLAUDE_PATTERNS) > 0

    def test_gemini_patterns_exist(self):
        """Test that Gemini patterns are defined."""
        assert len(GEMINI_PATTERNS) > 0

    def test_lmstudio_patterns_exist(self):
        """Test that LMStudio patterns are defined."""
        assert len(LMSTUDIO_PATTERNS) > 0

    def test_generic_patterns_exist(self):
        """Test that generic patterns are defined."""
        assert len(GENERIC_PATTERNS) > 0

        error_types = set()
        for pattern in GENERIC_PATTERNS:
            error_types.add(pattern.exception_class)

        assert AdapterConnectionError in error_types
        assert AdapterNetworkError in error_types
        assert AdapterTimeoutError in error_types
        assert EmptyResponseError in error_types


class TestErrorNormalizer:
    """Tests for ErrorNormalizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = ErrorNormalizer()

    def test_normalize_timeout_error(self):
        """Test normalizing TimeoutError."""
        error = TimeoutError("Connection timed out")
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, AdapterTimeoutError)
        assert result.error_code == ErrorCode.ADAPTER_TIMEOUT
        assert result.details.provider == "openrouter"

    def test_normalize_runtime_error_with_rate_limit(self):
        """Test normalizing RuntimeError with rate limit message."""
        error = RuntimeError("429 rate limit exceeded")
        result = self.normalizer.normalize(
            error, provider="openrouter", model="gpt-4"
        )

        assert isinstance(result, RateLimitError)
        assert result.details.provider == "openrouter"
        assert result.details.model == "gpt-4"

    def test_normalize_connection_error(self):
        """Test normalizing connection errors."""
        error = RuntimeError("Connection refused to localhost:11434")
        result = self.normalizer.normalize(error, provider="ollama")

        assert isinstance(result, AdapterConnectionError)
        assert "ollama" in result.details.provider

    def test_normalize_preserves_original_error(self):
        """Test that original error is preserved in details."""
        error = ValueError("Some custom error message")
        result = self.normalizer.normalize(error, provider="test")

        assert result.details.original_error == "Some custom error message"
        assert result.details.original_error_type == "ValueError"

    def test_normalize_already_normalized_error(self):
        """Test that already-normalized errors are returned as-is."""
        original = RateLimitError(
            message="Rate limit",
            details=ErrorDetails(provider="original"),
        )

        result = self.normalizer.normalize(original, provider="openrouter")

        # Should update provider if not set
        assert result is original
        assert result.details.provider == "original"  # Already had provider

    def test_normalize_updates_missing_details(self):
        """Test that normalization fills in missing details."""
        original = RateLimitError()  # No details

        result = self.normalizer.normalize(
            original, provider="openrouter", model="gpt-4"
        )

        # Should fill in provider
        assert result.details.provider == "openrouter"

    def test_normalize_openrouter_auth_error(self):
        """Test normalizing OpenRouter authentication error."""
        error = RuntimeError("401 Unauthorized: Invalid API key")
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, InvalidAPIKeyError)
        assert result.error_code == ErrorCode.INVALID_API_KEY

    def test_normalize_openrouter_model_not_found(self):
        """Test normalizing OpenRouter model not found error."""
        error = RuntimeError("Model 'invalid/model' not found")
        result = self.normalizer.normalize(
            error, provider="openrouter", model="invalid/model"
        )

        assert isinstance(result, InvalidModelError)

    def test_normalize_openrouter_content_filter(self):
        """Test normalizing OpenRouter content filter error."""
        error = RuntimeError("Content blocked by safety filter")
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, ContentFilteredError)

    def test_normalize_ollama_connection_refused(self):
        """Test normalizing Ollama connection refused error."""
        error = RuntimeError("Connection refused: ECONNREFUSED localhost:11434")
        result = self.normalizer.normalize(error, provider="ollama")

        assert isinstance(result, AdapterConnectionError)
        assert "connect" in result.message.lower()

    def test_normalize_ollama_model_not_found(self):
        """Test normalizing Ollama model not found error."""
        error = RuntimeError("Model 'llama3' not found. Please pull the model first.")
        result = self.normalizer.normalize(error, provider="ollama")

        assert isinstance(result, InvalidModelError)
        assert "pull" in result.message.lower()

    def test_normalize_gemini_input_too_long(self):
        """Test normalizing Gemini input too long error."""
        error = RuntimeError("Invalid argument: Input too long")
        result = self.normalizer.normalize(error, provider="gemini")

        assert isinstance(result, PromptTooLongError)

    def test_normalize_gemini_safety_filter(self):
        """Test normalizing Gemini safety filter error."""
        error = RuntimeError("Response blocked for safety reasons")
        result = self.normalizer.normalize(error, provider="gemini")

        assert isinstance(result, ContentFilteredError)

    def test_normalize_unknown_error(self):
        """Test normalizing unknown errors."""
        error = RuntimeError("Some completely unknown error format")
        result = self.normalizer.normalize(error, provider="unknown")

        assert isinstance(result, InvalidResponseError)
        assert result.details.provider == "unknown"


class TestHTTPErrorNormalization:
    """Tests for HTTP error normalization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = ErrorNormalizer()

    def _create_http_error(
        self, status_code: int, body: str = "", headers: dict = None
    ):
        """Helper to create mock HTTP errors."""
        request = httpx.Request("POST", "https://api.test.com/v1/chat")
        response = httpx.Response(
            status_code,
            request=request,
            content=body.encode(),
            headers=headers or {},
        )
        return httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=request,
            response=response,
        )

    def test_normalize_http_401(self):
        """Test normalizing HTTP 401 Unauthorized."""
        error = self._create_http_error(401)
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, InvalidAPIKeyError)
        assert result.details.status_code == 401

    def test_normalize_http_403(self):
        """Test normalizing HTTP 403 Forbidden."""
        error = self._create_http_error(403)
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, PermissionDeniedError)
        assert result.details.status_code == 403

    def test_normalize_http_404(self):
        """Test normalizing HTTP 404 Not Found."""
        error = self._create_http_error(404)
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, InvalidModelError)
        assert result.details.status_code == 404

    def test_normalize_http_429(self):
        """Test normalizing HTTP 429 Rate Limited."""
        error = self._create_http_error(
            429, headers={"retry-after": "30"}
        )
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, RateLimitError)
        assert result.details.status_code == 429
        assert result.details.retry_after == 30

    def test_normalize_http_500(self):
        """Test normalizing HTTP 500 Internal Server Error."""
        error = self._create_http_error(500)
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, ServiceUnavailableError)
        assert result.details.status_code == 500

    def test_normalize_http_502(self):
        """Test normalizing HTTP 502 Bad Gateway."""
        error = self._create_http_error(502)
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, ServiceUnavailableError)

    def test_normalize_http_503(self):
        """Test normalizing HTTP 503 Service Unavailable."""
        error = self._create_http_error(503)
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, ServiceUnavailableError)

    def test_normalize_http_504(self):
        """Test normalizing HTTP 504 Gateway Timeout."""
        error = self._create_http_error(504)
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, AdapterTimeoutError)

    def test_normalize_http_extracts_request_id(self):
        """Test that request ID is extracted from headers."""
        error = self._create_http_error(
            500, headers={"x-request-id": "req-12345"}
        )
        result = self.normalizer.normalize(error, provider="openrouter")

        assert result.details.request_id == "req-12345"

    def test_normalize_http_parses_json_body(self):
        """Test that JSON error body is parsed."""
        error = self._create_http_error(
            400, body='{"error": {"message": "Bad request"}}'
        )
        result = self.normalizer.normalize(error, provider="openrouter")

        assert "response_body" in result.details.additional

    def test_normalize_httpx_timeout(self):
        """Test normalizing httpx TimeoutException."""
        request = httpx.Request("POST", "https://api.test.com")
        error = httpx.TimeoutException("Read timeout", request=request)
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, AdapterTimeoutError)

    def test_normalize_httpx_connect_error(self):
        """Test normalizing httpx ConnectError."""
        request = httpx.Request("POST", "https://api.test.com")
        error = httpx.ConnectError("Connection refused", request=request)
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, AdapterConnectionError)

    def test_normalize_httpx_network_error(self):
        """Test normalizing httpx NetworkError."""
        request = httpx.Request("POST", "https://api.test.com")
        error = httpx.NetworkError("Network unreachable", request=request)
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, AdapterNetworkError)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_normalize_error_function(self):
        """Test normalize_error convenience function."""
        error = TimeoutError("Timeout")
        result = normalize_error(error, provider="test")

        assert isinstance(result, AdapterTimeoutError)

    def test_get_error_response(self):
        """Test get_error_response function."""
        error = RateLimitError(
            message="Rate limit exceeded",
            details=ErrorDetails(provider="openrouter", retry_after=30),
        )
        response = get_error_response(error)

        assert response["error_code"] == "RATE_LIMITED"
        assert response["message"] == "Rate limit exceeded"
        assert response["details"]["provider"] == "openrouter"
        assert response["details"]["retry_after"] == 30

    def test_is_retryable_error_true(self):
        """Test is_retryable_error returns True for retryable errors."""
        retryable_errors = [
            AdapterTimeoutError(),
            AdapterNetworkError(),
            AdapterConnectionError(),
            RateLimitError(),
            ServiceUnavailableError(),
        ]

        for error in retryable_errors:
            assert is_retryable_error(error), f"{type(error).__name__} should be retryable"

    def test_is_retryable_error_false(self):
        """Test is_retryable_error returns False for non-retryable errors."""
        non_retryable_errors = [
            InvalidAPIKeyError(),
            PermissionDeniedError(),
            InvalidModelError(),
            PromptTooLongError(),
            ContentFilteredError(),
        ]

        for error in non_retryable_errors:
            assert not is_retryable_error(error), f"{type(error).__name__} should not be retryable"


class TestProviderSpecificNormalization:
    """Tests for provider-specific error normalization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = ErrorNormalizer()

    def test_openrouter_quota_exceeded(self):
        """Test OpenRouter quota exceeded error."""
        error = RuntimeError("Insufficient credits in your account")
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, QuotaExceededError)

    def test_openrouter_bad_gateway(self):
        """Test OpenRouter 502 bad gateway error."""
        error = RuntimeError("502 Bad Gateway - upstream provider unavailable")
        result = self.normalizer.normalize(error, provider="openrouter")

        assert isinstance(result, ServiceUnavailableError)

    def test_ollama_out_of_memory(self):
        """Test Ollama out of memory error."""
        error = RuntimeError("Out of memory: failed to allocate GPU memory")
        result = self.normalizer.normalize(error, provider="ollama")

        assert isinstance(result, ServiceUnavailableError)
        assert "memory" in result.message.lower()

    def test_claude_service_overloaded(self):
        """Test Claude service overloaded error."""
        error = RuntimeError("503 Service overloaded, please try again")
        result = self.normalizer.normalize(error, provider="claude")

        assert isinstance(result, ServiceUnavailableError)

    def test_gemini_unauthenticated(self):
        """Test Gemini UNAUTHENTICATED error."""
        error = RuntimeError("UNAUTHENTICATED: Invalid API key")
        result = self.normalizer.normalize(error, provider="gemini")

        assert isinstance(result, InvalidAPIKeyError)

    def test_lmstudio_no_model_loaded(self):
        """Test LMStudio no model loaded error."""
        error = RuntimeError("No model is currently loaded")
        result = self.normalizer.normalize(error, provider="lmstudio")

        assert isinstance(result, ModelUnavailableError)
        assert "load" in result.message.lower()


class TestEdgeCases:
    """Tests for edge cases in error normalization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = ErrorNormalizer()

    def test_empty_error_message(self):
        """Test handling empty error message."""
        error = RuntimeError("")
        result = self.normalizer.normalize(error, provider="test")

        assert isinstance(result, InvalidResponseError)

    def test_none_provider(self):
        """Test handling None provider (should use string 'None')."""
        error = TimeoutError("Timeout")
        result = self.normalizer.normalize(error, provider=None)

        assert isinstance(result, AdapterTimeoutError)
        assert result.details.provider is None

    def test_very_long_error_message(self):
        """Test handling very long error messages."""
        long_message = "Error: " + "x" * 10000
        error = RuntimeError(long_message)
        result = self.normalizer.normalize(error, provider="test")

        # Should truncate in the message
        assert len(result.message) < 1000

    def test_unicode_error_message(self):
        """Test handling Unicode in error messages."""
        error = RuntimeError("Error with unicode chars")
        result = self.normalizer.normalize(error, provider="test")

        assert isinstance(result, AICouncilError)

    def test_request_id_propagation(self):
        """Test that request_id is properly propagated."""
        error = RuntimeError("Some error")
        result = self.normalizer.normalize(
            error, provider="test", request_id="req-abc-123"
        )

        assert result.details.request_id == "req-abc-123"
