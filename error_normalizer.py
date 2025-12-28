"""Error normalizer for consistent error handling across providers.

This module normalizes provider-specific error messages and responses
into a consistent structure using the custom exception hierarchy.

Supports:
- OpenRouter
- Ollama
- Claude
- Gemini
- LMStudio
"""

import re
from typing import Any, Optional, Type

import httpx

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


class ErrorPattern:
    """Pattern for matching and normalizing errors."""

    def __init__(
        self,
        pattern: str,
        exception_class: Type[AICouncilError],
        message_template: str,
        flags: int = re.IGNORECASE,
    ):
        """Initialize error pattern.

        Args:
            pattern: Regex pattern to match error messages
            exception_class: Exception class to raise
            message_template: Template for normalized message
            flags: Regex flags (default: case-insensitive)
        """
        self.regex = re.compile(pattern, flags)
        self.exception_class = exception_class
        self.message_template = message_template

    def matches(self, error_msg: str) -> bool:
        """Check if error message matches this pattern."""
        return bool(self.regex.search(error_msg))

    def create_exception(
        self,
        error_msg: str,
        details: Optional[ErrorDetails] = None,
    ) -> AICouncilError:
        """Create normalized exception from error message."""
        return self.exception_class(
            message=self.message_template,
            details=details,
        )


# =============================================================================
# Provider-Specific Error Patterns
# =============================================================================

# OpenRouter error patterns
OPENROUTER_PATTERNS = [
    ErrorPattern(
        r"(401|unauthorized|invalid.*api.?key|authentication failed)",
        InvalidAPIKeyError,
        "Invalid or missing OpenRouter API key",
    ),
    ErrorPattern(
        r"(429|rate.?limit|too many requests)",
        RateLimitError,
        "OpenRouter rate limit exceeded. Please wait and retry.",
    ),
    ErrorPattern(
        r"(503|service.?unavailable|overloaded|over capacity)",
        ServiceUnavailableError,
        "OpenRouter service temporarily unavailable",
    ),
    ErrorPattern(
        r"(502|bad gateway)",
        ServiceUnavailableError,
        "OpenRouter gateway error. The upstream provider may be unavailable.",
    ),
    ErrorPattern(
        r"(model.*not.*found|unknown.*model|invalid.*model)",
        InvalidModelError,
        "Model not found on OpenRouter",
    ),
    ErrorPattern(
        r"(context.*length|token.*limit|prompt.*too.*long|max.*tokens.*exceeded)",
        PromptTooLongError,
        "Prompt exceeds model context length limit",
    ),
    ErrorPattern(
        r"(content.*filter|safety|moderation|blocked)",
        ContentFilteredError,
        "Content was filtered by safety systems",
    ),
    ErrorPattern(
        r"(quota.*exceeded|billing|payment|insufficient.*credits)",
        QuotaExceededError,
        "OpenRouter account quota exceeded",
    ),
    ErrorPattern(
        r"(permission.*denied|access.*denied|forbidden|403)",
        PermissionDeniedError,
        "Access denied to OpenRouter resource",
    ),
]

# Ollama error patterns
OLLAMA_PATTERNS = [
    ErrorPattern(
        r"(connection.*refused|cannot connect|connection error|ECONNREFUSED)",
        AdapterConnectionError,
        "Cannot connect to Ollama. Is the server running?",
    ),
    ErrorPattern(
        r"(model.*not.*found|pull.*model|no.*model)",
        InvalidModelError,
        "Model not found in Ollama. Run 'ollama pull <model>' to download.",
    ),
    ErrorPattern(
        r"(timeout|timed out|deadline exceeded)",
        AdapterTimeoutError,
        "Ollama request timed out",
    ),
    ErrorPattern(
        r"(out of memory|OOM|memory allocation|not enough memory)",
        ServiceUnavailableError,
        "Ollama out of memory. Try a smaller model or reduce context size.",
    ),
    ErrorPattern(
        r"(gpu.*error|cuda.*error|metal.*error)",
        ServiceUnavailableError,
        "Ollama GPU error. Check GPU drivers and memory.",
    ),
]

# Claude CLI error patterns
CLAUDE_PATTERNS = [
    ErrorPattern(
        r"(api.*key|authentication|unauthorized|401)",
        InvalidAPIKeyError,
        "Invalid Claude API key or authentication failed",
    ),
    ErrorPattern(
        r"(rate.?limit|429|too many requests|over.?capacity)",
        RateLimitError,
        "Claude rate limit exceeded. Please wait and retry.",
    ),
    ErrorPattern(
        r"(503|overloaded|service.*unavailable|over capacity)",
        ServiceUnavailableError,
        "Claude service temporarily overloaded",
    ),
    ErrorPattern(
        r"(timeout|timed out)",
        AdapterTimeoutError,
        "Claude request timed out",
    ),
    ErrorPattern(
        r"(context.*length|max.*tokens|prompt.*too.*long|input too long)",
        PromptTooLongError,
        "Prompt exceeds Claude context length limit",
    ),
    ErrorPattern(
        r"(content.*policy|safety|harmful|blocked)",
        ContentFilteredError,
        "Content was filtered by Claude safety systems",
    ),
    ErrorPattern(
        r"(model.*not.*found|invalid.*model|unknown.*model)",
        InvalidModelError,
        "Invalid Claude model specified",
    ),
]

# Gemini CLI error patterns
GEMINI_PATTERNS = [
    ErrorPattern(
        r"(api.*key|authentication|unauthorized|401|UNAUTHENTICATED)",
        InvalidAPIKeyError,
        "Invalid Gemini API key or authentication failed",
    ),
    ErrorPattern(
        r"(rate.?limit|429|resource.*exhausted|quota)",
        RateLimitError,
        "Gemini rate limit exceeded. Please wait and retry.",
    ),
    ErrorPattern(
        r"(503|service.*unavailable|unavailable)",
        ServiceUnavailableError,
        "Gemini service temporarily unavailable",
    ),
    ErrorPattern(
        r"(timeout|deadline.*exceeded)",
        AdapterTimeoutError,
        "Gemini request timed out",
    ),
    ErrorPattern(
        r"(invalid.*argument|input too long|max.*input|context.*length)",
        PromptTooLongError,
        "Prompt exceeds Gemini context length limit",
    ),
    ErrorPattern(
        r"(safety|blocked|harm|recitation)",
        ContentFilteredError,
        "Content was filtered by Gemini safety systems",
    ),
    ErrorPattern(
        r"(model.*not.*found|invalid.*model|not.*supported)",
        InvalidModelError,
        "Invalid Gemini model specified",
    ),
    ErrorPattern(
        r"(permission.*denied|forbidden|403)",
        PermissionDeniedError,
        "Access denied to Gemini API",
    ),
]

# LMStudio error patterns
LMSTUDIO_PATTERNS = [
    ErrorPattern(
        r"(connection.*refused|cannot connect|ECONNREFUSED)",
        AdapterConnectionError,
        "Cannot connect to LM Studio. Is the server running?",
    ),
    ErrorPattern(
        r"(no.*model.*loaded|model not loaded)",
        ModelUnavailableError,
        "No model loaded in LM Studio. Please load a model first.",
    ),
    ErrorPattern(
        r"(timeout|timed out)",
        AdapterTimeoutError,
        "LM Studio request timed out",
    ),
    ErrorPattern(
        r"(context.*length|max.*tokens|too long)",
        PromptTooLongError,
        "Prompt exceeds model context length limit",
    ),
]

# Generic patterns (apply to all providers)
GENERIC_PATTERNS = [
    ErrorPattern(
        r"(connection.*refused|cannot connect|ECONNREFUSED)",
        AdapterConnectionError,
        "Connection refused. Is the service running?",
    ),
    ErrorPattern(
        r"(network.*error|connection.*reset|connection.*closed)",
        AdapterNetworkError,
        "Network error occurred",
    ),
    ErrorPattern(
        r"(timeout|timed out|deadline exceeded)",
        AdapterTimeoutError,
        "Request timed out",
    ),
    ErrorPattern(
        r"(empty.*response|no.*content|null.*response)",
        EmptyResponseError,
        "Received empty response from model",
    ),
    ErrorPattern(
        r"(parse.*error|invalid.*json|json.*decode|malformed)",
        ResponseParseError,
        "Failed to parse response",
    ),
]

# Provider pattern mapping
PROVIDER_PATTERNS = {
    "openrouter": OPENROUTER_PATTERNS,
    "ollama": OLLAMA_PATTERNS,
    "claude": CLAUDE_PATTERNS,
    "gemini": GEMINI_PATTERNS,
    "lmstudio": LMSTUDIO_PATTERNS,
}


class ErrorNormalizer:
    """Normalizes provider-specific errors to consistent exceptions.

    Example:
        normalizer = ErrorNormalizer()

        # Normalize a provider error
        try:
            response = await adapter.invoke(prompt, model)
        except Exception as e:
            normalized = normalizer.normalize(
                error=e,
                provider="openrouter",
                model="anthropic/claude-3.5-sonnet"
            )
            raise normalized
    """

    def __init__(self):
        """Initialize error normalizer."""
        pass

    def normalize(
        self,
        error: Exception,
        provider: str,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AICouncilError:
        """Normalize an exception to a consistent AICouncilError.

        Args:
            error: The original exception
            provider: Provider name (openrouter, ollama, claude, gemini, lmstudio)
            model: Optional model identifier
            request_id: Optional request ID for debugging

        Returns:
            Normalized AICouncilError
        """
        # Build error details
        details = ErrorDetails(
            provider=provider,
            model=model,
            original_error=str(error),
            original_error_type=type(error).__name__,
            request_id=request_id,
        )

        # Handle httpx errors specially
        if isinstance(error, httpx.HTTPStatusError):
            return self._normalize_http_error(error, provider, details)

        if isinstance(error, httpx.TimeoutException):
            return AdapterTimeoutError(
                message=f"Request to {provider} timed out",
                details=details,
            )

        if isinstance(error, httpx.ConnectError):
            return AdapterConnectionError(
                message=f"Failed to connect to {provider}",
                details=details,
            )

        if isinstance(error, httpx.NetworkError):
            return AdapterNetworkError(
                message=f"Network error connecting to {provider}",
                details=details,
            )

        # Handle asyncio.TimeoutError
        if isinstance(error, TimeoutError):
            return AdapterTimeoutError(
                message=f"Request to {provider} timed out",
                details=details,
            )

        # Handle RuntimeError (common for CLI adapters)
        if isinstance(error, RuntimeError):
            return self._normalize_by_message(str(error), provider, details)

        # Handle ValueError (validation errors)
        if isinstance(error, ValueError):
            return self._normalize_by_message(str(error), provider, details)

        # Handle already-normalized errors
        if isinstance(error, AICouncilError):
            # Update details if not already set
            if not error.details.provider:
                error.details.provider = provider
            if not error.details.model and model:
                error.details.model = model
            return error

        # Try to normalize by error message
        return self._normalize_by_message(str(error), provider, details)

    def _normalize_http_error(
        self,
        error: httpx.HTTPStatusError,
        provider: str,
        details: ErrorDetails,
    ) -> AICouncilError:
        """Normalize HTTP status errors."""
        status_code = error.response.status_code
        details.status_code = status_code

        # Try to extract error body
        try:
            error_body = error.response.json()
            details.additional["response_body"] = error_body
            error_msg = str(error_body)
        except Exception:
            error_msg = error.response.text

        # Check for retry-after header
        retry_after = error.response.headers.get("retry-after")
        if retry_after:
            try:
                details.retry_after = int(retry_after)
            except ValueError:
                pass

        # Check for request ID in headers
        request_id = (
            error.response.headers.get("x-request-id")
            or error.response.headers.get("x-trace-id")
        )
        if request_id:
            details.request_id = request_id

        # Map status codes to exceptions
        if status_code == 401:
            return InvalidAPIKeyError(
                message=f"Authentication failed with {provider}",
                details=details,
            )
        elif status_code == 403:
            return PermissionDeniedError(
                message=f"Access denied to {provider}",
                details=details,
            )
        elif status_code == 404:
            return InvalidModelError(
                message=f"Model or endpoint not found on {provider}",
                details=details,
            )
        elif status_code == 429:
            return RateLimitError(
                message=f"Rate limit exceeded on {provider}",
                details=details,
            )
        elif status_code == 500:
            return ServiceUnavailableError(
                message=f"Internal server error from {provider}",
                details=details,
            )
        elif status_code == 502:
            return ServiceUnavailableError(
                message=f"Bad gateway from {provider}",
                details=details,
            )
        elif status_code == 503:
            return ServiceUnavailableError(
                message=f"Service unavailable from {provider}",
                details=details,
            )
        elif status_code == 504:
            return AdapterTimeoutError(
                message=f"Gateway timeout from {provider}",
                details=details,
            )

        # Try to match by error message
        return self._normalize_by_message(error_msg, provider, details)

    def _normalize_by_message(
        self,
        error_msg: str,
        provider: str,
        details: ErrorDetails,
    ) -> AICouncilError:
        """Normalize error by matching message patterns."""
        # Try provider-specific patterns first
        provider_patterns = PROVIDER_PATTERNS.get(provider.lower(), [])
        for pattern in provider_patterns:
            if pattern.matches(error_msg):
                return pattern.create_exception(error_msg, details)

        # Try generic patterns
        for pattern in GENERIC_PATTERNS:
            if pattern.matches(error_msg):
                return pattern.create_exception(error_msg, details)

        # No pattern matched - return a generic error
        return InvalidResponseError(
            message=f"Unexpected error from {provider}: {error_msg[:200]}",
            details=details,
        )


# Singleton instance for convenience
_normalizer = ErrorNormalizer()


def normalize_error(
    error: Exception,
    provider: str,
    model: Optional[str] = None,
    request_id: Optional[str] = None,
) -> AICouncilError:
    """Convenience function to normalize errors.

    Args:
        error: The original exception
        provider: Provider name
        model: Optional model identifier
        request_id: Optional request ID

    Returns:
        Normalized AICouncilError
    """
    return _normalizer.normalize(error, provider, model, request_id)


def get_error_response(
    error: AICouncilError,
) -> dict[str, Any]:
    """Get a standardized error response dictionary.

    Args:
        error: The AICouncilError to convert

    Returns:
        Dictionary with error_code, message, and details
    """
    return error.to_dict()


def is_retryable_error(error: AICouncilError) -> bool:
    """Check if an error should be retried.

    Args:
        error: The error to check

    Returns:
        True if the error is retryable
    """
    retryable_codes = {
        ErrorCode.ADAPTER_TIMEOUT,
        ErrorCode.ADAPTER_NETWORK_ERROR,
        ErrorCode.ADAPTER_CONNECTION_ERROR,
        ErrorCode.RATE_LIMITED,
        ErrorCode.SERVICE_UNAVAILABLE,
        ErrorCode.PROVIDER_UNAVAILABLE,
    }
    return error.error_code in retryable_codes
