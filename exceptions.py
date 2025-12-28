"""Custom exception hierarchy with error codes for AI Counsel.

This module provides a structured exception hierarchy with:
- Consistent error codes for programmatic handling
- Detailed error information for debugging
- Support for provider-specific error details
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ErrorCode(str, Enum):
    """Standard error codes for AI Counsel exceptions.

    Error codes are grouped by category:
    - ADAPTER_* : Adapter-level errors (network, timeout, etc.)
    - MODEL_* : Model-specific errors (unavailable, invalid response, etc.)
    - CONFIG_* : Configuration errors
    - VALIDATION_* : Input validation errors
    - DELIBERATION_* : Deliberation engine errors
    - TOOL_* : Tool execution errors
    """

    # Adapter errors (1xx)
    ADAPTER_TIMEOUT = "ADAPTER_TIMEOUT"
    ADAPTER_CONNECTION_ERROR = "ADAPTER_CONNECTION_ERROR"
    ADAPTER_NETWORK_ERROR = "ADAPTER_NETWORK_ERROR"

    # Rate limiting and availability (2xx)
    RATE_LIMITED = "RATE_LIMITED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"

    # Authentication and authorization (3xx)
    AUTH_ERROR = "AUTH_ERROR"
    INVALID_API_KEY = "INVALID_API_KEY"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

    # Model response errors (4xx)
    INVALID_RESPONSE = "INVALID_RESPONSE"
    RESPONSE_PARSE_ERROR = "RESPONSE_PARSE_ERROR"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    CONTENT_FILTERED = "CONTENT_FILTERED"

    # Input validation errors (5xx)
    PROMPT_TOO_LONG = "PROMPT_TOO_LONG"
    INVALID_MODEL = "INVALID_MODEL"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"

    # Configuration errors (6xx)
    CONFIG_ERROR = "CONFIG_ERROR"
    MISSING_CONFIG = "MISSING_CONFIG"
    INVALID_CONFIG = "INVALID_CONFIG"

    # Deliberation errors (7xx)
    DELIBERATION_FAILED = "DELIBERATION_FAILED"
    CONVERGENCE_FAILED = "CONVERGENCE_FAILED"
    VOTING_FAILED = "VOTING_FAILED"
    ALL_PARTICIPANTS_FAILED = "ALL_PARTICIPANTS_FAILED"

    # Tool execution errors (8xx)
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_PERMISSION_DENIED = "TOOL_PERMISSION_DENIED"

    # Unknown/generic errors (9xx)
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class ErrorDetails:
    """Additional details for error context.

    Attributes:
        provider: The provider that generated the error (e.g., 'openrouter', 'ollama')
        model: The model that was being invoked
        original_error: The original error message from the provider
        original_error_type: The type of the original exception
        status_code: HTTP status code if applicable
        retry_after: Seconds to wait before retrying (for rate limits)
        request_id: Request ID from the provider for debugging
        additional: Any additional context-specific information
    """

    provider: Optional[str] = None
    model: Optional[str] = None
    original_error: Optional[str] = None
    original_error_type: Optional[str] = None
    status_code: Optional[int] = None
    retry_after: Optional[int] = None
    request_id: Optional[str] = None
    additional: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert details to dictionary, excluding None values."""
        result = {}
        if self.provider is not None:
            result["provider"] = self.provider
        if self.model is not None:
            result["model"] = self.model
        if self.original_error is not None:
            result["original_error"] = self.original_error
        if self.original_error_type is not None:
            result["original_error_type"] = self.original_error_type
        if self.status_code is not None:
            result["status_code"] = self.status_code
        if self.retry_after is not None:
            result["retry_after"] = self.retry_after
        if self.request_id is not None:
            result["request_id"] = self.request_id
        if self.additional:
            result["additional"] = self.additional
        return result


class AICouncilError(Exception):
    """Base exception for all AI Counsel errors.

    All custom exceptions in AI Counsel should inherit from this class.
    Provides structured error information with:
    - error_code: A standardized error code from ErrorCode enum
    - message: Human-readable error message
    - details: Additional context via ErrorDetails

    Example:
        try:
            response = await adapter.invoke(prompt, model)
        except AICouncilError as e:
            logger.error(
                f"Error {e.error_code}: {e.message}",
                extra={"details": e.details.to_dict()}
            )
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[ErrorDetails] = None,
    ):
        """Initialize AICouncilError.

        Args:
            error_code: Standardized error code
            message: Human-readable error message
            details: Optional additional error context
        """
        self.error_code = error_code
        self.message = message
        self.details = details or ErrorDetails()
        super().__init__(f"[{error_code.value}] {message}")

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details.to_dict(),
        }


# =============================================================================
# Adapter Exceptions
# =============================================================================


class AdapterError(AICouncilError):
    """Base exception for adapter-related errors."""

    pass


class AdapterTimeoutError(AdapterError):
    """Raised when an adapter operation times out."""

    def __init__(
        self,
        message: str = "Adapter operation timed out",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.ADAPTER_TIMEOUT, message, details)


class AdapterConnectionError(AdapterError):
    """Raised when connection to the provider fails."""

    def __init__(
        self,
        message: str = "Failed to connect to provider",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.ADAPTER_CONNECTION_ERROR, message, details)


class AdapterNetworkError(AdapterError):
    """Raised for general network errors."""

    def __init__(
        self,
        message: str = "Network error occurred",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.ADAPTER_NETWORK_ERROR, message, details)


# =============================================================================
# Rate Limiting and Availability Exceptions
# =============================================================================


class RateLimitError(AICouncilError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.RATE_LIMITED, message, details)


class ModelUnavailableError(AICouncilError):
    """Raised when a specific model is unavailable."""

    def __init__(
        self,
        message: str = "Model is unavailable",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.MODEL_UNAVAILABLE, message, details)


class ServiceUnavailableError(AICouncilError):
    """Raised when the service is temporarily unavailable (503)."""

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.SERVICE_UNAVAILABLE, message, details)


class ProviderUnavailableError(AICouncilError):
    """Raised when an entire provider is unavailable."""

    def __init__(
        self,
        message: str = "Provider is unavailable",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.PROVIDER_UNAVAILABLE, message, details)


# =============================================================================
# Authentication and Authorization Exceptions
# =============================================================================


class AuthenticationError(AICouncilError):
    """Base exception for authentication-related errors."""

    pass


class InvalidAPIKeyError(AuthenticationError):
    """Raised when API key is invalid or missing."""

    def __init__(
        self,
        message: str = "Invalid or missing API key",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.INVALID_API_KEY, message, details)


class PermissionDeniedError(AuthenticationError):
    """Raised when access to a resource is denied."""

    def __init__(
        self,
        message: str = "Permission denied",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.PERMISSION_DENIED, message, details)


class QuotaExceededError(AuthenticationError):
    """Raised when usage quota is exceeded."""

    def __init__(
        self,
        message: str = "Usage quota exceeded",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.QUOTA_EXCEEDED, message, details)


# =============================================================================
# Model Response Exceptions
# =============================================================================


class ResponseError(AICouncilError):
    """Base exception for response-related errors."""

    pass


class InvalidResponseError(ResponseError):
    """Raised when response format is invalid."""

    def __init__(
        self,
        message: str = "Invalid response format",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.INVALID_RESPONSE, message, details)


class ResponseParseError(ResponseError):
    """Raised when response cannot be parsed."""

    def __init__(
        self,
        message: str = "Failed to parse response",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.RESPONSE_PARSE_ERROR, message, details)


class EmptyResponseError(ResponseError):
    """Raised when response is empty."""

    def __init__(
        self,
        message: str = "Empty response received",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.EMPTY_RESPONSE, message, details)


class ContentFilteredError(ResponseError):
    """Raised when content is filtered by the provider."""

    def __init__(
        self,
        message: str = "Content was filtered",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.CONTENT_FILTERED, message, details)


# =============================================================================
# Validation Exceptions
# =============================================================================


class ValidationError(AICouncilError):
    """Base exception for validation errors."""

    pass


class PromptTooLongError(ValidationError):
    """Raised when prompt exceeds maximum length."""

    def __init__(
        self,
        message: str = "Prompt exceeds maximum length",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.PROMPT_TOO_LONG, message, details)


class InvalidModelError(ValidationError):
    """Raised when model identifier is invalid."""

    def __init__(
        self,
        message: str = "Invalid model identifier",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.INVALID_MODEL, message, details)


class InvalidParameterError(ValidationError):
    """Raised when a parameter value is invalid."""

    def __init__(
        self,
        message: str = "Invalid parameter value",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.INVALID_PARAMETER, message, details)


class MissingRequiredFieldError(ValidationError):
    """Raised when a required field is missing."""

    def __init__(
        self,
        message: str = "Missing required field",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.MISSING_REQUIRED_FIELD, message, details)


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigurationError(AICouncilError):
    """Base exception for configuration errors."""

    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(
        self,
        message: str = "Missing required configuration",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.MISSING_CONFIG, message, details)


class InvalidConfigError(ConfigurationError):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        message: str = "Invalid configuration",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.INVALID_CONFIG, message, details)


# =============================================================================
# Deliberation Exceptions
# =============================================================================


class DeliberationError(AICouncilError):
    """Base exception for deliberation-related errors."""

    pass


class DeliberationFailedError(DeliberationError):
    """Raised when a deliberation fails completely."""

    def __init__(
        self,
        message: str = "Deliberation failed",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.DELIBERATION_FAILED, message, details)


class ConvergenceFailedError(DeliberationError):
    """Raised when convergence detection fails."""

    def __init__(
        self,
        message: str = "Convergence detection failed",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.CONVERGENCE_FAILED, message, details)


class VotingFailedError(DeliberationError):
    """Raised when voting fails."""

    def __init__(
        self,
        message: str = "Voting failed",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.VOTING_FAILED, message, details)


class AllParticipantsFailedError(DeliberationError):
    """Raised when all participants fail in a round."""

    def __init__(
        self,
        message: str = "All participants failed",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.ALL_PARTICIPANTS_FAILED, message, details)


# =============================================================================
# Tool Execution Exceptions
# =============================================================================


class ToolError(AICouncilError):
    """Base exception for tool execution errors."""

    pass


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""

    def __init__(
        self,
        message: str = "Tool execution failed",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.TOOL_EXECUTION_ERROR, message, details)


class ToolTimeoutError(ToolError):
    """Raised when tool execution times out."""

    def __init__(
        self,
        message: str = "Tool execution timed out",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.TOOL_TIMEOUT, message, details)


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found."""

    def __init__(
        self,
        message: str = "Tool not found",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.TOOL_NOT_FOUND, message, details)


class ToolPermissionDeniedError(ToolError):
    """Raised when tool access is denied."""

    def __init__(
        self,
        message: str = "Tool access denied",
        details: Optional[ErrorDetails] = None,
    ):
        super().__init__(ErrorCode.TOOL_PERMISSION_DENIED, message, details)
