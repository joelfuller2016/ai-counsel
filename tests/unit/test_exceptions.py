"""Tests for custom exception hierarchy."""

import pytest

from exceptions import (
    AICouncilError,
    AdapterConnectionError,
    AdapterError,
    AdapterNetworkError,
    AdapterTimeoutError,
    AllParticipantsFailedError,
    AuthenticationError,
    ConfigurationError,
    ContentFilteredError,
    ConvergenceFailedError,
    DeliberationError,
    DeliberationFailedError,
    EmptyResponseError,
    ErrorCode,
    ErrorDetails,
    InvalidAPIKeyError,
    InvalidConfigError,
    InvalidModelError,
    InvalidParameterError,
    InvalidResponseError,
    MissingConfigError,
    MissingRequiredFieldError,
    ModelUnavailableError,
    PermissionDeniedError,
    PromptTooLongError,
    QuotaExceededError,
    RateLimitError,
    ResponseError,
    ResponseParseError,
    ServiceUnavailableError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionDeniedError,
    ToolTimeoutError,
    ValidationError,
    VotingFailedError,
)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_error_code_values(self):
        """Test that error codes have correct string values."""
        assert ErrorCode.ADAPTER_TIMEOUT.value == "ADAPTER_TIMEOUT"
        assert ErrorCode.RATE_LIMITED.value == "RATE_LIMITED"
        assert ErrorCode.MODEL_UNAVAILABLE.value == "MODEL_UNAVAILABLE"
        assert ErrorCode.INVALID_API_KEY.value == "INVALID_API_KEY"

    def test_error_code_is_string_enum(self):
        """Test that ErrorCode inherits from str."""
        assert isinstance(ErrorCode.ADAPTER_TIMEOUT, str)
        assert ErrorCode.ADAPTER_TIMEOUT == "ADAPTER_TIMEOUT"


class TestErrorDetails:
    """Tests for ErrorDetails dataclass."""

    def test_default_values(self):
        """Test that ErrorDetails has sensible defaults."""
        details = ErrorDetails()
        assert details.provider is None
        assert details.model is None
        assert details.original_error is None
        assert details.status_code is None
        assert details.retry_after is None
        assert details.additional == {}

    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        details = ErrorDetails(provider="openrouter", model="gpt-4")
        result = details.to_dict()
        assert result == {"provider": "openrouter", "model": "gpt-4"}
        assert "original_error" not in result
        assert "status_code" not in result

    def test_to_dict_includes_additional(self):
        """Test that to_dict includes additional info."""
        details = ErrorDetails(
            provider="openrouter",
            additional={"response_body": {"error": "test"}},
        )
        result = details.to_dict()
        assert result["additional"]["response_body"] == {"error": "test"}

    def test_full_details(self):
        """Test ErrorDetails with all fields populated."""
        details = ErrorDetails(
            provider="openrouter",
            model="anthropic/claude-3.5-sonnet",
            original_error="Connection refused",
            original_error_type="ConnectionError",
            status_code=503,
            retry_after=30,
            request_id="req-12345",
            additional={"attempt": 3},
        )
        result = details.to_dict()
        assert len(result) == 8
        assert result["provider"] == "openrouter"
        assert result["status_code"] == 503
        assert result["retry_after"] == 30


class TestAICouncilError:
    """Tests for base AICouncilError."""

    def test_basic_construction(self):
        """Test basic exception construction."""
        error = AICouncilError(
            error_code=ErrorCode.ADAPTER_TIMEOUT,
            message="Request timed out",
        )
        assert error.error_code == ErrorCode.ADAPTER_TIMEOUT
        assert error.message == "Request timed out"
        assert str(error) == "[ADAPTER_TIMEOUT] Request timed out"

    def test_with_details(self):
        """Test exception with details."""
        details = ErrorDetails(provider="ollama", model="llama2")
        error = AICouncilError(
            error_code=ErrorCode.MODEL_UNAVAILABLE,
            message="Model not found",
            details=details,
        )
        assert error.details.provider == "ollama"
        assert error.details.model == "llama2"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        details = ErrorDetails(provider="openrouter", status_code=429)
        error = AICouncilError(
            error_code=ErrorCode.RATE_LIMITED,
            message="Rate limit exceeded",
            details=details,
        )
        result = error.to_dict()
        assert result["error_code"] == "RATE_LIMITED"
        assert result["message"] == "Rate limit exceeded"
        assert result["details"]["provider"] == "openrouter"
        assert result["details"]["status_code"] == 429

    def test_inheritance(self):
        """Test that AICouncilError inherits from Exception."""
        error = AICouncilError(ErrorCode.UNKNOWN_ERROR, "Test")
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self):
        """Test that exceptions can be raised and caught."""
        with pytest.raises(AICouncilError) as exc_info:
            raise AICouncilError(ErrorCode.INTERNAL_ERROR, "Internal error")
        assert exc_info.value.error_code == ErrorCode.INTERNAL_ERROR


class TestAdapterExceptions:
    """Tests for adapter-related exceptions."""

    def test_adapter_timeout_error(self):
        """Test AdapterTimeoutError defaults."""
        error = AdapterTimeoutError()
        assert error.error_code == ErrorCode.ADAPTER_TIMEOUT
        assert "timed out" in error.message.lower()

    def test_adapter_timeout_error_custom_message(self):
        """Test AdapterTimeoutError with custom message."""
        error = AdapterTimeoutError(
            message="Claude request timed out after 120s",
            details=ErrorDetails(provider="claude"),
        )
        assert "120s" in error.message
        assert error.details.provider == "claude"

    def test_adapter_connection_error(self):
        """Test AdapterConnectionError."""
        error = AdapterConnectionError()
        assert error.error_code == ErrorCode.ADAPTER_CONNECTION_ERROR
        assert isinstance(error, AdapterError)

    def test_adapter_network_error(self):
        """Test AdapterNetworkError."""
        error = AdapterNetworkError()
        assert error.error_code == ErrorCode.ADAPTER_NETWORK_ERROR
        assert isinstance(error, AdapterError)

    def test_adapter_inheritance(self):
        """Test that adapter errors inherit from AdapterError."""
        errors = [
            AdapterTimeoutError(),
            AdapterConnectionError(),
            AdapterNetworkError(),
        ]
        for error in errors:
            assert isinstance(error, AdapterError)
            assert isinstance(error, AICouncilError)


class TestRateLimitingExceptions:
    """Tests for rate limiting and availability exceptions."""

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError(
            details=ErrorDetails(retry_after=30)
        )
        assert error.error_code == ErrorCode.RATE_LIMITED
        assert error.details.retry_after == 30

    def test_model_unavailable_error(self):
        """Test ModelUnavailableError."""
        error = ModelUnavailableError(
            message="Model anthropic/claude-3.5-sonnet is unavailable",
            details=ErrorDetails(model="anthropic/claude-3.5-sonnet"),
        )
        assert error.error_code == ErrorCode.MODEL_UNAVAILABLE
        assert "claude" in error.message.lower()

    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError."""
        error = ServiceUnavailableError()
        assert error.error_code == ErrorCode.SERVICE_UNAVAILABLE


class TestAuthenticationExceptions:
    """Tests for authentication exceptions."""

    def test_invalid_api_key_error(self):
        """Test InvalidAPIKeyError."""
        error = InvalidAPIKeyError()
        assert error.error_code == ErrorCode.INVALID_API_KEY
        assert isinstance(error, AuthenticationError)

    def test_permission_denied_error(self):
        """Test PermissionDeniedError."""
        error = PermissionDeniedError()
        assert error.error_code == ErrorCode.PERMISSION_DENIED

    def test_quota_exceeded_error(self):
        """Test QuotaExceededError."""
        error = QuotaExceededError(
            message="Monthly quota exceeded",
        )
        assert error.error_code == ErrorCode.QUOTA_EXCEEDED
        assert "quota" in error.message.lower()


class TestResponseExceptions:
    """Tests for response-related exceptions."""

    def test_invalid_response_error(self):
        """Test InvalidResponseError."""
        error = InvalidResponseError()
        assert error.error_code == ErrorCode.INVALID_RESPONSE
        assert isinstance(error, ResponseError)

    def test_response_parse_error(self):
        """Test ResponseParseError."""
        error = ResponseParseError(
            message="Failed to parse JSON response",
        )
        assert error.error_code == ErrorCode.RESPONSE_PARSE_ERROR

    def test_empty_response_error(self):
        """Test EmptyResponseError."""
        error = EmptyResponseError()
        assert error.error_code == ErrorCode.EMPTY_RESPONSE

    def test_content_filtered_error(self):
        """Test ContentFilteredError."""
        error = ContentFilteredError(
            message="Response blocked by safety filter",
        )
        assert error.error_code == ErrorCode.CONTENT_FILTERED


class TestValidationExceptions:
    """Tests for validation exceptions."""

    def test_prompt_too_long_error(self):
        """Test PromptTooLongError."""
        error = PromptTooLongError(
            message="Prompt exceeds 100,000 character limit",
        )
        assert error.error_code == ErrorCode.PROMPT_TOO_LONG
        assert isinstance(error, ValidationError)

    def test_invalid_model_error(self):
        """Test InvalidModelError."""
        error = InvalidModelError(
            message="Model 'invalid-model' not found",
        )
        assert error.error_code == ErrorCode.INVALID_MODEL

    def test_invalid_parameter_error(self):
        """Test InvalidParameterError."""
        error = InvalidParameterError()
        assert error.error_code == ErrorCode.INVALID_PARAMETER

    def test_missing_required_field_error(self):
        """Test MissingRequiredFieldError."""
        error = MissingRequiredFieldError(
            message="Field 'working_directory' is required",
        )
        assert error.error_code == ErrorCode.MISSING_REQUIRED_FIELD


class TestConfigurationExceptions:
    """Tests for configuration exceptions."""

    def test_missing_config_error(self):
        """Test MissingConfigError."""
        error = MissingConfigError()
        assert error.error_code == ErrorCode.MISSING_CONFIG
        assert isinstance(error, ConfigurationError)

    def test_invalid_config_error(self):
        """Test InvalidConfigError."""
        error = InvalidConfigError(
            message="Invalid adapter configuration",
        )
        assert error.error_code == ErrorCode.INVALID_CONFIG


class TestDeliberationExceptions:
    """Tests for deliberation exceptions."""

    def test_deliberation_failed_error(self):
        """Test DeliberationFailedError."""
        error = DeliberationFailedError()
        assert error.error_code == ErrorCode.DELIBERATION_FAILED
        assert isinstance(error, DeliberationError)

    def test_convergence_failed_error(self):
        """Test ConvergenceFailedError."""
        error = ConvergenceFailedError()
        assert error.error_code == ErrorCode.CONVERGENCE_FAILED

    def test_voting_failed_error(self):
        """Test VotingFailedError."""
        error = VotingFailedError()
        assert error.error_code == ErrorCode.VOTING_FAILED

    def test_all_participants_failed_error(self):
        """Test AllParticipantsFailedError."""
        error = AllParticipantsFailedError(
            message="All 3 participants failed in round 2",
        )
        assert error.error_code == ErrorCode.ALL_PARTICIPANTS_FAILED


class TestToolExceptions:
    """Tests for tool execution exceptions."""

    def test_tool_execution_error(self):
        """Test ToolExecutionError."""
        error = ToolExecutionError()
        assert error.error_code == ErrorCode.TOOL_EXECUTION_ERROR
        assert isinstance(error, ToolError)

    def test_tool_timeout_error(self):
        """Test ToolTimeoutError."""
        error = ToolTimeoutError(
            message="read_file timed out after 10s",
        )
        assert error.error_code == ErrorCode.TOOL_TIMEOUT

    def test_tool_not_found_error(self):
        """Test ToolNotFoundError."""
        error = ToolNotFoundError(
            message="Tool 'unknown_tool' not found",
        )
        assert error.error_code == ErrorCode.TOOL_NOT_FOUND

    def test_tool_permission_denied_error(self):
        """Test ToolPermissionDeniedError."""
        error = ToolPermissionDeniedError()
        assert error.error_code == ErrorCode.TOOL_PERMISSION_DENIED


class TestExceptionHierarchy:
    """Tests for overall exception hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from AICouncilError."""
        exception_classes = [
            AdapterTimeoutError,
            AdapterConnectionError,
            AdapterNetworkError,
            RateLimitError,
            ModelUnavailableError,
            ServiceUnavailableError,
            InvalidAPIKeyError,
            PermissionDeniedError,
            QuotaExceededError,
            InvalidResponseError,
            ResponseParseError,
            EmptyResponseError,
            ContentFilteredError,
            PromptTooLongError,
            InvalidModelError,
            InvalidParameterError,
            MissingRequiredFieldError,
            MissingConfigError,
            InvalidConfigError,
            DeliberationFailedError,
            ConvergenceFailedError,
            VotingFailedError,
            AllParticipantsFailedError,
            ToolExecutionError,
            ToolTimeoutError,
            ToolNotFoundError,
            ToolPermissionDeniedError,
        ]
        for cls in exception_classes:
            error = cls()
            assert isinstance(error, AICouncilError), f"{cls.__name__} should inherit from AICouncilError"
            assert isinstance(error, Exception), f"{cls.__name__} should inherit from Exception"

    def test_exception_can_be_caught_by_category(self):
        """Test that exceptions can be caught by category."""
        # Catch by adapter category
        with pytest.raises(AdapterError):
            raise AdapterTimeoutError()

        # Catch by response category
        with pytest.raises(ResponseError):
            raise InvalidResponseError()

        # Catch by validation category
        with pytest.raises(ValidationError):
            raise PromptTooLongError()

        # Catch by configuration category
        with pytest.raises(ConfigurationError):
            raise InvalidConfigError()

        # Catch by deliberation category
        with pytest.raises(DeliberationError):
            raise VotingFailedError()

        # Catch by tool category
        with pytest.raises(ToolError):
            raise ToolTimeoutError()

    def test_exception_can_be_caught_by_base(self):
        """Test that all exceptions can be caught by AICouncilError."""
        exceptions_to_raise = [
            AdapterTimeoutError(),
            RateLimitError(),
            InvalidAPIKeyError(),
            InvalidResponseError(),
            PromptTooLongError(),
            InvalidConfigError(),
            DeliberationFailedError(),
            ToolExecutionError(),
        ]
        for exc in exceptions_to_raise:
            with pytest.raises(AICouncilError):
                raise exc
