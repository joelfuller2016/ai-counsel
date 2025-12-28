"""Structured JSON logging with correlation IDs for AI Counsel.

This module provides:
- JSONFormatter for structured log output
- Correlation ID generation and management
- Thread-safe correlation context
- Configurable log format (json/text) via config.yaml

Usage:
    from utils.logging import configure_logging, generate_correlation_id, set_correlation_id

    # Configure logging at startup
    configure_logging(format="json", level="INFO")

    # Generate and set correlation ID for a deliberation
    correlation_id = generate_correlation_id()
    set_correlation_id(correlation_id)

    # Logs will now include the correlation ID
    logger.info("Starting deliberation", extra={"question": "What color?"})
"""

import contextvars
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Thread-local storage for correlation ID using contextvars (async-safe)
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for tracing.

    Returns:
        A UUID4-based correlation ID string.

    Example:
        correlation_id = generate_correlation_id()
        # Returns: "d7e8f9a0-1234-5678-9abc-def012345678"
    """
    return str(uuid.uuid4())


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context.

    Returns:
        The current correlation ID, or None if not set.
    """
    return _correlation_id.get()


def set_correlation_id(correlation_id: Optional[str]) -> None:
    """Set the correlation ID for the current context.

    Args:
        correlation_id: The correlation ID to set, or None to clear.
    """
    _correlation_id.set(correlation_id)


class CorrelationContext:
    """Context manager for correlation ID scope.

    Usage:
        with CorrelationContext(generate_correlation_id()) as correlation_id:
            logger.info("Inside correlation scope")
            # All logs in this scope will include the correlation ID
    """

    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize correlation context.

        Args:
            correlation_id: The correlation ID to use, or None to auto-generate.
        """
        self.correlation_id = correlation_id or generate_correlation_id()
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> str:
        """Enter the correlation context."""
        self._token = _correlation_id.set(self.correlation_id)
        return self.correlation_id

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the correlation context and restore previous value."""
        if self._token is not None:
            _correlation_id.reset(self._token)


class JSONFormatter(logging.Formatter):
    """JSON log formatter with correlation ID support.

    Produces structured JSON log output with:
    - timestamp: ISO 8601 format with timezone
    - level: Log level name
    - logger: Logger name
    - message: Log message
    - correlation_id: Current correlation ID (if set)
    - Additional fields from record.extra

    Example output:
    {
        "timestamp": "2025-12-27T10:30:00.123456+00:00",
        "level": "INFO",
        "logger": "deliberation.engine",
        "message": "Starting deliberation",
        "correlation_id": "d7e8f9a0-1234-5678-9abc-def012345678",
        "question": "What color should we use?",
        "participants": 3
    }
    """

    # Fields that are standard logging fields (not to be included in extra)
    RESERVED_ATTRS = frozenset(
        [
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        ]
    )

    def __init__(
        self,
        include_extra: bool = True,
        include_exception: bool = True,
        include_location: bool = False,
    ):
        """Initialize JSONFormatter.

        Args:
            include_extra: Include extra fields from log record (default: True)
            include_exception: Include exception info if present (default: True)
            include_location: Include file/line/function info (default: False)
        """
        super().__init__()
        self.include_extra = include_extra
        self.include_exception = include_exception
        self.include_location = include_location

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log string.
        """
        # Build base log entry
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if present
        correlation_id = get_correlation_id()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Add location info if requested
        if self.include_location:
            log_entry["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add extra fields from record
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                    try:
                        # Attempt to serialize the value
                        json.dumps(value)
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        # If not JSON-serializable, convert to string
                        log_entry[key] = str(value)

        # Add exception info if present
        if self.include_exception and record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Text log formatter with correlation ID support.

    Produces human-readable log output with optional correlation ID.

    Example output:
    2025-12-27 10:30:00,123 | INFO | [d7e8f9a0] | deliberation.engine | Starting deliberation
    """

    def __init__(self, include_correlation: bool = True):
        """Initialize TextFormatter.

        Args:
            include_correlation: Include correlation ID in output (default: True)
        """
        super().__init__()
        self.include_correlation = include_correlation

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text string.

        Args:
            record: The log record to format.

        Returns:
            Text-formatted log string.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

        # Get correlation ID (truncated for readability)
        correlation_part = ""
        if self.include_correlation:
            correlation_id = get_correlation_id()
            if correlation_id:
                # Show first 8 chars of correlation ID
                correlation_part = f" | [{correlation_id[:8]}]"

        # Build the log line
        base_msg = f"{timestamp} | {record.levelname:8} |{correlation_part} {record.name} | {record.getMessage()}"

        # Add exception info if present
        if record.exc_info:
            base_msg += f"\n{self.formatException(record.exc_info)}"

        return base_msg


def configure_logging(
    format: str = "text",
    level: str = "INFO",
    log_file: Optional[str] = None,
    include_location: bool = False,
) -> None:
    """Configure logging for AI Counsel.

    Args:
        format: Log format - "json" or "text" (default: "text")
        level: Log level - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: "INFO")
        log_file: Optional file path for logging (in addition to console)
        include_location: Include file/line/function in logs (default: False)

    Example:
        # JSON logging for production
        configure_logging(format="json", level="INFO", log_file="app.log")

        # Text logging for development
        configure_logging(format="text", level="DEBUG")
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter based on format type
    if format.lower() == "json":
        formatter = JSONFormatter(include_location=include_location)
    else:
        formatter = TextFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger_with_context(
    name: str,
    correlation_id: Optional[str] = None,
) -> logging.LoggerAdapter:
    """Get a logger adapter with correlation context.

    Args:
        name: Logger name
        correlation_id: Optional correlation ID to include in all logs

    Returns:
        LoggerAdapter with correlation context

    Example:
        logger = get_logger_with_context("deliberation.engine", correlation_id)
        logger.info("Starting round", extra={"round": 1})
    """
    base_logger = logging.getLogger(name)

    class CorrelationAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            # Ensure extra dict exists
            extra = kwargs.get("extra", {})

            # Add correlation ID if set
            cid = correlation_id or get_correlation_id()
            if cid:
                extra["correlation_id"] = cid

            kwargs["extra"] = extra
            return msg, kwargs

    return CorrelationAdapter(base_logger, {})
