"""Tests for structured JSON logging with correlation IDs."""

import json
import logging
import pytest
from io import StringIO

from utils.logging import (
    CorrelationContext,
    JSONFormatter,
    TextFormatter,
    configure_logging,
    generate_correlation_id,
    get_correlation_id,
    get_logger_with_context,
    set_correlation_id,
)


class TestCorrelationId:
    """Tests for correlation ID generation and management."""

    def test_generate_correlation_id_format(self):
        """Test that generated correlation IDs are valid UUIDs."""
        correlation_id = generate_correlation_id()
        # UUID4 format: 8-4-4-4-12 = 36 characters
        assert len(correlation_id) == 36
        parts = correlation_id.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

    def test_generate_correlation_id_unique(self):
        """Test that each generated ID is unique."""
        ids = [generate_correlation_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_get_set_correlation_id(self):
        """Test getting and setting correlation ID."""
        # Initially should be None
        original = get_correlation_id()

        # Set a value
        test_id = "test-correlation-id"
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id

        # Clear it
        set_correlation_id(None)
        assert get_correlation_id() is None

        # Restore original state
        set_correlation_id(original)

    def test_correlation_context_manager(self):
        """Test CorrelationContext context manager."""
        original = get_correlation_id()

        with CorrelationContext("test-context-id") as cid:
            assert cid == "test-context-id"
            assert get_correlation_id() == "test-context-id"

        # After context, should be restored
        assert get_correlation_id() == original

    def test_correlation_context_auto_generate(self):
        """Test CorrelationContext auto-generates ID if not provided."""
        original = get_correlation_id()

        with CorrelationContext() as cid:
            assert cid is not None
            assert len(cid) == 36  # UUID format
            assert get_correlation_id() == cid

        assert get_correlation_id() == original

    def test_nested_correlation_contexts(self):
        """Test that nested contexts restore correctly."""
        original = get_correlation_id()

        with CorrelationContext("outer") as outer_id:
            assert outer_id == "outer"
            assert get_correlation_id() == "outer"

            with CorrelationContext("inner") as inner_id:
                assert inner_id == "inner"
                assert get_correlation_id() == "inner"

            # Inner context exited, should be back to outer
            assert get_correlation_id() == "outer"

        # All contexts exited
        assert get_correlation_id() == original


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test_json_formatter")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.formatter = JSONFormatter()
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

        # Clear correlation ID
        set_correlation_id(None)

    def teardown_method(self):
        """Clean up after tests."""
        self.logger.handlers.clear()
        set_correlation_id(None)

    def test_basic_json_format(self):
        """Test that output is valid JSON."""
        self.logger.info("Test message")
        output = self.stream.getvalue().strip()

        log_data = json.loads(output)
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Test message"
        assert log_data["logger"] == "test_json_formatter"
        assert "timestamp" in log_data

    def test_includes_correlation_id(self):
        """Test that correlation ID is included when set."""
        set_correlation_id("test-corr-id")

        self.logger.info("Message with correlation")
        output = self.stream.getvalue().strip()

        log_data = json.loads(output)
        assert log_data["correlation_id"] == "test-corr-id"

    def test_excludes_correlation_id_when_not_set(self):
        """Test that correlation ID is not included when not set."""
        self.logger.info("Message without correlation")
        output = self.stream.getvalue().strip()

        log_data = json.loads(output)
        assert "correlation_id" not in log_data

    def test_includes_extra_fields(self):
        """Test that extra fields are included."""
        self.logger.info(
            "Processing request",
            extra={"model": "gpt-4", "round": 2, "participant_count": 3},
        )
        output = self.stream.getvalue().strip()

        log_data = json.loads(output)
        assert log_data["model"] == "gpt-4"
        assert log_data["round"] == 2
        assert log_data["participant_count"] == 3

    def test_all_log_levels(self):
        """Test all log levels produce valid JSON."""
        levels = [
            (self.logger.debug, "DEBUG"),
            (self.logger.info, "INFO"),
            (self.logger.warning, "WARNING"),
            (self.logger.error, "ERROR"),
            (self.logger.critical, "CRITICAL"),
        ]

        for log_method, level_name in levels:
            self.stream.truncate(0)
            self.stream.seek(0)

            log_method(f"Test {level_name}")
            output = self.stream.getvalue().strip()

            log_data = json.loads(output)
            assert log_data["level"] == level_name

    def test_timestamp_format(self):
        """Test that timestamp is in ISO format."""
        self.logger.info("Test")
        output = self.stream.getvalue().strip()

        log_data = json.loads(output)
        timestamp = log_data["timestamp"]

        # Should be ISO 8601 format
        assert "T" in timestamp
        assert ":" in timestamp
        # Should have timezone info
        assert "+" in timestamp or "Z" in timestamp

    def test_exception_formatting(self):
        """Test that exceptions are included."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            self.logger.exception("Error occurred")

        output = self.stream.getvalue().strip()
        log_data = json.loads(output)

        assert "exception" in log_data
        assert "ValueError" in log_data["exception"]
        assert "Test exception" in log_data["exception"]

    def test_include_location_option(self):
        """Test include_location option."""
        formatter_with_location = JSONFormatter(include_location=True)
        handler = logging.StreamHandler(StringIO())
        handler.setFormatter(formatter_with_location)

        logger = logging.getLogger("test_location")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("Test")
        output = handler.stream.getvalue().strip()

        log_data = json.loads(output)
        assert "location" in log_data
        assert "file" in log_data["location"]
        assert "line" in log_data["location"]
        assert "function" in log_data["location"]

    def test_non_serializable_extra_converted(self):
        """Test that non-serializable extra values are converted to strings."""

        class CustomObject:
            def __str__(self):
                return "custom_object_str"

        self.logger.info("Test", extra={"custom": CustomObject()})
        output = self.stream.getvalue().strip()

        log_data = json.loads(output)
        assert log_data["custom"] == "custom_object_str"


class TestTextFormatter:
    """Tests for TextFormatter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test_text_formatter")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.formatter = TextFormatter()
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

        set_correlation_id(None)

    def teardown_method(self):
        """Clean up after tests."""
        self.logger.handlers.clear()
        set_correlation_id(None)

    def test_basic_text_format(self):
        """Test basic text output format."""
        self.logger.info("Test message")
        output = self.stream.getvalue().strip()

        # Should contain timestamp, level, logger name, message
        assert "INFO" in output
        assert "test_text_formatter" in output
        assert "Test message" in output

    def test_includes_correlation_id(self):
        """Test that correlation ID is included when set."""
        set_correlation_id("abc12345-test-correlation")

        self.logger.info("Message")
        output = self.stream.getvalue().strip()

        # Should include first 8 chars of correlation ID
        assert "[abc12345]" in output

    def test_excludes_correlation_when_not_set(self):
        """Test that no correlation section when not set."""
        self.logger.info("Message")
        output = self.stream.getvalue().strip()

        # Should not have empty brackets
        assert "[]" not in output

    def test_disable_correlation_option(self):
        """Test include_correlation=False option."""
        formatter = TextFormatter(include_correlation=False)
        handler = logging.StreamHandler(StringIO())
        handler.setFormatter(formatter)

        logger = logging.getLogger("test_no_corr")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        set_correlation_id("test-id")
        logger.info("Test")
        output = handler.stream.getvalue().strip()

        assert "[test-id]" not in output

    def test_exception_formatting(self):
        """Test that exceptions are appended."""
        try:
            raise RuntimeError("Test error")
        except RuntimeError:
            self.logger.exception("Error occurred")

        output = self.stream.getvalue()
        assert "RuntimeError" in output
        assert "Test error" in output


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def teardown_method(self):
        """Clean up after tests."""
        # Reset root logger
        root = logging.getLogger()
        root.handlers.clear()
        set_correlation_id(None)

    def test_json_format_configuration(self):
        """Test configuring JSON format."""
        configure_logging(format="json", level="DEBUG")

        root = logging.getLogger()
        assert len(root.handlers) >= 1

        # Check that handler uses JSONFormatter
        handler = root.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_text_format_configuration(self):
        """Test configuring text format."""
        configure_logging(format="text", level="INFO")

        root = logging.getLogger()
        assert len(root.handlers) >= 1

        handler = root.handlers[0]
        assert isinstance(handler.formatter, TextFormatter)

    def test_level_configuration(self):
        """Test log level configuration."""
        configure_logging(format="text", level="WARNING")

        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_case_insensitive_level(self):
        """Test that level is case-insensitive."""
        configure_logging(format="text", level="debug")
        assert logging.getLogger().level == logging.DEBUG

        configure_logging(format="text", level="ERROR")
        assert logging.getLogger().level == logging.ERROR


class TestLoggerWithContext:
    """Tests for get_logger_with_context function."""

    def setup_method(self):
        """Set up test fixtures."""
        set_correlation_id(None)

    def teardown_method(self):
        """Clean up after tests."""
        set_correlation_id(None)

    def test_logger_includes_correlation(self):
        """Test that logger adapter includes correlation ID."""
        set_correlation_id("context-test-id")

        logger = get_logger_with_context("test_logger")

        # Set up a handler to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.logger.handlers.clear()
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.DEBUG)

        logger.info("Test message")
        output = stream.getvalue().strip()

        log_data = json.loads(output)
        assert log_data["correlation_id"] == "context-test-id"

    def test_logger_with_explicit_correlation(self):
        """Test logger with explicitly passed correlation ID."""
        logger = get_logger_with_context("test_logger", correlation_id="explicit-id")

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.logger.handlers.clear()
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.DEBUG)

        logger.info("Test message")
        output = stream.getvalue().strip()

        log_data = json.loads(output)
        assert log_data["correlation_id"] == "explicit-id"


class TestIntegration:
    """Integration tests for logging system."""

    def setup_method(self):
        """Set up test fixtures."""
        set_correlation_id(None)
        logging.getLogger().handlers.clear()

    def teardown_method(self):
        """Clean up after tests."""
        set_correlation_id(None)
        logging.getLogger().handlers.clear()

    def test_full_logging_workflow(self):
        """Test complete logging workflow with correlation ID."""
        # Configure logging
        configure_logging(format="json", level="DEBUG")

        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(handler)

        # Simulate deliberation with correlation context
        with CorrelationContext() as correlation_id:
            logger = logging.getLogger("deliberation.engine")

            logger.info(
                "Starting deliberation",
                extra={"question": "What color?", "participants": 3},
            )
            logger.debug("Round 1 started", extra={"round": 1})
            logger.info("Deliberation complete", extra={"result": "blue"})

        # Parse and verify logs
        lines = stream.getvalue().strip().split("\n")
        assert len(lines) == 3

        for line in lines:
            log_data = json.loads(line)
            assert log_data["correlation_id"] == correlation_id
            assert log_data["logger"] == "deliberation.engine"

        # Verify specific log contents
        start_log = json.loads(lines[0])
        assert start_log["message"] == "Starting deliberation"
        assert start_log["question"] == "What color?"
        assert start_log["participants"] == 3

    def test_correlation_id_persists_across_logs(self):
        """Test that same correlation ID appears in all logs within context."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())

        logger = logging.getLogger("test_persist")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        with CorrelationContext("persist-test-123") as _:
            logger.info("First")
            logger.info("Second")
            logger.info("Third")

        lines = stream.getvalue().strip().split("\n")
        correlation_ids = [json.loads(line)["correlation_id"] for line in lines]

        assert all(cid == "persist-test-123" for cid in correlation_ids)
