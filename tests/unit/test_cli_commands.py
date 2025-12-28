"""Unit tests for cli/commands.py CLI commands.

This module tests all CLI commands for configuration validation and diagnostics:
- 'validate-config': Validate config.yaml syntax and model availability
- 'dry-run': Simulate deliberation without calling models
- 'list-models': List all configured models with status

Tests cover:
- YAML syntax validation
- Pydantic schema validation
- Adapter availability checking
- Model registry validation
- Token estimation
- Participant parsing
- Output formatting (text, json, table)
- Error handling and edge cases
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from cli.commands import (
    ConfigValidationResult,
    cli,
    dry_run,
    estimate_token_usage,
    list_models,
    parse_participants,
    perform_full_validation,
    validate_config,
    validate_config_schema,
    validate_yaml_syntax,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def valid_config_data():
    """Minimal valid configuration data."""
    return {
        "version": "1.0",
        "cli_tools": {
            "claude": {
                "command": "claude",
                "args": ["-p", "--model", "{model}", "{prompt}"],
                "timeout": 60,
            },
            "codex": {
                "command": "codex",
                "args": ["exec", "--model", "{model}", "{prompt}"],
                "timeout": 60,
            },
        },
        "defaults": {
            "mode": "quick",
            "rounds": 2,
            "max_rounds": 5,
            "timeout_per_round": 120,
        },
        "model_registry": {
            "claude": [
                {
                    "id": "claude-sonnet-4-5-20250929",
                    "label": "Sonnet 4.5",
                    "tier": "balanced",
                    "default": True,
                    "enabled": True,
                },
                {
                    "id": "claude-opus-4-5-20251101",
                    "label": "Opus 4.5",
                    "tier": "premium",
                    "enabled": True,
                },
            ],
            "codex": [
                {
                    "id": "gpt-5.2-codex",
                    "label": "GPT-5.2 Codex",
                    "tier": "flagship",
                    "default": True,
                    "enabled": True,
                },
                {
                    "id": "gpt-5.1-codex-mini",
                    "label": "Mini",
                    "tier": "speed",
                    "enabled": False,
                },
            ],
        },
        "storage": {
            "transcripts_dir": "transcripts",
            "format": "markdown",
            "auto_export": True,
        },
        "deliberation": {
            "convergence_detection": {
                "enabled": True,
                "semantic_similarity_threshold": 0.85,
                "divergence_threshold": 0.40,
                "min_rounds_before_check": 1,
                "consecutive_stable_rounds": 2,
                "stance_stability_threshold": 0.80,
                "response_length_drop_threshold": 0.40,
            },
            "early_stopping": {
                "enabled": True,
                "threshold": 0.66,
                "respect_min_rounds": True,
            },
            "convergence_threshold": 0.8,
            "enable_convergence_detection": True,
        },
    }


@pytest.fixture
def temp_config_file(valid_config_data):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(valid_config_data, f)
        return Path(f.name)


# ============================================================================
# TEST: ConfigValidationResult
# ============================================================================


class TestConfigValidationResult:
    """Tests for ConfigValidationResult class."""

    def test_valid_when_no_errors(self):
        """Test that result is valid when no errors."""
        result = ConfigValidationResult()
        assert result.valid is True

    def test_invalid_when_has_errors(self):
        """Test that result is invalid when has errors."""
        result = ConfigValidationResult()
        result.add_error("Some error")
        assert result.valid is False

    def test_warnings_dont_affect_validity(self):
        """Test that warnings don't affect validity."""
        result = ConfigValidationResult()
        result.add_warning("Some warning")
        assert result.valid is True

    def test_info_messages_stored(self):
        """Test that info messages are stored."""
        result = ConfigValidationResult()
        result.add_info("Some info")
        assert "Some info" in result.info

    def test_to_dict_includes_all_fields(self):
        """Test to_dict includes all fields."""
        result = ConfigValidationResult()
        result.add_error("Error 1")
        result.add_warning("Warning 1")
        result.add_info("Info 1")

        d = result.to_dict()
        assert d["valid"] is False
        assert d["error_count"] == 1
        assert d["warning_count"] == 1
        assert "Error 1" in d["errors"]
        assert "Warning 1" in d["warnings"]
        assert "Info 1" in d["info"]


# ============================================================================
# TEST: validate_yaml_syntax
# ============================================================================


class TestValidateYamlSyntax:
    """Tests for YAML syntax validation."""

    def test_valid_yaml_returns_true(self, temp_config_file):
        """Test valid YAML file returns True."""
        is_valid, error, data = validate_yaml_syntax(temp_config_file)
        assert is_valid is True
        assert error is None
        assert data is not None
        assert "version" in data

    def test_invalid_yaml_syntax_returns_false(self):
        """Test invalid YAML syntax returns False with error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: syntax:")
            temp_path = Path(f.name)

        is_valid, error, data = validate_yaml_syntax(temp_path)
        assert is_valid is False
        assert error is not None
        assert "YAML syntax error" in error
        assert data is None

        temp_path.unlink()

    def test_missing_file_returns_false(self):
        """Test missing file returns False with error."""
        is_valid, error, data = validate_yaml_syntax(Path("/nonexistent/config.yaml"))
        assert is_valid is False
        assert error is not None
        assert "not found" in error
        assert data is None


# ============================================================================
# TEST: validate_config_schema
# ============================================================================


class TestValidateConfigSchema:
    """Tests for Pydantic schema validation."""

    def test_valid_config_returns_true(self, temp_config_file):
        """Test valid config returns True with config object."""
        is_valid, error, config = validate_config_schema(temp_config_file)
        assert is_valid is True
        assert error is None
        assert config is not None
        assert config.version == "1.0"

    def test_invalid_schema_returns_false(self):
        """Test invalid schema returns False with error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Missing required fields
            yaml.dump({"version": "1.0"}, f)
            temp_path = Path(f.name)

        is_valid, error, config = validate_config_schema(temp_path)
        assert is_valid is False
        assert error is not None
        assert "validation error" in error.lower() or "adapters" in error.lower()
        assert config is None

        temp_path.unlink()


# ============================================================================
# TEST: estimate_token_usage
# ============================================================================


class TestEstimateTokenUsage:
    """Tests for token usage estimation."""

    def test_estimate_returns_expected_fields(self):
        """Test estimate returns all expected fields."""
        estimate = estimate_token_usage(
            question="What is the best approach?",
            context="Some context here",
            rounds=2,
            participant_count=3,
        )

        assert "question_tokens" in estimate
        assert "context_tokens" in estimate
        assert "total_estimated_input_tokens" in estimate
        assert "total_estimated_output_tokens" in estimate
        assert "total_estimated_tokens" in estimate
        assert "note" in estimate

    def test_estimate_scales_with_rounds(self):
        """Test estimate scales with number of rounds."""
        estimate_2 = estimate_token_usage("Q", None, rounds=2, participant_count=2)
        estimate_4 = estimate_token_usage("Q", None, rounds=4, participant_count=2)

        # More rounds = more tokens
        assert (
            estimate_4["total_estimated_tokens"] > estimate_2["total_estimated_tokens"]
        )

    def test_estimate_scales_with_participants(self):
        """Test estimate scales with number of participants."""
        estimate_2 = estimate_token_usage("Q", None, rounds=2, participant_count=2)
        estimate_4 = estimate_token_usage("Q", None, rounds=2, participant_count=4)

        # More participants = more tokens
        assert (
            estimate_4["total_estimated_tokens"] > estimate_2["total_estimated_tokens"]
        )

    def test_estimate_includes_context(self):
        """Test estimate includes context tokens."""
        estimate_no_ctx = estimate_token_usage("Q", None, rounds=2, participant_count=2)
        estimate_with_ctx = estimate_token_usage(
            "Q", "A" * 1000, rounds=2, participant_count=2
        )

        assert estimate_with_ctx["context_tokens"] > 0
        assert estimate_no_ctx["context_tokens"] == 0


# ============================================================================
# TEST: parse_participants
# ============================================================================


class TestParseParticipants:
    """Tests for participant string parsing."""

    def test_parse_adapter_model_format(self, temp_config_file):
        """Test parsing 'adapter:model' format."""
        _, _, config = validate_config_schema(temp_config_file)

        participants = parse_participants(
            "claude:claude-sonnet-4-5-20250929,codex:gpt-5.2-codex",
            config,
        )

        assert len(participants) == 2
        assert participants[0]["cli"] == "claude"
        assert participants[0]["model"] == "claude-sonnet-4-5-20250929"
        assert participants[1]["cli"] == "codex"
        assert participants[1]["model"] == "gpt-5.2-codex"

    def test_parse_adapter_only_uses_default(self, temp_config_file):
        """Test parsing 'adapter' only uses default model."""
        _, _, config = validate_config_schema(temp_config_file)

        participants = parse_participants("claude,codex", config)

        assert len(participants) == 2
        assert participants[0]["cli"] == "claude"
        assert participants[0]["model"] == "claude-sonnet-4-5-20250929"  # default
        assert participants[1]["cli"] == "codex"
        assert participants[1]["model"] == "gpt-5.2-codex"  # default

    def test_parse_handles_whitespace(self, temp_config_file):
        """Test parsing handles whitespace."""
        _, _, config = validate_config_schema(temp_config_file)

        participants = parse_participants(" claude , codex ", config)

        assert len(participants) == 2
        assert participants[0]["cli"] == "claude"
        assert participants[1]["cli"] == "codex"

    def test_parse_ignores_empty_parts(self, temp_config_file):
        """Test parsing ignores empty parts."""
        _, _, config = validate_config_schema(temp_config_file)

        participants = parse_participants("claude,,codex,", config)

        assert len(participants) == 2


# ============================================================================
# TEST: validate-config command
# ============================================================================


class TestValidateConfigCommand:
    """Tests for 'validate-config' CLI command."""

    def test_valid_config_exits_zero(self, cli_runner, temp_config_file):
        """Test valid config exits with code 0."""
        with patch("cli.commands.DEFAULT_CONFIG_PATH", temp_config_file):
            result = cli_runner.invoke(validate_config, [])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_valid_config_shows_success(self, cli_runner, temp_config_file):
        """Test valid config shows success message."""
        with patch("cli.commands.DEFAULT_CONFIG_PATH", temp_config_file):
            result = cli_runner.invoke(validate_config, [])

        assert "Configuration is valid" in result.output

    def test_custom_config_path(self, cli_runner, temp_config_file):
        """Test custom config path is used."""
        result = cli_runner.invoke(
            validate_config,
            ["--config", str(temp_config_file)],
        )

        assert result.exit_code == 0

    def test_json_output_format(self, cli_runner, temp_config_file):
        """Test JSON output format."""
        with patch("cli.commands.DEFAULT_CONFIG_PATH", temp_config_file):
            result = cli_runner.invoke(validate_config, ["--format", "json"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert "valid" in output
        assert "errors" in output
        assert "warnings" in output

    def test_invalid_yaml_exits_nonzero(self, cli_runner):
        """Test invalid YAML exits with non-zero code."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: syntax:")
            temp_path = f.name

        result = cli_runner.invoke(validate_config, ["--config", temp_path])

        assert result.exit_code == 1
        Path(temp_path).unlink()

    def test_missing_config_exits_nonzero(self, cli_runner):
        """Test missing config file exits with non-zero code."""
        result = cli_runner.invoke(
            validate_config,
            ["--config", "/nonexistent/config.yaml"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_displays_adapter_info(self, cli_runner, temp_config_file):
        """Test that adapter info is displayed."""
        with patch("cli.commands.DEFAULT_CONFIG_PATH", temp_config_file):
            with patch("cli.commands.check_adapter_availability") as mock_check:
                mock_check.return_value = (True, "found in PATH")
                result = cli_runner.invoke(validate_config, [])

        # Should show adapter info
        assert result.exit_code == 0


# ============================================================================
# TEST: dry-run command
# ============================================================================


class TestDryRunCommand:
    """Tests for 'dry-run' CLI command."""

    def test_dry_run_requires_question(self, cli_runner):
        """Test dry-run requires --question."""
        result = cli_runner.invoke(dry_run, ["--participants", "claude,codex"])

        assert result.exit_code != 0
        assert "Missing option '--question'" in result.output

    def test_dry_run_requires_participants(self, cli_runner):
        """Test dry-run requires --participants."""
        result = cli_runner.invoke(dry_run, ["--question", "Test question"])

        assert result.exit_code != 0
        assert "Missing option '--participants'" in result.output

    def test_dry_run_text_output(self, cli_runner, temp_config_file):
        """Test dry-run text output format."""
        result = cli_runner.invoke(
            dry_run,
            [
                "--question",
                "Should we use TypeScript?",
                "--participants",
                "claude,codex",
                "--config",
                str(temp_config_file),
            ],
        )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "No API calls will be made" in result.output
        assert "TypeScript" in result.output
        assert "claude" in result.output.lower()
        assert "codex" in result.output.lower()

    def test_dry_run_json_output(self, cli_runner, temp_config_file):
        """Test dry-run JSON output format."""
        result = cli_runner.invoke(
            dry_run,
            [
                "--question",
                "Test question",
                "--participants",
                "claude,codex",
                "--format",
                "json",
                "--config",
                str(temp_config_file),
            ],
        )

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["dry_run"] is True
        assert output["would_execute"] is False
        assert "models_to_invoke" in output
        assert "token_estimate" in output

    def test_dry_run_shows_token_estimate(self, cli_runner, temp_config_file):
        """Test dry-run shows token estimate."""
        result = cli_runner.invoke(
            dry_run,
            [
                "--question",
                "Test question",
                "--participants",
                "claude,codex",
                "--config",
                str(temp_config_file),
            ],
        )

        assert result.exit_code == 0
        assert "Token Estimates" in result.output
        assert "input tokens" in result.output.lower()
        assert "output tokens" in result.output.lower()

    def test_dry_run_with_context(self, cli_runner, temp_config_file):
        """Test dry-run with context option."""
        result = cli_runner.invoke(
            dry_run,
            [
                "--question",
                "Test question",
                "--participants",
                "claude,codex",
                "--context",
                "Some additional context",
                "--config",
                str(temp_config_file),
            ],
        )

        assert result.exit_code == 0
        assert "Context" in result.output

    def test_dry_run_with_rounds(self, cli_runner, temp_config_file):
        """Test dry-run with custom rounds."""
        result = cli_runner.invoke(
            dry_run,
            [
                "--question",
                "Test",
                "--participants",
                "claude,codex",
                "--rounds",
                "3",
                "--format",
                "json",
                "--config",
                str(temp_config_file),
            ],
        )

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["request"]["rounds"] == 3

    def test_dry_run_validates_participants(self, cli_runner, temp_config_file):
        """Test dry-run validates participants."""
        result = cli_runner.invoke(
            dry_run,
            [
                "--question",
                "Test",
                "--participants",
                "nonexistent_adapter",
                "--config",
                str(temp_config_file),
            ],
        )

        assert result.exit_code == 1
        assert (
            "error" in result.output.lower()
            or "not configured" in result.output.lower()
        )

    def test_dry_run_requires_two_participants(self, cli_runner, temp_config_file):
        """Test dry-run requires at least 2 participants."""
        result = cli_runner.invoke(
            dry_run,
            [
                "--question",
                "Test",
                "--participants",
                "claude",
                "--config",
                str(temp_config_file),
            ],
        )

        assert result.exit_code == 1
        assert "at least 2 participants" in result.output.lower()


# ============================================================================
# TEST: list-models command
# ============================================================================


class TestListModelsCommand:
    """Tests for 'list-models' CLI command."""

    def test_list_models_table_format(self, cli_runner, temp_config_file):
        """Test list-models table format (default)."""
        result = cli_runner.invoke(
            list_models,
            ["--config", str(temp_config_file)],
        )

        assert result.exit_code == 0
        assert "CLAUDE" in result.output
        assert "CODEX" in result.output
        assert "claude-sonnet-4-5-20250929" in result.output

    def test_list_models_json_format(self, cli_runner, temp_config_file):
        """Test list-models JSON format."""
        result = cli_runner.invoke(
            list_models,
            ["--config", str(temp_config_file), "--format", "json"],
        )

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert "adapters" in output
        assert "claude" in output["adapters"]
        assert "codex" in output["adapters"]
        assert "total_models" in output

    def test_list_models_text_format(self, cli_runner, temp_config_file):
        """Test list-models text format."""
        result = cli_runner.invoke(
            list_models,
            ["--config", str(temp_config_file), "--format", "text"],
        )

        assert result.exit_code == 0
        assert "claude:" in result.output
        assert "codex:" in result.output

    def test_list_models_filter_by_adapter(self, cli_runner, temp_config_file):
        """Test list-models filtering by adapter."""
        result = cli_runner.invoke(
            list_models,
            ["--config", str(temp_config_file), "--adapter", "claude"],
        )

        assert result.exit_code == 0
        assert "claude" in result.output.lower()
        # Should not show codex when filtered
        # (Note: codex will not appear in the models section)

    def test_list_models_show_disabled(self, cli_runner, temp_config_file):
        """Test list-models --show-disabled includes disabled models."""
        # Default (without --show-disabled)
        result_default = cli_runner.invoke(
            list_models,
            ["--config", str(temp_config_file), "--format", "json"],
        )
        output_default = json.loads(result_default.output)

        # With --show-disabled
        result_disabled = cli_runner.invoke(
            list_models,
            ["--config", str(temp_config_file), "--format", "json", "--show-disabled"],
        )
        output_disabled = json.loads(result_disabled.output)

        # Should have more models when showing disabled
        assert output_disabled["total_models"] >= output_default["total_models"]

    def test_list_models_shows_tier(self, cli_runner, temp_config_file):
        """Test list-models shows tier information."""
        result = cli_runner.invoke(
            list_models,
            ["--config", str(temp_config_file)],
        )

        assert result.exit_code == 0
        # Should show tier info
        assert "balanced" in result.output.lower() or "premium" in result.output.lower()

    def test_list_models_shows_free_vs_paid(self, cli_runner):
        """Test list-models distinguishes free vs paid tiers."""
        config_data = {
            "version": "1.0",
            "cli_tools": {
                "openrouter": {
                    "command": "openrouter",
                    "args": ["{prompt}"],
                    "timeout": 60,
                }
            },
            "defaults": {
                "mode": "quick",
                "rounds": 2,
                "max_rounds": 5,
                "timeout_per_round": 120,
            },
            "model_registry": {
                "openrouter": [
                    {"id": "paid-model", "tier": "premium", "enabled": True},
                    {"id": "free-model", "tier": "free", "enabled": True},
                    {"id": "free-fast-model", "tier": "free-fast", "enabled": True},
                ],
            },
            "storage": {
                "transcripts_dir": "t",
                "format": "markdown",
                "auto_export": True,
            },
            "deliberation": {
                "convergence_detection": {
                    "enabled": True,
                    "semantic_similarity_threshold": 0.85,
                    "divergence_threshold": 0.40,
                    "min_rounds_before_check": 1,
                    "consecutive_stable_rounds": 2,
                    "stance_stability_threshold": 0.80,
                    "response_length_drop_threshold": 0.40,
                },
                "early_stopping": {
                    "enabled": True,
                    "threshold": 0.66,
                    "respect_min_rounds": True,
                },
                "convergence_threshold": 0.8,
                "enable_convergence_detection": True,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        result = cli_runner.invoke(
            list_models,
            ["--config", temp_path],
        )

        assert result.exit_code == 0
        # Should show both free and paid
        assert "free" in result.output.lower()
        assert "paid" in result.output.lower()

        Path(temp_path).unlink()


# ============================================================================
# TEST: Help text
# ============================================================================


class TestHelpText:
    """Tests for CLI help text."""

    def test_cli_group_help(self, cli_runner):
        """Test CLI group help text."""
        result = cli_runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "AI Counsel CLI" in result.output
        assert "validate-config" in result.output
        assert "dry-run" in result.output
        assert "list-models" in result.output

    def test_validate_config_help(self, cli_runner):
        """Test validate-config help text."""
        result = cli_runner.invoke(validate_config, ["--help"])

        assert result.exit_code == 0
        assert "Validate config.yaml" in result.output
        assert "--config" in result.output
        assert "--format" in result.output

    def test_dry_run_help(self, cli_runner):
        """Test dry-run help text."""
        result = cli_runner.invoke(dry_run, ["--help"])

        assert result.exit_code == 0
        assert "Simulate deliberation" in result.output
        assert "--question" in result.output
        assert "--participants" in result.output
        assert "--rounds" in result.output

    def test_list_models_help(self, cli_runner):
        """Test list-models help text."""
        result = cli_runner.invoke(list_models, ["--help"])

        assert result.exit_code == 0
        assert "List all configured models" in result.output
        assert "--adapter" in result.output
        assert "--format" in result.output
        assert "--show-disabled" in result.output


# ============================================================================
# TEST: Integration with project config
# ============================================================================


class TestIntegrationWithProjectConfig:
    """Tests using actual project config.yaml."""

    def test_validate_project_config(self, cli_runner):
        """Test validating actual project config.yaml."""
        result = cli_runner.invoke(validate_config, [])

        # Project config should be valid
        assert result.exit_code == 0

    def test_list_models_project_config(self, cli_runner):
        """Test listing models from actual project config."""
        result = cli_runner.invoke(list_models, ["--format", "json"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["total_models"] > 0

    def test_dry_run_project_config(self, cli_runner):
        """Test dry-run with actual project config."""
        result = cli_runner.invoke(
            dry_run,
            [
                "--question",
                "Test question for dry run",
                "--participants",
                "claude,codex",
            ],
        )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output


# ============================================================================
# TEST: Update Models Command
# ============================================================================


class TestUpdateModelsCommand:
    """Tests for update-models CLI command."""

    def test_update_models_help(self, cli_runner):
        """Test update-models help text."""
        from cli.commands import update_models

        result = cli_runner.invoke(update_models, ["--help"])

        assert result.exit_code == 0
        assert "Update free models list" in result.output
        assert "--dry-run" in result.output
        assert "--api-key" in result.output
        assert "--verbose" in result.output

    def test_update_models_missing_config(self, cli_runner):
        """Test update-models with missing config file."""
        from cli.commands import update_models

        result = cli_runner.invoke(
            update_models,
            ["--config", "/nonexistent/config.yaml"],
        )

        assert result.exit_code == 1
        assert "Error" in result.output
