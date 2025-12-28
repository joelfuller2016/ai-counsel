"""CLI commands for AI Counsel configuration and diagnostics.

Provides command-line interface for:
- --validate-config: Validate config.yaml syntax and model availability
- --dry-run: Simulate deliberation without calling models
- --list-models: List all configured models with status
- --health-check: Check availability of HTTP models (e.g., OpenRouter free models)

Usage:
    python -m cli.commands validate-config [--config PATH]
    python -m cli.commands dry-run --question "..." --participants "claude:sonnet,codex:gpt-5.2" [OPTIONS]
    python -m cli.commands list-models [--adapter NAME] [--format FORMAT]
    python -m cli.commands health-check [--adapter NAME] [--model MODEL_ID]
"""

import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import click
import yaml

from datetime import datetime

from adapters import create_adapter
from adapters.base_http import BaseHTTPAdapter, HealthCheckResult
from models.config import (
    CLIAdapterConfig,
    CLIToolConfig,
    Config,
    HTTPAdapterConfig,
    load_config,
)
from models.model_registry import ModelRegistry
from models.openrouter_discovery import (
    OpenRouterModelDiscovery,
    OpenRouterModelInfo,
)

logger = logging.getLogger(__name__)

# Default config path (project root)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class ConfigValidationResult:
    """Result of configuration validation."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        self.info.append(message)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


def validate_yaml_syntax(
    config_path: Path,
) -> tuple[bool, Optional[str], Optional[dict]]:
    """Validate YAML syntax.

    Returns:
        Tuple of (is_valid, error_message, parsed_data)
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return True, None, data
    except yaml.YAMLError as e:
        return False, f"YAML syntax error: {e}", None
    except FileNotFoundError:
        return False, f"Config file not found: {config_path}", None
    except Exception as e:
        return False, f"Error reading config: {e}", None


def validate_config_schema(
    config_path: Path,
) -> tuple[bool, Optional[str], Optional[Config]]:
    """Validate config against Pydantic schema.

    Returns:
        Tuple of (is_valid, error_message, config_object)
    """
    try:
        config = load_config(str(config_path))
        return True, None, config
    except Exception as e:
        return False, f"Schema validation error: {e}", None


def check_adapter_availability(
    adapter_name: str, config: CLIToolConfig | CLIAdapterConfig | HTTPAdapterConfig
) -> tuple[bool, str]:
    """Check if adapter command/service is available.

    Returns:
        Tuple of (is_available, status_message)
    """
    if isinstance(config, (CLIToolConfig, CLIAdapterConfig)):
        # Check if CLI command exists
        command = config.command
        if shutil.which(command):
            return True, f"CLI command '{command}' found in PATH"
        else:
            return False, f"CLI command '{command}' not found in PATH"
    elif isinstance(config, HTTPAdapterConfig):
        # For HTTP adapters, just check if URL is configured
        if config.base_url:
            return True, f"HTTP adapter configured with base_url: {config.base_url}"
        else:
            return False, "HTTP adapter missing base_url"
    return False, "Unknown adapter type"


def validate_model_registry(
    config: Config, registry: ModelRegistry
) -> list[tuple[str, str, bool, str]]:
    """Validate model registry entries.

    Returns:
        List of (adapter, model_id, is_valid, message) tuples
    """
    results = []

    if not config.model_registry:
        return results

    for adapter_name, models in config.model_registry.items():
        # Check if adapter is configured
        adapter_configured = False
        if config.adapters and adapter_name in config.adapters:
            adapter_configured = True
        if config.cli_tools and adapter_name in config.cli_tools:
            adapter_configured = True

        for model in models:
            if not adapter_configured:
                results.append(
                    (
                        adapter_name,
                        model.id,
                        False,
                        f"Adapter '{adapter_name}' not configured in adapters or cli_tools section",
                    )
                )
            elif not model.enabled:
                results.append((adapter_name, model.id, True, "Model is disabled"))
            else:
                results.append(
                    (adapter_name, model.id, True, "Model configured and enabled")
                )

    return results


def perform_full_validation(config_path: Path) -> ConfigValidationResult:
    """Perform complete configuration validation.

    Checks:
    1. YAML syntax
    2. Pydantic schema validation
    3. Adapter availability (CLI commands in PATH, HTTP endpoints configured)
    4. Model registry entries
    """
    result = ConfigValidationResult()

    # Step 1: YAML syntax
    yaml_valid, yaml_error, _ = validate_yaml_syntax(config_path)
    if not yaml_valid:
        result.add_error(yaml_error)
        return result
    result.add_info("YAML syntax: OK")

    # Step 2: Schema validation
    schema_valid, schema_error, config = validate_config_schema(config_path)
    if not schema_valid:
        result.add_error(schema_error)
        return result
    result.add_info("Pydantic schema: OK")

    # Step 3: Check adapters
    adapters_found: dict[str, tuple[bool, str]] = {}

    if config.adapters:
        for name, adapter_config in config.adapters.items():
            available, message = check_adapter_availability(name, adapter_config)
            adapters_found[name] = (available, message)
            if available:
                result.add_info(f"Adapter '{name}': {message}")
            else:
                result.add_warning(f"Adapter '{name}': {message}")

    if config.cli_tools:
        for name, tool_config in config.cli_tools.items():
            if name not in adapters_found:  # Don't duplicate
                available, message = check_adapter_availability(name, tool_config)
                adapters_found[name] = (available, message)
                if available:
                    result.add_info(f"CLI tool '{name}': {message}")
                else:
                    result.add_warning(f"CLI tool '{name}': {message}")

    if not adapters_found:
        result.add_error("No adapters configured")
        return result

    # Step 4: Validate model registry
    registry = ModelRegistry(config)
    model_results = validate_model_registry(config, registry)

    enabled_count = 0
    disabled_count = 0
    orphaned_count = 0

    for adapter, model_id, is_valid, message in model_results:
        if not is_valid:
            result.add_error(f"Model '{model_id}' ({adapter}): {message}")
            orphaned_count += 1
        elif "disabled" in message.lower():
            result.add_info(f"Model '{model_id}' ({adapter}): {message}")
            disabled_count += 1
        else:
            enabled_count += 1

    result.add_info(
        f"Models: {enabled_count} enabled, {disabled_count} disabled, {orphaned_count} orphaned"
    )

    return result


def estimate_token_usage(
    question: str, context: Optional[str], rounds: int, participant_count: int
) -> dict:
    """Estimate token usage for a deliberation.

    This is a rough estimation based on typical response sizes.
    """
    # Rough estimates (1 token ~= 4 chars for English)
    chars_per_token = 4

    # Input tokens
    question_tokens = len(question) // chars_per_token
    context_tokens = len(context) // chars_per_token if context else 0

    # System prompt estimate (includes voting instructions, etc.)
    system_prompt_tokens = 500

    # Context accumulation (previous responses fed to next round)
    # Assume ~800 tokens per response, accumulating each round
    response_tokens_per_participant = 800
    context_growth_per_round = response_tokens_per_participant * participant_count

    # Output tokens (responses)
    output_tokens_per_round = response_tokens_per_participant * participant_count
    total_output_tokens = output_tokens_per_round * rounds

    # Input tokens grow each round due to context
    total_input_tokens = 0
    for r in range(rounds):
        round_input = system_prompt_tokens + question_tokens + context_tokens
        round_input += context_growth_per_round * r  # Previous rounds' responses
        total_input_tokens += round_input * participant_count

    return {
        "question_tokens": question_tokens,
        "context_tokens": context_tokens,
        "estimated_input_tokens_per_round": [
            (
                system_prompt_tokens
                + question_tokens
                + context_tokens
                + context_growth_per_round * r
            )
            * participant_count
            for r in range(rounds)
        ],
        "estimated_output_tokens_per_round": output_tokens_per_round,
        "total_estimated_input_tokens": total_input_tokens,
        "total_estimated_output_tokens": total_output_tokens,
        "total_estimated_tokens": total_input_tokens + total_output_tokens,
        "note": "Estimates based on ~4 chars/token and ~800 tokens/response",
    }


def parse_participants(participants_str: str, config: Config) -> list[dict]:
    """Parse participant string into list of participant dicts.

    Format: "adapter:model,adapter:model" or "adapter,adapter" (uses default models)
    Examples:
        "claude:claude-sonnet-4-5-20250929,codex:gpt-5.2-codex"
        "claude,codex,droid"  (uses default models)
    """
    participants = []
    registry = ModelRegistry(config)

    for part in participants_str.split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            adapter, model = part.split(":", 1)
            adapter = adapter.strip()
            model = model.strip()
        else:
            adapter = part
            model = None

        # Resolve model if not specified
        if not model:
            model = registry.get_default(adapter)

        participants.append(
            {
                "cli": adapter,
                "model": model,
            }
        )

    return participants


# ============================================================================
# CLI Commands
# ============================================================================


@click.group()
def cli():
    """AI Counsel CLI - Configuration and diagnostics tools."""
    pass


@cli.command("validate-config")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=False),
    default=None,
    help="Path to config.yaml (default: project root)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def validate_config(config_path: Optional[str], output_format: str) -> None:
    """Validate config.yaml syntax and model availability.

    Checks:
    - YAML syntax
    - Pydantic schema validation
    - Adapter availability (CLI commands in PATH)
    - Model registry entries

    Example:
        python -m cli.commands validate-config
        python -m cli.commands validate-config --config /path/to/config.yaml
        python -m cli.commands validate-config --format json
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    result = perform_full_validation(path)

    if output_format == "json":
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        # Text format
        if result.valid:
            click.echo(click.style("Configuration is valid", fg="green", bold=True))
        else:
            click.echo(click.style("Configuration has errors", fg="red", bold=True))

        if result.errors:
            click.echo(click.style("\nErrors:", fg="red"))
            for error in result.errors:
                click.echo(click.style(f"  - {error}", fg="red"))

        if result.warnings:
            click.echo(click.style("\nWarnings:", fg="yellow"))
            for warning in result.warnings:
                click.echo(click.style(f"  - {warning}", fg="yellow"))

        if result.info:
            click.echo(click.style("\nInfo:", fg="blue"))
            for info in result.info:
                click.echo(f"  - {info}")

    sys.exit(0 if result.valid else 1)


@cli.command("dry-run")
@click.option(
    "--question",
    "-q",
    required=True,
    help="The question for deliberation",
)
@click.option(
    "--participants",
    "-p",
    required=True,
    help="Participants as 'adapter:model,adapter:model' or 'adapter,adapter'",
)
@click.option(
    "--rounds",
    "-r",
    type=int,
    default=2,
    help="Number of deliberation rounds (default: 2)",
)
@click.option(
    "--context",
    help="Additional context for the deliberation",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=False),
    default=None,
    help="Path to config.yaml (default: project root)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def dry_run(
    question: str,
    participants: str,
    rounds: int,
    context: Optional[str],
    config_path: Optional[str],
    output_format: str,
) -> None:
    """Simulate deliberation without calling models.

    Shows:
    - Which models would be invoked
    - Estimated token usage
    - Configuration details

    Example:
        python -m cli.commands dry-run --question "Should we use TypeScript?" --participants "claude,codex"
        python -m cli.commands dry-run -q "API design?" -p "claude:claude-sonnet-4-5-20250929,codex:gpt-5.2-codex" -r 3
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    # Validate config first
    schema_valid, schema_error, config = validate_config_schema(path)
    if not schema_valid:
        click.echo(click.style(f"Config error: {schema_error}", fg="red"), err=True)
        sys.exit(1)

    # Parse participants
    try:
        parsed_participants = parse_participants(participants, config)
    except Exception as e:
        click.echo(click.style(f"Error parsing participants: {e}", fg="red"), err=True)
        sys.exit(1)

    if len(parsed_participants) < 2:
        click.echo(
            click.style("Error: At least 2 participants required", fg="red"), err=True
        )
        sys.exit(1)

    # Validate each participant
    registry = ModelRegistry(config)
    validation_errors = []

    for p in parsed_participants:
        adapter = p["cli"]
        model = p["model"]

        # Check adapter exists
        adapter_exists = False
        if config.adapters and adapter in config.adapters:
            adapter_exists = True
        if config.cli_tools and adapter in config.cli_tools:
            adapter_exists = True

        if not adapter_exists:
            validation_errors.append(f"Adapter '{adapter}' not configured")
            continue

        # Check model is valid
        if model:
            validation = registry.validate_model(adapter, model)
            if not validation.valid:
                validation_errors.append(
                    validation.error_message
                    or f"Invalid model '{model}' for adapter '{adapter}'"
                )

    if validation_errors:
        for error in validation_errors:
            click.echo(click.style(f"Validation error: {error}", fg="red"), err=True)
        sys.exit(1)

    # Estimate token usage
    token_estimate = estimate_token_usage(
        question=question,
        context=context,
        rounds=rounds,
        participant_count=len(parsed_participants),
    )

    # Build result
    result = {
        "dry_run": True,
        "would_execute": False,
        "request": {
            "question": question,
            "context": context,
            "rounds": rounds,
            "participants": parsed_participants,
        },
        "models_to_invoke": [
            {
                "adapter": p["cli"],
                "model": p["model"],
                "invocations": rounds,
            }
            for p in parsed_participants
        ],
        "token_estimate": token_estimate,
        "total_model_calls": len(parsed_participants) * rounds,
    }

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(click.style("=== DRY RUN ===", fg="cyan", bold=True))
        click.echo(click.style("No API calls will be made\n", fg="cyan"))

        click.echo(click.style("Question:", bold=True))
        click.echo(f"  {question}\n")

        if context:
            click.echo(click.style("Context:", bold=True))
            click.echo(f"  {context[:100]}{'...' if len(context) > 100 else ''}\n")

        click.echo(click.style("Configuration:", bold=True))
        click.echo(f"  Rounds: {rounds}")
        click.echo(f"  Participants: {len(parsed_participants)}")
        click.echo(f"  Total model calls: {len(parsed_participants) * rounds}\n")

        click.echo(click.style("Models to invoke:", bold=True))
        for p in parsed_participants:
            click.echo(f"  - {p['cli']}: {p['model']} ({rounds} calls)")

        click.echo(click.style("\nToken Estimates:", bold=True))
        click.echo(
            f"  Estimated input tokens: {token_estimate['total_estimated_input_tokens']:,}"
        )
        click.echo(
            f"  Estimated output tokens: {token_estimate['total_estimated_output_tokens']:,}"
        )
        click.echo(
            f"  Total estimated tokens: {token_estimate['total_estimated_tokens']:,}"
        )
        click.echo(f"  Note: {token_estimate['note']}")


@cli.command("list-models")
@click.option(
    "--adapter",
    "-a",
    help="Filter by adapter name (e.g., claude, codex, openrouter)",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=False),
    default=None,
    help="Path to config.yaml (default: project root)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json", "table"]),
    default="table",
    help="Output format",
)
@click.option(
    "--show-disabled",
    is_flag=True,
    help="Include disabled models in output",
)
def list_models(
    adapter: Optional[str],
    config_path: Optional[str],
    output_format: str,
    show_disabled: bool,
) -> None:
    """List all configured models with status.

    Shows:
    - Model ID, adapter, tier, enabled status
    - Groups by adapter
    - Indicates free vs paid tiers

    Example:
        python -m cli.commands list-models
        python -m cli.commands list-models --adapter openrouter
        python -m cli.commands list-models --format json --show-disabled
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    # Load config
    schema_valid, schema_error, config = validate_config_schema(path)
    if not schema_valid:
        click.echo(click.style(f"Config error: {schema_error}", fg="red"), err=True)
        sys.exit(1)

    registry = ModelRegistry(config)

    # Collect models
    models_by_adapter: dict[str, list[dict]] = {}

    if config.model_registry:
        for adapter_name, models in config.model_registry.items():
            if adapter and adapter_name != adapter:
                continue

            adapter_models = []
            for model in models:
                if not show_disabled and not model.enabled:
                    continue

                adapter_models.append(
                    {
                        "id": model.id,
                        "label": model.label or model.id,
                        "tier": model.tier or "general",
                        "enabled": model.enabled,
                        "default": model.default,
                        "timeout": model.timeout,
                    }
                )

            if adapter_models:
                models_by_adapter[adapter_name] = adapter_models

    # Determine which tiers are free
    FREE_TIERS = {"free", "free-fast"}

    if output_format == "json":
        result = {
            "adapters": models_by_adapter,
            "total_models": sum(len(m) for m in models_by_adapter.values()),
            "free_tiers": list(FREE_TIERS),
        }
        click.echo(json.dumps(result, indent=2))

    elif output_format == "table":
        for adapter_name, models in sorted(models_by_adapter.items()):
            click.echo(click.style(f"\n{adapter_name.upper()}", fg="cyan", bold=True))
            click.echo("-" * 80)

            # Header
            click.echo(f"{'ID':<45} {'Tier':<15} {'Status':<10} {'Default'}")
            click.echo("-" * 80)

            for model in models:
                model_id = model["id"]
                if len(model_id) > 44:
                    model_id = model_id[:41] + "..."

                tier = model["tier"]
                is_free = tier.lower() in FREE_TIERS if tier else False

                # Color coding
                if not model["enabled"]:
                    status = click.style("disabled", fg="red")
                elif is_free:
                    status = click.style("free", fg="green")
                else:
                    status = click.style("paid", fg="yellow")

                default_marker = "*" if model["default"] else ""

                click.echo(f"{model_id:<45} {tier:<15} {status:<19} {default_marker}")

        click.echo("")
        total = sum(len(m) for m in models_by_adapter.values())
        click.echo(f"Total: {total} models")
        if not show_disabled:
            click.echo("(Use --show-disabled to include disabled models)")

    else:  # text format
        for adapter_name, models in sorted(models_by_adapter.items()):
            click.echo(f"\n{adapter_name}:")
            for model in models:
                status = "enabled" if model["enabled"] else "disabled"
                tier = model["tier"]
                is_free = tier.lower() in FREE_TIERS if tier else False
                cost = "free" if is_free else "paid"
                default_marker = " (default)" if model["default"] else ""

                click.echo(
                    f"  - {model['id']}: {tier} ({cost}, {status}){default_marker}"
                )


async def check_model_health(
    adapter: BaseHTTPAdapter,
    model_id: str,
    timeout: float = 15.0,
) -> HealthCheckResult:
    """Check health of a single model.

    Args:
        adapter: HTTP adapter instance
        model_id: Model identifier to check
        timeout: Timeout in seconds

    Returns:
        HealthCheckResult with availability status
    """
    return await adapter.health_check(model=model_id, timeout=timeout)


async def check_all_http_models(
    config: Config,
    adapter_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    timeout: float = 15.0,
) -> list[HealthCheckResult]:
    """Check health of all HTTP-based models.

    Args:
        config: Application configuration
        adapter_filter: Optional filter for specific adapter
        model_filter: Optional filter for specific model
        timeout: Timeout per health check in seconds

    Returns:
        List of HealthCheckResult for each checked model
    """
    results: list[HealthCheckResult] = []

    if not config.model_registry:
        return results

    # Get HTTP adapters from config
    http_adapters: dict[str, HTTPAdapterConfig] = {}
    if config.adapters:
        for name, adapter_config in config.adapters.items():
            if isinstance(adapter_config, HTTPAdapterConfig):
                if adapter_filter and name != adapter_filter:
                    continue
                http_adapters[name] = adapter_config

    if not http_adapters:
        return results

    # Check each model
    tasks = []
    model_info = []  # Track (adapter_name, model_id) for each task

    for adapter_name, adapter_config in http_adapters.items():
        if adapter_name not in config.model_registry:
            continue

        # Create adapter instance
        try:
            adapter = create_adapter(adapter_name, adapter_config)
            if not isinstance(adapter, BaseHTTPAdapter):
                continue
        except Exception as e:
            logger.warning(f"Failed to create adapter {adapter_name}: {e}")
            continue

        # Get models for this adapter
        models = config.model_registry[adapter_name]

        for model_def in models:
            if not model_def.enabled:
                continue

            if model_filter and model_def.id != model_filter:
                continue

            # Create health check task
            task = check_model_health(adapter, model_def.id, timeout)
            tasks.append(task)
            model_info.append((adapter_name, model_def.id))

    if not tasks:
        return results

    # Run health checks concurrently
    check_results = await asyncio.gather(*tasks, return_exceptions=True)

    for (adapter_name, model_id), result in zip(model_info, check_results):
        if isinstance(result, Exception):
            results.append(
                HealthCheckResult(
                    available=False,
                    error=f"{type(result).__name__}: {str(result)}",
                    model=model_id,
                    adapter=adapter_name,
                )
            )
        else:
            results.append(result)

    return results


@cli.command("health-check")
@click.option(
    "--adapter",
    "-a",
    help="Filter by adapter name (e.g., openrouter, ollama, lmstudio)",
)
@click.option(
    "--model",
    "-m",
    "model_id",
    help="Check specific model ID only",
)
@click.option(
    "--timeout",
    "-t",
    type=float,
    default=15.0,
    help="Timeout per health check in seconds (default: 15)",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=False),
    default=None,
    help="Path to config.yaml (default: project root)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json", "table"]),
    default="table",
    help="Output format",
)
def health_check(
    adapter: Optional[str],
    model_id: Optional[str],
    timeout: float,
    config_path: Optional[str],
    output_format: str,
) -> None:
    """Check availability of HTTP-based models before deliberation.

    Tests connectivity and response from configured HTTP adapters
    (OpenRouter, Ollama, LM Studio). Useful for checking free-tier
    model availability which may be rate-limited.

    Example:
        python -m cli.commands health-check
        python -m cli.commands health-check --adapter openrouter
        python -m cli.commands health-check --model "meta-llama/llama-3.2-3b-instruct:free"
        python -m cli.commands health-check --timeout 30 --format json
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    # Load config
    schema_valid, schema_error, config = validate_config_schema(path)
    if not schema_valid:
        click.echo(click.style(f"Config error: {schema_error}", fg="red"), err=True)
        sys.exit(1)

    # Run health checks
    click.echo(click.style("Checking model availability...\n", fg="cyan"))

    try:
        results = asyncio.run(
            check_all_http_models(
                config=config,
                adapter_filter=adapter,
                model_filter=model_id,
                timeout=timeout,
            )
        )
    except Exception as e:
        click.echo(click.style(f"Error running health checks: {e}", fg="red"), err=True)
        sys.exit(1)

    if not results:
        click.echo(click.style("No HTTP models found to check.", fg="yellow"))
        click.echo(
            "Health checks only apply to HTTP adapters (openrouter, ollama, lmstudio)."
        )
        if adapter:
            click.echo(f"Adapter filter: {adapter}")
        sys.exit(0)

    # Count results
    available_count = sum(1 for r in results if r.available)
    unavailable_count = len(results) - available_count

    if output_format == "json":
        output = {
            "results": [
                {
                    "adapter": r.adapter,
                    "model": r.model,
                    "available": r.available,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in results
            ],
            "summary": {
                "total": len(results),
                "available": available_count,
                "unavailable": unavailable_count,
            },
        }
        click.echo(json.dumps(output, indent=2))

    elif output_format == "table":
        # Group by adapter
        results_by_adapter: dict[str, list[HealthCheckResult]] = {}
        for r in results:
            adapter_name = r.adapter or "unknown"
            if adapter_name not in results_by_adapter:
                results_by_adapter[adapter_name] = []
            results_by_adapter[adapter_name].append(r)

        for adapter_name, adapter_results in sorted(results_by_adapter.items()):
            click.echo(click.style(f"\n{adapter_name.upper()}", fg="cyan", bold=True))
            click.echo("-" * 90)

            # Header
            click.echo(f"{'Model':<50} {'Status':<12} {'Latency':<12} {'Error'}")
            click.echo("-" * 90)

            for r in adapter_results:
                model_display = r.model or "unknown"
                if len(model_display) > 49:
                    model_display = model_display[:46] + "..."

                if r.available:
                    status = click.style("available", fg="green")
                    latency = f"{r.latency_ms:.0f}ms" if r.latency_ms else "-"
                    error = ""
                else:
                    status = click.style("unavailable", fg="red")
                    latency = "-"
                    error = (
                        r.error[:30] + "..."
                        if r.error and len(r.error) > 30
                        else (r.error or "")
                    )

                click.echo(f"{model_display:<50} {status:<21} {latency:<12} {error}")

        click.echo("")
        click.echo("-" * 90)

        # Summary
        if available_count == len(results):
            click.echo(
                click.style(
                    f"All {len(results)} models available", fg="green", bold=True
                )
            )
        elif unavailable_count == len(results):
            click.echo(
                click.style(
                    f"All {len(results)} models unavailable", fg="red", bold=True
                )
            )
        else:
            click.echo(
                f"Summary: "
                + click.style(f"{available_count} available", fg="green")
                + ", "
                + click.style(f"{unavailable_count} unavailable", fg="red")
                + f" (of {len(results)} total)"
            )

    else:  # text format
        for r in results:
            if r.available:
                latency = f" ({r.latency_ms:.0f}ms)" if r.latency_ms else ""
                click.echo(
                    click.style(f"[OK] {r.adapter}/{r.model}{latency}", fg="green")
                )
            else:
                click.echo(
                    click.style(f"[FAIL] {r.adapter}/{r.model}: {r.error}", fg="red")
                )

        click.echo("")
        click.echo(f"Available: {available_count}/{len(results)}")

    # Exit with error if any models unavailable
    sys.exit(0 if unavailable_count == 0 else 1)


async def update_free_models_async(
    config_path: Path,
    api_key: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Update free models in config from OpenRouter API.

    Args:
        config_path: Path to config.yaml
        api_key: Optional OpenRouter API key
        dry_run: If True, don't write changes

    Returns:
        Dict with update results
    """
    # Load current config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        original_content = f.read()
        config_data = yaml.safe_load(original_content)

    # Get existing openrouter models
    existing_openrouter = config_data.get("model_registry", {}).get("openrouter", [])

    # Fetch models from API
    discovery = OpenRouterModelDiscovery(api_key=api_key)
    result = await discovery.fetch_models()

    if result.error:
        return {
            "success": False,
            "error": result.error,
            "models_found": 0,
        }

    # Filter to free models
    free_models = discovery.filter_free_models(result.models)

    if not free_models:
        return {
            "success": False,
            "error": "No free models found in API response",
            "total_models": result.total_count,
        }

    # Build lookup of existing customizations by model ID
    existing_by_id = {}
    for entry in existing_openrouter:
        model_id = entry.get("id", "")
        if model_id.endswith(":free") or "FREE" in entry.get("label", ""):
            existing_by_id[model_id] = entry

    # Generate new model entries preserving user customizations
    timestamp = datetime.now().isoformat()
    new_models = []

    for model in free_models:
        existing = existing_by_id.get(model.id, {})

        # Determine tier based on model characteristics
        model_id_lower = model.id.lower()
        name_lower = model.name.lower()

        if any(
            kw in model_id_lower or kw in name_lower
            for kw in ["deepseek-r1", "qwq", "reasoning"]
        ):
            default_tier = "free-reasoning"
        elif any(
            kw in model_id_lower or kw in name_lower
            for kw in ["code", "coder", "devstral", "starcoder"]
        ):
            default_tier = "free-coding"
        elif any(
            kw in model_id_lower or kw in name_lower
            for kw in ["flash", "mini", "scout", "small"]
        ):
            default_tier = "free-fast"
        elif model.context_length >= 100000:
            default_tier = "free-reliable"
        else:
            default_tier = "free"

        # Generate timeout based on model characteristics
        if any(
            kw in model_id_lower or kw in name_lower
            for kw in ["deepseek-r1", "qwq", "reasoning"]
        ):
            default_timeout = 120
        elif any(
            kw in model_id_lower or kw in name_lower for kw in ["70b", "72b", "65b"]
        ):
            default_timeout = 60
        elif any(
            kw in model_id_lower or kw in name_lower
            for kw in ["flash", "mini", "scout", "small"]
        ):
            default_timeout = 30
        else:
            default_timeout = 45

        # Generate label
        label = model.name
        if label.endswith("(free)"):
            label = label[:-6].strip()
        if len(label) > 50:
            label = label[:47] + "..."
        label = f"{label} FREE"

        new_models.append(
            {
                "id": model.id,
                "label": existing.get("label", label),
                "tier": existing.get("tier", default_tier),
                "enabled": existing.get("enabled", True),
                "timeout": existing.get("timeout", default_timeout),
            }
        )

    # Categorize models
    categories = discovery.categorize_free_models(free_models)
    category_counts = {cat: len(models) for cat, models in categories.items() if models}

    if not dry_run:
        # Update the config data
        if "model_registry" not in config_data:
            config_data["model_registry"] = {}

        # Keep non-free models, replace free models
        non_free_models = [
            m
            for m in existing_openrouter
            if not (m.get("id", "").endswith(":free") or "FREE" in m.get("label", ""))
        ]
        config_data["model_registry"]["openrouter"] = non_free_models + new_models

        # Add last_updated comment at the top
        header_comment = f"# OpenRouter free models last updated: {timestamp}\n"

        # Write updated config
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(header_comment)
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Updated {config_path} with {len(free_models)} free models")

    return {
        "success": True,
        "models_found": len(free_models),
        "total_models": result.total_count,
        "timestamp": timestamp,
        "dry_run": dry_run,
        "categories": category_counts,
    }


@cli.command("update-models")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=False),
    default=None,
    help="Path to config.yaml (default: project root)",
)
@click.option(
    "--api-key",
    "-k",
    envvar="OPENROUTER_API_KEY",
    help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
)
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    help="Show what would be updated without making changes",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed output",
)
def update_models(
    config_path: Optional[str],
    api_key: Optional[str],
    dry_run: bool,
    verbose: bool,
) -> None:
    """Update free models list from OpenRouter API.

    Fetches the latest available models from OpenRouter's /api/v1/models
    endpoint, filters for free models (zero cost), and updates the
    config.yaml file.

    User customizations (tiers, timeouts) are preserved.

    Examples:
        python -m cli.commands update-models
        python -m cli.commands update-models --dry-run
        python -m cli.commands update-models --config /path/to/config.yaml
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    click.echo(click.style("Fetching models from OpenRouter API...\n", fg="cyan"))

    try:
        result = asyncio.run(
            update_free_models_async(path, api_key=api_key, dry_run=dry_run)
        )

        if result.get("success"):
            mode = "[DRY RUN] " if dry_run else ""
            click.echo(
                click.style(
                    f"{mode}Free models update complete!", fg="green", bold=True
                )
            )
            click.echo(f'  Total models in API: {result["total_models"]}')
            click.echo(f'  Free models found: {result["models_found"]}')
            click.echo(f'  Timestamp: {result["timestamp"]}')

            if verbose and result.get("categories"):
                click.echo("\n  Categories:")
                for cat, count in result["categories"].items():
                    click.echo(f"    {cat}: {count}")

            if dry_run:
                click.echo("\n  Run without --dry-run to apply changes.")
        else:
            click.echo(click.style(f'Error: {result.get("error")}', fg="red"), err=True)
            sys.exit(1)

    except FileNotFoundError as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error updating models: {e}", exc_info=True)
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


# Entry point for `python -m cli.commands`
if __name__ == "__main__":
    cli()
