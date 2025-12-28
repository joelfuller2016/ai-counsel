"""Tests for the model registry utility."""
import pytest

from models.config import Config, load_config
from models.model_registry import ModelRegistry, ModelValidationResult


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    config = load_config()
    return ModelRegistry(config)


def test_registry_lists_models(registry: ModelRegistry):
    claude_entries = registry.list_for_adapter("claude")
    assert claude_entries, "Expected allowlisted Claude models"
    assert claude_entries[0].id == "claude-opus-4-5-20251101"  # First model in config
    assert registry.get_default("claude") == "claude-opus-4-5-20251101"  # Default model


def test_registry_enforces_allowlist(registry: ModelRegistry):
    assert registry.is_allowed("claude", "claude-haiku-4-5-20251001") is True
    assert registry.is_allowed("claude", "non-existent-model") is False


def test_registry_is_permissive_for_unmanaged_adapters(registry: ModelRegistry):
    # Adapters with no registry (e.g., llamacpp) accept any model name
    assert registry.is_allowed("llamacpp", "/path/to/model.gguf") is True


# ============================================================================
# Tests for enabled field feature
# ============================================================================


def _minimal_config(model_registry: dict) -> Config:
    """Helper to create minimal Config for testing model registry."""
    return Config.model_validate(
        {
            "version": "1.0",
            "cli_tools": {"test": {"command": "test", "args": [], "timeout": 60}},
            "defaults": {
                "mode": "quick",
                "rounds": 2,
                "max_rounds": 5,
                "timeout_per_round": 120,
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
                    "response_length_drop_threshold": 0.50,
                },
                "early_stopping": {
                    "enabled": True,
                    "threshold": 0.66,
                    "respect_min_rounds": True,
                },
                "convergence_threshold": 0.85,
                "enable_convergence_detection": True,
            },
            "model_registry": model_registry,
        }
    )


@pytest.fixture
def config_with_enabled_models() -> Config:
    """Create a config with mix of enabled and disabled models."""
    return _minimal_config(
        {
            "test_adapter": [
                {
                    "id": "model-enabled-1",
                    "label": "Enabled Model 1",
                    "enabled": True,
                    "default": True,
                },
                {"id": "model-enabled-2", "label": "Enabled Model 2", "enabled": True},
                {
                    "id": "model-disabled-1",
                    "label": "Disabled Model 1",
                    "enabled": False,
                },
                {
                    "id": "model-disabled-2",
                    "label": "Disabled Model 2",
                    "enabled": False,
                },
            ]
        }
    )


@pytest.fixture
def config_with_implicit_enabled() -> Config:
    """Create a config with models that don't specify enabled field."""
    return _minimal_config(
        {
            "test_adapter": [
                {"id": "model-implicit-1", "label": "Implicitly Enabled Model 1"},
                {
                    "id": "model-implicit-2",
                    "label": "Implicitly Enabled Model 2",
                    "default": True,
                },
            ]
        }
    )


@pytest.fixture
def config_with_all_disabled() -> Config:
    """Create a config where all models are disabled."""
    return _minimal_config(
        {
            "test_adapter": [
                {
                    "id": "model-disabled-1",
                    "label": "Disabled Model 1",
                    "enabled": False,
                },
                {
                    "id": "model-disabled-2",
                    "label": "Disabled Model 2",
                    "enabled": False,
                },
            ]
        }
    )


@pytest.fixture
def config_with_empty_registry() -> Config:
    """Create a config with empty model registry."""
    return _minimal_config({"test_adapter": []})


def test_list_for_adapter_returns_only_enabled_models(
    config_with_enabled_models: Config,
):
    """Test that list_for_adapter() filters out disabled models."""
    registry = ModelRegistry(config_with_enabled_models)

    enabled_models = registry.list_for_adapter("test_adapter")

    # Should only return enabled models
    assert len(enabled_models) == 2
    assert all(entry.id.startswith("model-enabled") for entry in enabled_models)

    # Verify specific models are present
    model_ids = {entry.id for entry in enabled_models}
    assert "model-enabled-1" in model_ids
    assert "model-enabled-2" in model_ids

    # Verify disabled models are not present
    assert "model-disabled-1" not in model_ids
    assert "model-disabled-2" not in model_ids


def test_list_for_adapter_with_disabled_models_excludes_them(
    config_with_enabled_models: Config,
):
    """Test that disabled models are explicitly excluded from results."""
    registry = ModelRegistry(config_with_enabled_models)

    enabled_models = registry.list_for_adapter("test_adapter")

    # None of the returned entries should have disabled models
    for entry in enabled_models:
        assert "disabled" not in entry.id.lower()


def test_backward_compatibility_models_without_enabled_field_default_to_true(
    config_with_implicit_enabled: Config,
):
    """Test that models without explicit enabled field default to enabled=True."""
    registry = ModelRegistry(config_with_implicit_enabled)

    models = registry.list_for_adapter("test_adapter")

    # All models should be returned (default enabled=True)
    assert len(models) == 2
    assert models[0].id == "model-implicit-2"  # default=True comes first
    assert models[1].id == "model-implicit-1"


def test_get_all_models_returns_all_regardless_of_enabled_status(
    config_with_enabled_models: Config,
):
    """Test that get_all_models() returns both enabled and disabled models."""
    registry = ModelRegistry(config_with_enabled_models)

    all_models = registry.get_all_models("test_adapter")

    # Should return all 4 models (2 enabled + 2 disabled)
    assert len(all_models) == 4

    model_ids = {entry.id for entry in all_models}
    assert "model-enabled-1" in model_ids
    assert "model-enabled-2" in model_ids
    assert "model-disabled-1" in model_ids
    assert "model-disabled-2" in model_ids


def test_get_all_models_preserves_ordering(config_with_enabled_models: Config):
    """Test that get_all_models() preserves the registry ordering."""
    registry = ModelRegistry(config_with_enabled_models)

    all_models = registry.get_all_models("test_adapter")

    # Default model should come first
    assert all_models[0].id == "model-enabled-1"
    assert all_models[0].default is True


def test_empty_registry_returns_empty_list(config_with_empty_registry: Config):
    """Test behavior with empty model registry."""
    registry = ModelRegistry(config_with_empty_registry)

    enabled_models = registry.list_for_adapter("test_adapter")
    all_models = registry.get_all_models("test_adapter")

    assert len(enabled_models) == 0
    assert len(all_models) == 0


def test_all_disabled_models_returns_empty_enabled_list(
    config_with_all_disabled: Config,
):
    """Test that when all models are disabled, list_for_adapter returns empty."""
    registry = ModelRegistry(config_with_all_disabled)

    enabled_models = registry.list_for_adapter("test_adapter")

    # No enabled models
    assert len(enabled_models) == 0


def test_all_disabled_models_but_get_all_returns_all(config_with_all_disabled: Config):
    """Test that get_all_models() still returns disabled models."""
    registry = ModelRegistry(config_with_all_disabled)

    all_models = registry.get_all_models("test_adapter")

    # Should return all disabled models
    assert len(all_models) == 2
    model_ids = {entry.id for entry in all_models}
    assert "model-disabled-1" in model_ids
    assert "model-disabled-2" in model_ids


def test_nonexistent_adapter_returns_empty_lists():
    """Test behavior when querying adapter that doesn't exist."""
    config = _minimal_config({})
    registry = ModelRegistry(config)

    enabled_models = registry.list_for_adapter("nonexistent")
    all_models = registry.get_all_models("nonexistent")

    assert len(enabled_models) == 0
    assert len(all_models) == 0


def test_allowed_ids_only_includes_enabled_models(config_with_enabled_models: Config):
    """Test that allowed_ids() only includes enabled models."""
    registry = ModelRegistry(config_with_enabled_models)

    allowed = registry.allowed_ids("test_adapter")

    # Should only include enabled models
    assert len(allowed) == 2
    assert "model-enabled-1" in allowed
    assert "model-enabled-2" in allowed
    assert "model-disabled-1" not in allowed
    assert "model-disabled-2" not in allowed


def test_is_allowed_returns_false_for_disabled_models(
    config_with_enabled_models: Config,
):
    """Test that is_allowed() returns False for disabled models."""
    registry = ModelRegistry(config_with_enabled_models)

    # Enabled models should be allowed
    assert registry.is_allowed("test_adapter", "model-enabled-1") is True
    assert registry.is_allowed("test_adapter", "model-enabled-2") is True

    # Disabled models should NOT be allowed
    assert registry.is_allowed("test_adapter", "model-disabled-1") is False
    assert registry.is_allowed("test_adapter", "model-disabled-2") is False


def test_get_default_ignores_disabled_models():
    """Test that get_default() skips disabled models even if marked as default."""
    config = _minimal_config(
        {
            "test_adapter": [
                {
                    "id": "model-disabled-default",
                    "label": "Disabled Default",
                    "enabled": False,
                    "default": True,
                },
                {
                    "id": "model-enabled-fallback",
                    "label": "Enabled Fallback",
                    "enabled": True,
                },
            ]
        }
    )
    registry = ModelRegistry(config)

    # Should skip disabled default and return first enabled model
    default = registry.get_default("test_adapter")
    assert default == "model-enabled-fallback"


def test_get_default_with_all_disabled_returns_none(config_with_all_disabled: Config):
    """Test that get_default() returns None when all models are disabled."""
    registry = ModelRegistry(config_with_all_disabled)

    default = registry.get_default("test_adapter")
    assert default is None


def test_mixed_enabled_disabled_maintains_ordering():
    """Test that enabled filtering preserves original order."""
    config = _minimal_config(
        {
            "test_adapter": [
                {
                    "id": "model-z-enabled",
                    "label": "Z Enabled",
                    "enabled": True,
                    "default": True,
                },
                {"id": "model-a-disabled", "label": "A Disabled", "enabled": False},
                {"id": "model-m-enabled", "label": "M Enabled", "enabled": True},
            ]
        }
    )
    registry = ModelRegistry(config)

    enabled_models = registry.list_for_adapter("test_adapter")

    # Should have 2 enabled models in original order (default first, then alphabetical)
    assert len(enabled_models) == 2
    assert enabled_models[0].id == "model-z-enabled"  # default=True comes first
    assert enabled_models[1].id == "model-m-enabled"


def test_enabled_field_is_stored_in_registry_entry():
    """Test that enabled status is preserved in RegistryEntry."""
    config = _minimal_config(
        {"test_adapter": [{"id": "model-1", "label": "Model 1", "enabled": True}]}
    )
    registry = ModelRegistry(config)

    # This tests that the enabled field is properly handled during construction
    # Even though RegistryEntry doesn't expose enabled, the filtering works
    models = registry.list_for_adapter("test_adapter")
    assert len(models) == 1
    assert models[0].id == "model-1"


# ============================================================================
# Tests for validate_model feature (GitHub Issue #36)
# ============================================================================


@pytest.fixture
def config_for_validation() -> Config:
    """Create a config with various model names for testing validation."""
    return _minimal_config(
        {
            "test_adapter": [
                {
                    "id": "claude-sonnet-4-5-20250929",
                    "label": "Claude Sonnet 4.5",
                    "enabled": True,
                    "default": True,
                },
                {
                    "id": "claude-opus-4-5-20251101",
                    "label": "Claude Opus 4.5",
                    "enabled": True,
                },
                {
                    "id": "claude-haiku-4-5-20251001",
                    "label": "Claude Haiku 4.5",
                    "enabled": True,
                },
                {
                    "id": "claude-3-5-sonnet-20240620",
                    "label": "Claude 3.5 Sonnet (Legacy)",
                    "enabled": False,
                },
            ]
        }
    )


def test_validate_model_returns_valid_for_enabled_model(config_for_validation: Config):
    """Test that validate_model returns valid=True for enabled models."""
    registry = ModelRegistry(config_for_validation)

    result = registry.validate_model("test_adapter", "claude-sonnet-4-5-20250929")

    assert result.valid is True
    assert result.exists is True
    assert result.enabled is True
    assert result.error_message is None
    assert result.similar_models == []


def test_validate_model_returns_invalid_for_disabled_model(
    config_for_validation: Config,
):
    """Test that validate_model returns valid=False for disabled models."""
    registry = ModelRegistry(config_for_validation)

    result = registry.validate_model("test_adapter", "claude-3-5-sonnet-20240620")

    assert result.valid is False
    assert result.exists is True
    assert result.enabled is False
    assert result.error_message is not None
    assert "exists but is disabled" in result.error_message
    assert "list_models" in result.error_message


def test_validate_model_returns_invalid_for_nonexistent_model(
    config_for_validation: Config,
):
    """Test that validate_model returns valid=False for non-existent models."""
    registry = ModelRegistry(config_for_validation)

    result = registry.validate_model("test_adapter", "nonexistent-model")

    assert result.valid is False
    assert result.exists is False
    assert result.enabled is None
    assert result.error_message is not None
    assert "not found" in result.error_message
    assert "list_models" in result.error_message


def test_validate_model_suggests_similar_models_for_typos(
    config_for_validation: Config,
):
    """Test that validate_model suggests similar models for typos."""
    registry = ModelRegistry(config_for_validation)

    # Typo in model name: "sonnet" -> "sonet"
    result = registry.validate_model("test_adapter", "claude-sonet-4-5-20250929")

    assert result.valid is False
    assert len(result.similar_models) > 0
    assert "claude-sonnet-4-5-20250929" in result.similar_models
    assert "Did you mean" in result.error_message


def test_validate_model_suggests_similar_for_partial_match(
    config_for_validation: Config,
):
    """Test that validate_model suggests models for partial matches."""
    registry = ModelRegistry(config_for_validation)

    # Partial match: just "opus"
    result = registry.validate_model("test_adapter", "claude-opus")

    assert result.valid is False
    # Should suggest the full opus model name
    assert any("opus" in m for m in result.similar_models)


def test_validate_model_returns_valid_for_unrestricted_adapter():
    """Test that validate_model returns valid for adapters not in registry."""
    config = _minimal_config({})  # Empty registry
    registry = ModelRegistry(config)

    # llamacpp is unrestricted (not in registry)
    result = registry.validate_model("llamacpp", "/path/to/any/model.gguf")

    assert result.valid is True
    assert result.exists is True
    assert result.enabled is True


def test_validate_model_error_message_lists_available_models():
    """Test that error message includes available models when no similar ones."""
    config = _minimal_config(
        {
            "test_adapter": [
                {"id": "alpha-model", "label": "Alpha", "enabled": True},
                {"id": "beta-model", "label": "Beta", "enabled": True},
                {"id": "gamma-model", "label": "Gamma", "enabled": True},
                {"id": "delta-model", "label": "Delta", "enabled": True},
            ]
        }
    )
    registry = ModelRegistry(config)

    # Use a model name that's very different from all options
    result = registry.validate_model("test_adapter", "completely-different-zzz-999")

    assert result.valid is False
    # Should list available models since no similar ones
    assert "Available models:" in result.error_message or "alpha" in result.error_message.lower()


def test_validate_model_max_suggestions_parameter():
    """Test that max_suggestions parameter limits the suggestions."""
    config = _minimal_config(
        {
            "test_adapter": [
                {"id": "model-aaa", "label": "Model AAA", "enabled": True},
                {"id": "model-aab", "label": "Model AAB", "enabled": True},
                {"id": "model-aac", "label": "Model AAC", "enabled": True},
                {"id": "model-aad", "label": "Model AAD", "enabled": True},
                {"id": "model-aae", "label": "Model AAE", "enabled": True},
            ]
        }
    )
    registry = ModelRegistry(config)

    # Request only 2 suggestions
    result = registry.validate_model("test_adapter", "model-aa", max_suggestions=2)

    assert len(result.similar_models) <= 2


def test_validate_model_case_insensitive_matching(config_for_validation: Config):
    """Test that similar model matching is case-insensitive."""
    registry = ModelRegistry(config_for_validation)

    # Use uppercase
    result = registry.validate_model("test_adapter", "CLAUDE-SONNET-4-5")

    assert result.valid is False
    # Should still find the lowercase version as similar
    assert len(result.similar_models) > 0


def test_validate_model_disabled_model_suggests_similar_enabled():
    """Test that for disabled models, we suggest similar enabled ones."""
    config = _minimal_config(
        {
            "test_adapter": [
                {
                    "id": "claude-3-5-sonnet-v1",
                    "label": "Claude 3.5 Sonnet v1",
                    "enabled": False,
                },
                {
                    "id": "claude-3-5-sonnet-v2",
                    "label": "Claude 3.5 Sonnet v2",
                    "enabled": True,
                },
            ]
        }
    )
    registry = ModelRegistry(config)

    result = registry.validate_model("test_adapter", "claude-3-5-sonnet-v1")

    assert result.valid is False
    assert result.exists is True
    assert result.enabled is False
    # Should suggest the enabled v2 version
    assert "claude-3-5-sonnet-v2" in result.similar_models


def test_model_validation_result_defaults():
    """Test ModelValidationResult default values."""
    result = ModelValidationResult(
        valid=False,
        model_id="test",
        adapter="test",
    )

    assert result.exists is False
    assert result.enabled is None
    assert result.similar_models == []
    assert result.error_message is None


# ============================================================================
# Tests for fallback_models feature (GitHub Issue #11)
# ============================================================================


@pytest.fixture
def config_with_fallback_models() -> Config:
    """Create a config with fallback models configured."""
    return _minimal_config(
        {
            "test_adapter": [
                {
                    "id": "primary-model",
                    "label": "Primary Model",
                    "enabled": True,
                    "default": True,
                    "fallback_models": ["fallback-1", "fallback-2", "fallback-3"],
                },
                {
                    "id": "fallback-1",
                    "label": "Fallback 1",
                    "enabled": True,
                    "fallback_models": ["fallback-2"],
                },
                {
                    "id": "fallback-2",
                    "label": "Fallback 2",
                    "enabled": True,
                    # No fallback_models - tests None case
                },
                {
                    "id": "fallback-3",
                    "label": "Fallback 3",
                    "enabled": True,
                    "fallback_models": [],  # Empty list
                },
            ]
        }
    )


def test_get_fallback_models_returns_configured_list(
    config_with_fallback_models: Config,
):
    """Test that get_fallback_models returns the configured fallback list."""
    registry = ModelRegistry(config_with_fallback_models)

    fallbacks = registry.get_fallback_models("test_adapter", "primary-model")

    assert fallbacks == ["fallback-1", "fallback-2", "fallback-3"]


def test_get_fallback_models_returns_empty_for_no_fallbacks(
    config_with_fallback_models: Config,
):
    """Test that get_fallback_models returns empty list when no fallbacks configured."""
    registry = ModelRegistry(config_with_fallback_models)

    # Model with no fallback_models field (None)
    fallbacks = registry.get_fallback_models("test_adapter", "fallback-2")
    assert fallbacks == []


def test_get_fallback_models_returns_empty_for_empty_list(
    config_with_fallback_models: Config,
):
    """Test that get_fallback_models returns empty list for empty fallback_models."""
    registry = ModelRegistry(config_with_fallback_models)

    # Model with empty fallback_models list
    fallbacks = registry.get_fallback_models("test_adapter", "fallback-3")
    assert fallbacks == []


def test_get_fallback_models_returns_empty_for_nonexistent_model(
    config_with_fallback_models: Config,
):
    """Test that get_fallback_models returns empty list for non-existent model."""
    registry = ModelRegistry(config_with_fallback_models)

    fallbacks = registry.get_fallback_models("test_adapter", "nonexistent-model")
    assert fallbacks == []


def test_get_fallback_models_returns_empty_for_nonexistent_adapter(
    config_with_fallback_models: Config,
):
    """Test that get_fallback_models returns empty list for non-existent adapter."""
    registry = ModelRegistry(config_with_fallback_models)

    fallbacks = registry.get_fallback_models("nonexistent-adapter", "primary-model")
    assert fallbacks == []


def test_fallback_models_stored_as_tuple_in_registry_entry(
    config_with_fallback_models: Config,
):
    """Test that fallback_models is stored as a tuple in RegistryEntry."""
    registry = ModelRegistry(config_with_fallback_models)

    models = registry.get_all_models("test_adapter")
    primary = next(m for m in models if m.id == "primary-model")

    # Should be stored as tuple (frozen dataclass requirement)
    assert isinstance(primary.fallback_models, tuple)
    assert primary.fallback_models == ("fallback-1", "fallback-2", "fallback-3")


def test_fallback_models_none_for_model_without_config(
    config_with_fallback_models: Config,
):
    """Test that fallback_models is None when not configured."""
    registry = ModelRegistry(config_with_fallback_models)

    models = registry.get_all_models("test_adapter")
    fallback2 = next(m for m in models if m.id == "fallback-2")

    assert fallback2.fallback_models is None


def test_get_fallback_models_with_single_fallback():
    """Test get_fallback_models with a single fallback configured."""
    config = _minimal_config(
        {
            "test_adapter": [
                {
                    "id": "single-fallback-model",
                    "label": "Model with Single Fallback",
                    "enabled": True,
                    "fallback_models": ["backup-model"],
                },
            ]
        }
    )
    registry = ModelRegistry(config)

    fallbacks = registry.get_fallback_models("test_adapter", "single-fallback-model")

    assert fallbacks == ["backup-model"]
    assert len(fallbacks) == 1


def test_fallback_models_preserves_order():
    """Test that fallback model order is preserved."""
    config = _minimal_config(
        {
            "test_adapter": [
                {
                    "id": "ordered-model",
                    "label": "Ordered Model",
                    "enabled": True,
                    "fallback_models": ["first", "second", "third", "fourth"],
                },
            ]
        }
    )
    registry = ModelRegistry(config)

    fallbacks = registry.get_fallback_models("test_adapter", "ordered-model")

    assert fallbacks == ["first", "second", "third", "fourth"]
    assert fallbacks[0] == "first"
    assert fallbacks[-1] == "fourth"
