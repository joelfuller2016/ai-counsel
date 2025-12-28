"""Model registry utilities derived from configuration."""
from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import builtins

from models.config import Config, ModelDefinition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegistryEntry:
    """Normalized model definition entry.

    Attributes:
        id: Unique model identifier used by adapter
        label: Human-friendly display name for UI dropdowns
        tier: Optional tier classification (speed, premium, coding, etc.)
        note: Optional descriptive text shown in model picker tooltips
        default: Whether marked as recommended default for this adapter
        enabled: Whether model is active and available for use
        timeout: Model-specific timeout in seconds (None = use adapter timeout)
        fallback_models: Ordered list of fallback model IDs (None = no fallbacks)
    """

    id: str
    label: str
    tier: Optional[str] = None
    note: Optional[str] = None
    default: bool = False
    enabled: bool = True
    timeout: Optional[int] = None
    fallback_models: Optional[tuple[str, ...]] = None


@dataclass
class ModelValidationResult:
    """Result of validating a model ID against the registry.

    Attributes:
        valid: Whether the model ID is valid and enabled
        model_id: The model ID that was validated
        adapter: The adapter name
        exists: Whether the model exists in registry (regardless of enabled state)
        enabled: Whether the model is enabled (None if doesn't exist)
        similar_models: List of similar model IDs (for suggestions)
        error_message: Human-readable error message if invalid
    """

    valid: bool
    model_id: str
    adapter: str
    exists: bool = False
    enabled: Optional[bool] = None
    similar_models: builtins.list[str] = None  # type: ignore[assignment]
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.similar_models is None:
            self.similar_models = []


class ModelRegistry:
    """In-memory view of configured model options per adapter."""

    def __init__(self, config: Config):
        self._entries: Dict[str, list[RegistryEntry]] = {}

        registry_cfg = getattr(config, "model_registry", None) or {}
        for cli, models in registry_cfg.items():
            normalized = []
            for model in models:
                # Pydantic ensures the structure, but guard against None just in case
                if isinstance(model, ModelDefinition):
                    model_def = model
                else:
                    model_def = ModelDefinition.model_validate(model)

                # Convert fallback_models list to tuple for frozen dataclass
                fallback_tuple = None
                if model_def.fallback_models:
                    fallback_tuple = tuple(model_def.fallback_models)

                normalized.append(
                    RegistryEntry(
                        id=model_def.id,
                        label=model_def.label or model_def.id,
                        tier=model_def.tier,
                        note=model_def.note,
                        default=bool(model_def.default),
                        enabled=bool(model_def.enabled),
                        timeout=model_def.timeout,
                        fallback_models=fallback_tuple,
                    )
                )

            # Ensure deterministic ordering (defaults first, then alphabetical)
            normalized.sort(
                key=lambda entry: (
                    not entry.default,
                    entry.label.lower(),
                )
            )
            self._entries[cli] = normalized

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def adapters(self) -> Iterable[str]:
        """Return adapter names with registry entries."""

        return self._entries.keys()

    def list(self) -> Dict[str, list[dict[str, str | bool]]]:
        """Return a serializable map of enabled model options by adapter."""

        result: Dict[str, list[dict[str, str | bool]]] = {}
        for cli, entries in self._entries.items():
            enabled_entries = [e for e in entries if e.enabled]
            result[cli] = [self._entry_to_dict(entry) for entry in enabled_entries]
        return result

    def list_for_adapter(self, cli: str) -> builtins.list[RegistryEntry]:
        """Return enabled entries for the given adapter (empty if none configured)."""

        entries = self._entries.get(cli, [])
        return [entry for entry in entries if entry.enabled]

    def get_all_models(self, cli: str) -> builtins.list[RegistryEntry]:
        """Return all entries for the given adapter, including disabled ones.

        Useful for administrative interfaces and debugging.
        """

        return builtins.list(self._entries.get(cli, []))

    def allowed_ids(self, cli: str) -> set[str]:
        """Return the set of allowed (enabled) model IDs for an adapter."""

        return {entry.id for entry in self._entries.get(cli, []) if entry.enabled}

    def get_default(self, cli: str) -> Optional[str]:
        """Return the default model id for an adapter, if configured.

        Only considers enabled models. If the marked default is disabled,
        returns the first enabled model. If no enabled models, returns None.
        """

        entries = self._entries.get(cli, [])
        if not entries:
            return None

        # Filter to enabled models only
        enabled_entries = [e for e in entries if e.enabled]
        if not enabled_entries:
            return None

        # Try to find the marked default among enabled models
        for entry in enabled_entries:
            if entry.default:
                return entry.id

        # Fallback to first enabled model
        # Check if we're skipping a disabled default (for operational visibility)
        all_defaults = [e for e in entries if e.default]
        if all_defaults and not all_defaults[0].enabled:
            logger.debug(
                f"Marked default '{all_defaults[0].id}' for adapter '{cli}' is disabled. "
                f"Falling back to first enabled model: '{enabled_entries[0].id}'"
            )
        
        return enabled_entries[0].id

    def is_allowed(self, cli: str, model_id: str) -> bool:
        """Check whether the given model id is allowlisted for the adapter."""

        if cli not in self._entries:
            return True  # Unrestricted adapter (e.g., open router, custom paths)
        return model_id in self.allowed_ids(cli)

    def get_model_timeout(self, cli: str, model_id: str) -> Optional[int]:
        """Get model-specific timeout for a given adapter and model.

        Args:
            cli: Adapter name (e.g., 'claude', 'codex')
            model_id: Model identifier

        Returns:
            Model-specific timeout in seconds, or None if not configured
            (meaning the adapter timeout should be used).
        """
        entries = self._entries.get(cli, [])
        for entry in entries:
            if entry.id == model_id:
                return entry.timeout
        return None

    def get_fallback_models(self, cli: str, model_id: str) -> builtins.list[str]:
        """Get ordered list of fallback model IDs for a given model.

        Args:
            cli: Adapter name (e.g., 'openrouter')
            model_id: Primary model identifier

        Returns:
            List of fallback model IDs in order of preference,
            or empty list if no fallbacks configured.
        """
        entries = self._entries.get(cli, [])
        for entry in entries:
            if entry.id == model_id:
                if entry.fallback_models:
                    return builtins.list(entry.fallback_models)
                return []
        return []

    def validate_model(
        self, cli: str, model_id: str, max_suggestions: int = 3
    ) -> ModelValidationResult:
        """Validate a model ID and provide detailed feedback.

        Args:
            cli: Adapter name
            model_id: Model ID to validate
            max_suggestions: Maximum number of similar model suggestions

        Returns:
            ModelValidationResult with validation details and suggestions
        """
        # If adapter is not in registry, it's unrestricted
        if cli not in self._entries:
            return ModelValidationResult(
                valid=True,
                model_id=model_id,
                adapter=cli,
                exists=True,
                enabled=True,
            )

        all_models = self._entries.get(cli, [])
        all_ids = [e.id for e in all_models]
        enabled_ids = [e.id for e in all_models if e.enabled]

        # Check if model exists
        matching_entry = next((e for e in all_models if e.id == model_id), None)

        if matching_entry:
            # Model exists - check if enabled
            if matching_entry.enabled:
                return ModelValidationResult(
                    valid=True,
                    model_id=model_id,
                    adapter=cli,
                    exists=True,
                    enabled=True,
                )
            else:
                # Model exists but is disabled
                similar = self._find_similar_models(
                    model_id, enabled_ids, max_suggestions
                )
                return ModelValidationResult(
                    valid=False,
                    model_id=model_id,
                    adapter=cli,
                    exists=True,
                    enabled=False,
                    similar_models=similar,
                    error_message=self._build_disabled_error(
                        model_id, cli, enabled_ids, similar
                    ),
                )
        else:
            # Model doesn't exist - find similar ones
            similar = self._find_similar_models(model_id, enabled_ids, max_suggestions)
            return ModelValidationResult(
                valid=False,
                model_id=model_id,
                adapter=cli,
                exists=False,
                enabled=None,
                similar_models=similar,
                error_message=self._build_not_found_error(
                    model_id, cli, enabled_ids, similar
                ),
            )

    def _find_similar_models(
        self, model_id: str, candidate_ids: builtins.list[str], max_results: int = 3
    ) -> builtins.list[str]:
        """Find model IDs similar to the given one using fuzzy matching.

        Args:
            model_id: The model ID to match against
            candidate_ids: List of valid model IDs to search
            max_results: Maximum number of suggestions to return

        Returns:
            List of similar model IDs, ordered by similarity (best first)
        """
        if not candidate_ids:
            return []

        # Use difflib to find close matches
        # cutoff=0.4 allows for moderate typos/variations
        matches = difflib.get_close_matches(
            model_id.lower(),
            [m.lower() for m in candidate_ids],
            n=max_results,
            cutoff=0.4,
        )

        # Map back to original case
        lower_to_original = {m.lower(): m for m in candidate_ids}
        return [lower_to_original[m] for m in matches]

    def _build_disabled_error(
        self,
        model_id: str,
        cli: str,
        enabled_ids: builtins.list[str],
        similar: builtins.list[str],
    ) -> str:
        """Build error message for a disabled model."""
        msg = f"Model '{model_id}' exists but is disabled for adapter '{cli}'."

        if similar:
            msg += f" Similar enabled models: {', '.join(similar)}."
        elif enabled_ids:
            # Show first few enabled models if no similar ones found
            preview = enabled_ids[:3]
            msg += f" Available models: {', '.join(preview)}"
            if len(enabled_ids) > 3:
                msg += f" (+{len(enabled_ids) - 3} more)"
            msg += "."

        msg += " Use 'list_models' tool to see all available models."
        return msg

    def _build_not_found_error(
        self,
        model_id: str,
        cli: str,
        enabled_ids: builtins.list[str],
        similar: builtins.list[str],
    ) -> str:
        """Build error message for a model that doesn't exist."""
        msg = f"Model '{model_id}' not found for adapter '{cli}'."

        if similar:
            msg += f" Did you mean: {', '.join(similar)}?"
        elif enabled_ids:
            # Show first few available models
            preview = enabled_ids[:3]
            msg += f" Available models: {', '.join(preview)}"
            if len(enabled_ids) > 3:
                msg += f" (+{len(enabled_ids) - 3} more)"
            msg += "."

        msg += " Use 'list_models' tool to see all available models."
        return msg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _entry_to_dict(
        entry: RegistryEntry, include_enabled: bool = False
    ) -> dict[str, str | bool]:
        """Serialize an entry for MCP responses.

        Args:
            entry: Registry entry to serialize
            include_enabled: Whether to include the enabled status (useful for admin interfaces)
        """

        payload: dict[str, str | bool] = {
            "id": entry.id,
            "label": entry.label,
        }
        if entry.tier:
            payload["tier"] = entry.tier
        if entry.note:
            payload["note"] = entry.note
        if entry.default:
            payload["default"] = True
        if include_enabled:
            payload["enabled"] = entry.enabled
        return payload
