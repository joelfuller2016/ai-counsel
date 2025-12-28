"""OpenRouter model discovery for dynamic model fetching.

This module provides functionality to:
1. Fetch available models from OpenRouter's /api/v1/models endpoint
2. Parse model capabilities, pricing, and context length
3. Filter for free models (pricing.prompt = 0)
4. Auto-update the config.yaml free models list

API Reference: https://openrouter.ai/api/v1/models
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterModelInfo:
    """Parsed model information from OpenRouter API."""

    id: str
    name: str
    description: str = ""
    context_length: int = 0
    pricing_prompt: float = 0.0  # Price per 1M tokens for input
    pricing_completion: float = 0.0  # Price per 1M tokens for output
    top_provider: Optional[str] = None
    is_free: bool = False
    architecture: Optional[str] = None
    modality: str = "text"  # text, multimodal, etc.
    supported_parameters: list[str] = field(default_factory=list)

    @classmethod
    def from_api_response(cls, data: dict) -> "OpenRouterModelInfo":
        """Parse model data from OpenRouter API response.

        API response structure:
        {
            "id": "meta-llama/llama-3.3-70b-instruct:free",
            "name": "Meta: Llama 3.3 70B Instruct (free)",
            "description": "...",
            "context_length": 131072,
            "pricing": {
                "prompt": "0",
                "completion": "0",
                "image": "0",
                "request": "0"
            },
            "top_provider": {
                "context_length": 131072,
                "max_completion_tokens": 8192,
                ...
            },
            "architecture": {
                "modality": "text->text",
                "tokenizer": "Llama3",
                "instruct_type": "llama3"
            },
            "supported_parameters": ["temperature", "top_p", ...]
        }
        """
        pricing = data.get("pricing", {})

        # Parse pricing - can be string or float
        def parse_price(val) -> float:
            if val is None:
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        pricing_prompt = parse_price(pricing.get("prompt", 0))
        pricing_completion = parse_price(pricing.get("completion", 0))

        # Check if model is free (zero cost for both prompt and completion)
        is_free = pricing_prompt == 0 and pricing_completion == 0

        # Parse architecture
        architecture = data.get("architecture", {})
        modality = (
            architecture.get("modality", "text->text")
            if isinstance(architecture, dict)
            else "text->text"
        )

        # Parse top provider info
        top_provider = data.get("top_provider", {})
        top_provider_str = None
        if isinstance(top_provider, dict) and top_provider:
            # Include context length from top provider if available
            if "context_length" in top_provider:
                top_provider_str = f"context={top_provider['context_length']}"

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            context_length=data.get("context_length", 0),
            pricing_prompt=pricing_prompt,
            pricing_completion=pricing_completion,
            top_provider=top_provider_str,
            is_free=is_free,
            architecture=(
                architecture.get("tokenizer")
                if isinstance(architecture, dict)
                else None
            ),
            modality=modality,
            supported_parameters=data.get("supported_parameters", []),
        )

    def to_config_entry(self, tier: str = "free", timeout: int = 60) -> dict:
        """Convert to config.yaml model_registry entry format.

        Returns:
            dict with id, label, tier, enabled, timeout
        """
        return {
            "id": self.id,
            "label": self._generate_label(),
            "tier": tier,
            "enabled": True,
            "timeout": timeout,
        }

    def _generate_label(self) -> str:
        """Generate a human-readable label from model name."""
        # Clean up the name: remove provider prefix, "(free)" suffix
        label = self.name
        if label.endswith("(free)"):
            label = label[:-6].strip()
        # Keep it concise
        if len(label) > 50:
            label = label[:47] + "..."
        return f"{label} FREE"


@dataclass
class ModelDiscoveryResult:
    """Result of model discovery operation."""

    models: list[OpenRouterModelInfo]
    total_count: int
    free_count: int
    timestamp: str
    error: Optional[str] = None


class OpenRouterModelDiscovery:
    """Discover available models from OpenRouter API.

    Usage:
        discovery = OpenRouterModelDiscovery(api_key="sk-or-...")
        result = await discovery.fetch_models()
        free_models = discovery.filter_free_models(result.models)
    """

    MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize model discovery.

        Args:
            api_key: OpenRouter API key (optional - models endpoint is public)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout

    async def fetch_models(self) -> ModelDiscoveryResult:
        """Fetch all available models from OpenRouter.

        Returns:
            ModelDiscoveryResult with parsed model information

        Raises:
            httpx.HTTPStatusError: On API error
            httpx.NetworkError: On network error
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"Fetching models from {self.MODELS_ENDPOINT}")
                response = await client.get(self.MODELS_ENDPOINT, headers=headers)
                response.raise_for_status()

                data = response.json()
                models_data = data.get("data", [])

                logger.info(f"Received {len(models_data)} models from OpenRouter")

                # Parse all models
                models = []
                for model_data in models_data:
                    try:
                        model_info = OpenRouterModelInfo.from_api_response(model_data)
                        models.append(model_info)
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse model {model_data.get('id', 'unknown')}: {e}"
                        )

                # Count free models
                free_count = sum(1 for m in models if m.is_free)

                return ModelDiscoveryResult(
                    models=models,
                    total_count=len(models),
                    free_count=free_count,
                    timestamp=datetime.now().isoformat(),
                )

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching models: {e.response.status_code}")
            return ModelDiscoveryResult(
                models=[],
                total_count=0,
                free_count=0,
                timestamp=datetime.now().isoformat(),
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return ModelDiscoveryResult(
                models=[],
                total_count=0,
                free_count=0,
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )

    def filter_free_models(
        self, models: list[OpenRouterModelInfo]
    ) -> list[OpenRouterModelInfo]:
        """Filter models to only include free models.

        Args:
            models: List of model info objects

        Returns:
            List of free models (pricing.prompt = 0)
        """
        free_models = [m for m in models if m.is_free]
        logger.info(f"Filtered to {len(free_models)} free models")
        return free_models

    def categorize_free_models(
        self, models: list[OpenRouterModelInfo]
    ) -> dict[str, list[OpenRouterModelInfo]]:
        """Categorize free models by type/capability.

        Args:
            models: List of free models

        Returns:
            Dict with categories: reasoning, coding, fast, reliable, multimodal
        """
        categories: dict[str, list[OpenRouterModelInfo]] = {
            "free-reasoning": [],
            "free-coding": [],
            "free-fast": [],
            "free-reliable": [],
            "free-multimodal": [],
        }

        for model in models:
            if not model.is_free:
                continue

            model_id_lower = model.id.lower()
            name_lower = model.name.lower()

            # Categorize based on model characteristics
            if any(
                kw in model_id_lower or kw in name_lower
                for kw in ["deepseek-r1", "qwq", "reasoning"]
            ):
                categories["free-reasoning"].append(model)
            elif any(
                kw in model_id_lower or kw in name_lower
                for kw in ["code", "coder", "devstral", "starcoder"]
            ):
                categories["free-coding"].append(model)
            elif any(
                kw in model_id_lower or kw in name_lower
                for kw in ["flash", "mini", "scout", "small"]
            ):
                categories["free-fast"].append(model)
            elif "multimodal" in model.modality or "image" in model.modality:
                categories["free-multimodal"].append(model)
            elif model.context_length >= 100000:
                # Large context models are generally more reliable
                categories["free-reliable"].append(model)
            else:
                # Default to fast category
                categories["free-fast"].append(model)

        return categories

    def generate_timeout(self, model: OpenRouterModelInfo) -> int:
        """Generate appropriate timeout based on model characteristics.

        Args:
            model: Model info

        Returns:
            Timeout in seconds
        """
        model_id_lower = model.id.lower()
        name_lower = model.name.lower()

        # Reasoning models need longer timeouts
        if any(
            kw in model_id_lower or kw in name_lower
            for kw in ["deepseek-r1", "qwq", "reasoning"]
        ):
            return 120

        # Large models (70B+) need more time
        if any(
            kw in model_id_lower or kw in name_lower for kw in ["70b", "72b", "65b"]
        ):
            return 60

        # Flash/mini models are fast
        if any(
            kw in model_id_lower or kw in name_lower
            for kw in ["flash", "mini", "scout", "small"]
        ):
            return 30

        # Default timeout
        return 45


def fetch_models_sync(api_key: Optional[str] = None) -> ModelDiscoveryResult:
    """Synchronous wrapper for fetch_models.

    Convenience function for CLI usage.

    Args:
        api_key: Optional OpenRouter API key

    Returns:
        ModelDiscoveryResult with discovered models
    """
    import asyncio

    discovery = OpenRouterModelDiscovery(api_key=api_key)
    return asyncio.run(discovery.fetch_models())
