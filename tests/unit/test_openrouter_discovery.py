"""Unit tests for OpenRouter model discovery.

Tests for:
- OpenRouterModelInfo.from_api_response() parsing
- OpenRouterModelDiscovery.fetch_models()
- OpenRouterModelDiscovery.filter_free_models()
- OpenRouterModelDiscovery.categorize_free_models()
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from models.openrouter_discovery import (
    OpenRouterModelDiscovery,
    OpenRouterModelInfo,
    ModelDiscoveryResult,
    fetch_models_sync,
)


class TestOpenRouterModelInfo:
    """Tests for OpenRouterModelInfo dataclass."""

    def test_from_api_response_basic(self):
        """Test parsing basic model info from API response."""
        data = {
            "id": "meta-llama/llama-3.3-70b-instruct:free",
            "name": "Meta: Llama 3.3 70B Instruct (free)",
            "description": "Large language model",
            "context_length": 131072,
            "pricing": {
                "prompt": "0",
                "completion": "0",
            },
        }

        model = OpenRouterModelInfo.from_api_response(data)

        assert model.id == "meta-llama/llama-3.3-70b-instruct:free"
        assert model.name == "Meta: Llama 3.3 70B Instruct (free)"
        assert model.description == "Large language model"
        assert model.context_length == 131072
        assert model.pricing_prompt == 0.0
        assert model.pricing_completion == 0.0
        assert model.is_free is True

    def test_from_api_response_paid_model(self):
        """Test parsing paid model (non-zero pricing)."""
        data = {
            "id": "openai/gpt-4",
            "name": "GPT-4",
            "pricing": {
                "prompt": "0.00003",
                "completion": "0.00006",
            },
        }

        model = OpenRouterModelInfo.from_api_response(data)

        assert model.is_free is False
        assert model.pricing_prompt == 0.00003
        assert model.pricing_completion == 0.00006

    def test_from_api_response_with_architecture(self):
        """Test parsing model with architecture info."""
        data = {
            "id": "test-model",
            "name": "Test Model",
            "architecture": {
                "modality": "text->text",
                "tokenizer": "Llama3",
                "instruct_type": "llama3",
            },
            "pricing": {"prompt": "0", "completion": "0"},
        }

        model = OpenRouterModelInfo.from_api_response(data)

        assert model.modality == "text->text"
        assert model.architecture == "Llama3"

    def test_from_api_response_with_top_provider(self):
        """Test parsing model with top provider info."""
        data = {
            "id": "test-model",
            "name": "Test Model",
            "top_provider": {
                "context_length": 8192,
                "max_completion_tokens": 4096,
            },
            "pricing": {"prompt": "0", "completion": "0"},
        }

        model = OpenRouterModelInfo.from_api_response(data)

        assert model.top_provider == "context=8192"

    def test_from_api_response_numeric_pricing(self):
        """Test parsing when pricing values are numeric (not strings)."""
        data = {
            "id": "test-model",
            "name": "Test Model",
            "pricing": {
                "prompt": 0.0,
                "completion": 0.0,
            },
        }

        model = OpenRouterModelInfo.from_api_response(data)

        assert model.is_free is True
        assert model.pricing_prompt == 0.0

    def test_from_api_response_missing_fields(self):
        """Test parsing with missing optional fields."""
        data = {
            "id": "minimal-model",
        }

        model = OpenRouterModelInfo.from_api_response(data)

        assert model.id == "minimal-model"
        assert model.name == ""
        assert model.description == ""
        assert model.context_length == 0
        assert model.is_free is True  # No pricing = free

    def test_to_config_entry(self):
        """Test generating config entry from model info."""
        model = OpenRouterModelInfo(
            id="meta-llama/llama-3.3-70b-instruct:free",
            name="Meta: Llama 3.3 70B Instruct (free)",
            context_length=131072,
            is_free=True,
        )

        entry = model.to_config_entry(tier="free-reliable", timeout=60)

        assert entry["id"] == "meta-llama/llama-3.3-70b-instruct:free"
        assert entry["tier"] == "free-reliable"
        assert entry["enabled"] is True
        assert entry["timeout"] == 60
        assert "FREE" in entry["label"]

    def test_generate_label_removes_free_suffix(self):
        """Test that (free) suffix is removed from label."""
        model = OpenRouterModelInfo(
            id="test",
            name="Model Name (free)",
            is_free=True,
        )

        label = model._generate_label()

        assert label == "Model Name FREE"
        assert "(free)" not in label

    def test_generate_label_truncates_long_names(self):
        """Test that long names are truncated."""
        model = OpenRouterModelInfo(
            id="test",
            name="A" * 60,  # Very long name
            is_free=True,
        )

        label = model._generate_label()

        assert len(label) <= 55  # 47 chars + "..." + " FREE"


class TestOpenRouterModelDiscovery:
    """Tests for OpenRouterModelDiscovery class."""

    @pytest.mark.asyncio
    async def test_fetch_models_success(self):
        """Test successful model fetching."""
        mock_response_data = {
            "data": [
                {
                    "id": "model-1:free",
                    "name": "Model 1 (free)",
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "model-2",
                    "name": "Model 2",
                    "pricing": {"prompt": "0.001", "completion": "0.002"},
                },
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            discovery = OpenRouterModelDiscovery()
            result = await discovery.fetch_models()

        assert result.error is None
        assert result.total_count == 2
        assert result.free_count == 1
        assert len(result.models) == 2
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_fetch_models_http_error(self):
        """Test handling of HTTP errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server error",
                request=MagicMock(),
                response=mock_response,
            )
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            discovery = OpenRouterModelDiscovery()
            result = await discovery.fetch_models()

        assert result.error is not None
        assert "500" in result.error
        assert result.total_count == 0
        assert len(result.models) == 0

    @pytest.mark.asyncio
    async def test_fetch_models_with_api_key(self):
        """Test that API key is included in request headers."""
        mock_response_data = {"data": []}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            discovery = OpenRouterModelDiscovery(api_key="sk-or-test-key")
            await discovery.fetch_models()

            # Check that headers include Authorization
            call_kwargs = mock_client.get.call_args
            headers = call_kwargs.kwargs.get("headers", {})
            assert headers.get("Authorization") == "Bearer sk-or-test-key"

    def test_filter_free_models(self):
        """Test filtering models to only free ones."""
        models = [
            OpenRouterModelInfo(id="free-1", name="Free 1", is_free=True),
            OpenRouterModelInfo(id="paid-1", name="Paid 1", is_free=False),
            OpenRouterModelInfo(id="free-2", name="Free 2", is_free=True),
        ]

        discovery = OpenRouterModelDiscovery()
        free_models = discovery.filter_free_models(models)

        assert len(free_models) == 2
        assert all(m.is_free for m in free_models)

    def test_categorize_free_models_reasoning(self):
        """Test categorization of reasoning models."""
        models = [
            OpenRouterModelInfo(
                id="deepseek/deepseek-r1:free",
                name="DeepSeek R1",
                is_free=True,
            ),
            OpenRouterModelInfo(
                id="qwen/qwq-32b:free",
                name="QwQ 32B",
                is_free=True,
            ),
        ]

        discovery = OpenRouterModelDiscovery()
        categories = discovery.categorize_free_models(models)

        assert len(categories["free-reasoning"]) == 2

    def test_categorize_free_models_coding(self):
        """Test categorization of coding models."""
        models = [
            OpenRouterModelInfo(
                id="deepseek/deepseek-coder:free",
                name="DeepSeek Coder",
                is_free=True,
            ),
            OpenRouterModelInfo(
                id="mistral/devstral:free",
                name="Devstral",
                is_free=True,
            ),
        ]

        discovery = OpenRouterModelDiscovery()
        categories = discovery.categorize_free_models(models)

        assert len(categories["free-coding"]) == 2

    def test_categorize_free_models_fast(self):
        """Test categorization of fast models."""
        models = [
            OpenRouterModelInfo(
                id="google/gemini-flash:free",
                name="Gemini Flash",
                is_free=True,
            ),
            OpenRouterModelInfo(
                id="meta-llama/llama-3-8b-instruct:free",
                name="Llama 3 8B Mini",
                is_free=True,
            ),
        ]

        discovery = OpenRouterModelDiscovery()
        categories = discovery.categorize_free_models(models)

        assert len(categories["free-fast"]) >= 1

    def test_categorize_free_models_reliable(self):
        """Test categorization of reliable models (large context)."""
        models = [
            OpenRouterModelInfo(
                id="model/large-context:free",
                name="Large Context Model",
                context_length=200000,
                is_free=True,
            ),
        ]

        discovery = OpenRouterModelDiscovery()
        categories = discovery.categorize_free_models(models)

        assert len(categories["free-reliable"]) == 1

    def test_generate_timeout_reasoning_models(self):
        """Test timeout generation for reasoning models."""
        discovery = OpenRouterModelDiscovery()

        model = OpenRouterModelInfo(
            id="deepseek/deepseek-r1:free",
            name="DeepSeek R1",
            is_free=True,
        )

        timeout = discovery.generate_timeout(model)

        assert timeout == 120

    def test_generate_timeout_large_models(self):
        """Test timeout generation for large models."""
        discovery = OpenRouterModelDiscovery()

        model = OpenRouterModelInfo(
            id="meta-llama/llama-3.3-70b-instruct:free",
            name="Llama 70B",
            is_free=True,
        )

        timeout = discovery.generate_timeout(model)

        assert timeout == 60

    def test_generate_timeout_fast_models(self):
        """Test timeout generation for fast models."""
        discovery = OpenRouterModelDiscovery()

        model = OpenRouterModelInfo(
            id="google/gemini-flash:free",
            name="Gemini Flash",
            is_free=True,
        )

        timeout = discovery.generate_timeout(model)

        assert timeout == 30


class TestFetchModelsSync:
    """Tests for synchronous fetch_models_sync helper."""

    def test_fetch_models_sync(self):
        """Test synchronous wrapper for fetch_models."""
        mock_response_data = {
            "data": [
                {
                    "id": "test-model:free",
                    "name": "Test Model (free)",
                    "pricing": {"prompt": "0", "completion": "0"},
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = fetch_models_sync()

        assert result.total_count == 1
        assert result.free_count == 1
