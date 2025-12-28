"""Base HTTP adapter with request/retry management."""
import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from tenacity import (retry, retry_if_exception, stop_after_attempt,
                      wait_exponential)

# Configure progress logger for HTTP adapter debugging
progress_logger = logging.getLogger("ai_counsel.progress")
if not progress_logger.handlers:
    # Log to both console and dedicated progress file
    project_dir = Path(__file__).parent.parent
    progress_file = project_dir / "deliberation_progress.log"
    progress_handler = logging.FileHandler(
        progress_file, mode="a", encoding="utf-8"
    )
    progress_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    ))
    progress_logger.addHandler(progress_handler)
    progress_logger.setLevel(logging.DEBUG)


@dataclass
class HealthCheckResult:
    """Result of a health check for an adapter/model combination.

    Attributes:
        available: Whether the model is available and responding
        latency_ms: Response latency in milliseconds (None if check failed)
        error: Error message if health check failed (None if successful)
        model: The model identifier that was checked
        adapter: The adapter name that was checked
    """

    available: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    model: Optional[str] = None
    adapter: Optional[str] = None


def is_retryable_http_error(exception):
    """
    Determine if an HTTP error should be retried.

    Retries on:
    - 5xx server errors
    - 429 rate limit errors
    - Network errors (connection, timeout)

    Does NOT retry on:
    - 4xx client errors (bad request, auth, etc.)

    Args:
        exception: The exception to check

    Returns:
        bool: True if the error should be retried
    """
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on 5xx server errors and 429 rate limit
        return (
            exception.response.status_code >= 500
            or exception.response.status_code == 429
        )

    # Retry on network errors
    return isinstance(
        exception, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)
    )


class BaseHTTPAdapter(ABC):
    """
    Abstract base class for HTTP API adapters.

    Handles HTTP requests, timeout management, retry logic with exponential backoff,
    and error handling. Subclasses must implement build_request() and parse_response()
    for API-specific logic.

    Example:
        class MyAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/generate", {"Content-Type": "application/json"}, {"prompt": prompt})

            def parse_response(self, response_json):
                return response_json["text"]

        adapter = MyAdapter(base_url="http://localhost:8080", timeout=60)
        result = await adapter.invoke(prompt="Hello", model="my-model")
    """

    # Default prompt length limits per HTTP adapter type (in characters)
    # These are conservative limits to prevent API rejection errors
    DEFAULT_PROMPT_LIMITS: dict[str, int] = {
        "ollama": 50_000,      # Local models typically have smaller context
        "lmstudio": 50_000,    # Local models typically have smaller context
        "openrouter": 100_000, # OpenRouter supports various models
    }

    # Adapter name for logging and limit lookup (subclasses should override)
    ADAPTER_NAME: str = "http"

    def __init__(
        self,
        base_url: str,
        timeout: int = 60,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        max_prompt_length: Optional[int] = None,
    ):
        """
        Initialize HTTP adapter.

        Args:
            base_url: Base URL for API (e.g., "http://localhost:11434")
            timeout: Request timeout in seconds (default: 60)
            max_retries: Maximum retry attempts for transient failures (default: 3)
            api_key: Optional API key for authentication
            headers: Optional default headers to include in all requests
            max_prompt_length: Maximum prompt length in characters. If not specified,
                uses adapter-specific default from DEFAULT_PROMPT_LIMITS.
        """
        self.base_url = base_url.rstrip("/")  # Remove trailing slash
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key
        self.default_headers = headers or {}
        self._max_prompt_length = max_prompt_length

    @property
    def max_prompt_length(self) -> int:
        """
        Get the maximum prompt length for this adapter.

        Returns configured value if set, otherwise looks up default by adapter name.
        Falls back to 100,000 characters if no specific default exists.
        """
        if self._max_prompt_length is not None:
            return self._max_prompt_length
        return self.DEFAULT_PROMPT_LIMITS.get(self.ADAPTER_NAME, 100_000)

    def validate_prompt_length(self, prompt: str) -> bool:
        """
        Validate that prompt length is within allowed limits.

        Args:
            prompt: The prompt text to validate

        Returns:
            True if prompt is valid length, False if too long
        """
        return len(prompt) <= self.max_prompt_length

    @abstractmethod
    def build_request(
        self, model: str, prompt: str
    ) -> Tuple[str, dict[str, str], dict]:
        """
        Build API-specific request components.

        Args:
            model: Model identifier
            prompt: The prompt to send

        Returns:
            Tuple of (endpoint, headers, body):
            - endpoint: Full URL path (e.g., "/api/generate")
            - headers: Request headers dict
            - body: Request body dict (will be JSON-encoded)
        """
        pass

    @abstractmethod
    def parse_response(self, response_json: dict) -> str:
        """
        Parse API-specific response to extract model output.

        Args:
            response_json: Parsed JSON response from API

        Returns:
            Extracted model response text
        """
        pass

    async def health_check(
        self, model: str, timeout: Optional[float] = None
    ) -> HealthCheckResult:
        """
        Check if a model is available and responding.

        Sends a minimal request to verify the model can respond. This is useful
        for checking model availability before starting a deliberation, especially
        for free-tier models that may have rate limits or availability issues.

        Subclasses can override this method to implement provider-specific health
        checks (e.g., using a dedicated /models endpoint or /health endpoint).

        Default implementation sends a minimal prompt and checks for a valid response.

        Args:
            model: Model identifier to check
            timeout: Optional timeout in seconds (defaults to 10s for health checks)

        Returns:
            HealthCheckResult with availability status, latency, and any errors
        """
        check_timeout = timeout if timeout is not None else 10.0
        start_time = datetime.now()

        try:
            # Build a minimal health check request
            endpoint, headers, body = self.build_request(model, "Hi")
            # Override max_tokens to get fastest possible response
            if "max_tokens" in body:
                body["max_tokens"] = 1

            full_url = f"{self.base_url}{endpoint}"

            async with httpx.AsyncClient(timeout=check_timeout) as client:
                response = await client.post(full_url, headers=headers, json=body)
                response.raise_for_status()

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            return HealthCheckResult(
                available=True,
                latency_ms=latency_ms,
                model=model,
                adapter=self.ADAPTER_NAME,
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = f"HTTP {e.response.status_code}"

            # Try to extract error details from response
            try:
                error_body = e.response.json()
                if "error" in error_body:
                    if isinstance(error_body["error"], dict):
                        error_msg = f"{error_msg}: {error_body['error'].get('message', str(error_body['error']))}"
                    else:
                        error_msg = f"{error_msg}: {error_body['error']}"
            except Exception:
                pass

            return HealthCheckResult(
                available=False,
                latency_ms=latency_ms,
                error=error_msg,
                model=model,
                adapter=self.ADAPTER_NAME,
            )

        except httpx.TimeoutException:
            return HealthCheckResult(
                available=False,
                error=f"Timeout after {check_timeout}s",
                model=model,
                adapter=self.ADAPTER_NAME,
            )

        except Exception as e:
            return HealthCheckResult(
                available=False,
                error=f"{type(e).__name__}: {str(e)}",
                model=model,
                adapter=self.ADAPTER_NAME,
            )

    async def invoke(
        self,
        prompt: str,
        model: str,
        context: Optional[str] = None,
        is_deliberation: bool = True,
        working_directory: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        timeout_override: Optional[int] = None,
    ) -> str:
        """
        Invoke the HTTP API with the given prompt and model.

        Args:
            prompt: The prompt to send to the model
            model: Model identifier
            context: Optional additional context to prepend to prompt
            is_deliberation: Whether this is part of a deliberation (unused for HTTP,
                           kept for API compatibility with BaseCLIAdapter)
            working_directory: Unused for HTTP adapters (kept for API compatibility)
            reasoning_effort: Unused for HTTP adapters (kept for API compatibility)
            timeout_override: Optional model-specific timeout in seconds. If provided,
                overrides the adapter's default timeout. Useful for reasoning models
                that require longer response times.

        Returns:
            Parsed response from the model

        Raises:
            TimeoutError: If request exceeds timeout
            httpx.HTTPStatusError: If API returns error status
            RuntimeError: If retries exhausted
        """
        # Use model-specific timeout if provided, otherwise use adapter default
        effective_timeout = timeout_override if timeout_override is not None else self.timeout
        # Build full prompt
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\n{prompt}"

        # Validate prompt length
        if not self.validate_prompt_length(full_prompt):
            raise ValueError(
                f"Prompt too long ({len(full_prompt):,} characters). "
                f"Maximum allowed for {self.ADAPTER_NAME}: {self.max_prompt_length:,} characters. "
                f"Consider reducing context or breaking into smaller chunks. "
                f"This validation prevents API rejection errors."
            )

        # Get request components from subclass
        endpoint, headers, body = self.build_request(model, full_prompt)

        # Build full URL
        full_url = f"{self.base_url}{endpoint}"

        # Log request details for debugging
        logger = logging.getLogger(__name__)
        body_str = json.dumps(body, default=str)

        # Enhanced progress logging
        progress_logger.info(f"[START] HTTP REQUEST | Model: {model} | URL: {full_url}")
        progress_logger.debug(f"   API Key present: {bool(self.api_key)}")
        progress_logger.debug(f"   Prompt length: {len(full_prompt)} chars")
        progress_logger.debug(f"   Body size: {len(body_str)} bytes")
        progress_logger.debug(f"   Headers: {list(headers.keys())}")
        timeout_info = f" (model override)" if timeout_override else ""
        progress_logger.debug(f"   Timeout: {effective_timeout}s{timeout_info}")

        start_time = datetime.now()

        # Execute request with retry logic
        try:
            response_json = await self._execute_request_with_retry(
                url=full_url, headers=headers, body=body, timeout=effective_timeout
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            progress_logger.info(f"[SUCCESS] HTTP REQUEST | Model: {model} | Time: {elapsed:.2f}s")
            progress_logger.debug(f"   Response keys: {list(response_json.keys()) if isinstance(response_json, dict) else 'N/A'}")
            return self.parse_response(response_json)

        except asyncio.TimeoutError:
            elapsed = (datetime.now() - start_time).total_seconds()
            progress_logger.error(f"[TIMEOUT] HTTP REQUEST | Model: {model} | Time: {elapsed:.2f}s")
            raise TimeoutError(f"HTTP request timed out after {effective_timeout}s")

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            progress_logger.error(f"[ERROR] HTTP REQUEST FAILED | Model: {model} | Time: {elapsed:.2f}s | Error: {type(e).__name__}: {str(e)[:200]}")
            raise

    async def invoke_with_fallback(
        self,
        prompt: str,
        model: str,
        fallback_models: List[str],
        context: Optional[str] = None,
        is_deliberation: bool = True,
        working_directory: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        timeout_override: Optional[int] = None,
    ) -> str:
        """
        Invoke the HTTP API with fallback chain support.

        Attempts the primary model first. If it fails with a retryable error
        (timeout, rate limit, server error), tries each fallback model in order.
        Stops and returns the response from the first successful model.

        Args:
            prompt: The prompt to send to the model
            model: Primary model identifier
            fallback_models: Ordered list of fallback model IDs to try on failure
            context: Optional additional context to prepend to prompt
            is_deliberation: Whether this is part of a deliberation
            working_directory: Unused for HTTP adapters
            reasoning_effort: Unused for HTTP adapters
            timeout_override: Optional model-specific timeout in seconds

        Returns:
            Parsed response from the first successful model (primary or fallback)

        Raises:
            Exception: If all models (primary + fallbacks) fail, raises the last error
        """
        all_models = [model] + list(fallback_models)
        last_error: Optional[Exception] = None

        for idx, current_model in enumerate(all_models):
            is_fallback = idx > 0
            model_type = "FALLBACK" if is_fallback else "PRIMARY"

            if is_fallback:
                progress_logger.warning(
                    f"[FALLBACK] Trying fallback model {idx}/{len(fallback_models)}: "
                    f"{current_model} (after {type(last_error).__name__})"
                )

            try:
                result = await self.invoke(
                    prompt=prompt,
                    model=current_model,
                    context=context,
                    is_deliberation=is_deliberation,
                    working_directory=working_directory,
                    reasoning_effort=reasoning_effort,
                    timeout_override=timeout_override,
                )

                if is_fallback:
                    progress_logger.info(
                        f"[FALLBACK_SUCCESS] Model {current_model} succeeded "
                        f"(fallback {idx}/{len(fallback_models)})"
                    )

                return result

            except (TimeoutError, httpx.HTTPStatusError, httpx.NetworkError, httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e

                # Log the failure
                error_msg = str(e)[:100]
                if isinstance(e, httpx.HTTPStatusError):
                    error_msg = f"HTTP {e.response.status_code}"

                progress_logger.warning(
                    f"[{model_type}_FAILED] Model {current_model} failed: {error_msg}"
                )

                # If this was the last model, re-raise
                if idx == len(all_models) - 1:
                    progress_logger.error(
                        f"[FALLBACK_EXHAUSTED] All {len(all_models)} models failed. "
                        f"Primary: {model}, Fallbacks: {fallback_models}"
                    )
                    raise

                # Otherwise, continue to next fallback
                continue

            except Exception as e:
                # Non-retryable error (e.g., ValueError for prompt too long)
                # Don't try fallbacks for these
                progress_logger.error(
                    f"[{model_type}_ERROR] Non-retryable error for {current_model}: "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                raise

        # Should never reach here, but just in case
        if last_error:
            raise last_error
        raise RuntimeError("No models to try")

    async def _execute_request_with_retry(
        self, url: str, headers: dict[str, str], body: dict, timeout: Optional[int] = None
    ) -> dict:
        """
        Execute HTTP POST request with retry logic.

        Uses tenacity for exponential backoff retry on:
        - 5xx server errors
        - 429 rate limit errors
        - Network errors (connection, timeout)

        Does NOT retry on:
        - 4xx client errors (bad request, auth, etc.)

        Args:
            url: Full request URL
            headers: Request headers
            body: Request body (will be JSON-encoded)
            timeout: Request timeout in seconds (uses adapter default if not specified)

        Returns:
            Parsed JSON response

        Raises:
            httpx.HTTPStatusError: On HTTP error (after retries exhausted for 5xx)
            httpx.NetworkError: On network error (after retries exhausted)
        """
        effective_timeout = timeout if timeout is not None else self.timeout

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception(is_retryable_http_error),
            reraise=True,
        )
        async def _make_request():
            async with httpx.AsyncClient(timeout=effective_timeout) as client:
                progress_logger.debug(f"   [POST] Making request to {url}")
                response = await client.post(url, headers=headers, json=body)
                progress_logger.debug(f"   [RESPONSE] Status: {response.status_code}")

                # Log error response body for 4xx errors (helps debugging)
                if 400 <= response.status_code < 500:
                    try:
                        error_body = response.json()
                        progress_logger.error(
                            f"   [HTTP_ERROR] {response.status_code}: {json.dumps(error_body, indent=2)}"
                        )
                    except Exception:
                        progress_logger.error(
                            f"   [HTTP_ERROR] {response.status_code} body: {response.text[:500]}"
                        )

                response.raise_for_status()  # Raise for 4xx/5xx
                return response.json()

        return await _make_request()
