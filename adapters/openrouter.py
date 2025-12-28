"""OpenRouter HTTP adapter with rate limit handling."""
import asyncio
import logging
import random
from datetime import datetime
from typing import Optional, Tuple

import httpx

from adapters.base_http import BaseHTTPAdapter, HealthCheckResult

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is hit and retries are exhausted.

    Attributes:
        retry_after: Suggested wait time in seconds (from Retry-After header)
        status_code: HTTP status code (429)
        message: Error message from the API
    """

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        status_code: int = 429,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.status_code = status_code
        self.message = message


class OpenRouterAdapter(BaseHTTPAdapter):
    """
    Adapter for OpenRouter API with rate limit handling.

    OpenRouter provides access to multiple LLM providers through a unified
    OpenAI-compatible API with authentication. Free-tier models have rate
    limits that require exponential backoff with jitter.

    Features:
    - Detects 429 rate limit responses
    - Implements exponential backoff with jitter
    - Parses Retry-After header when present
    - Configurable max_retries and base_retry_delay

    API Reference: https://openrouter.ai/docs
    Default endpoint: https://openrouter.ai/api/v1

    Rate Limit Behavior:
    - Free models: ~20 requests/minute, ~200 requests/day
    - Retry delay: base_delay * (2^attempt) + random jitter
    - Jitter: 0-1 second random delay to prevent thundering herd
    """

    ADAPTER_NAME = "openrouter"

    # Default rate limit retry settings
    DEFAULT_RATE_LIMIT_RETRIES = 3
    DEFAULT_BASE_RETRY_DELAY = 1.0  # seconds
    MAX_RETRY_DELAY = 60.0  # seconds (cap for exponential backoff)

    def __init__(
        self,
        base_url: str,
        timeout: int = 60,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        max_prompt_length: Optional[int] = None,
        rate_limit_retries: Optional[int] = None,
        base_retry_delay: Optional[float] = None,
    ):
        """
        Initialize OpenRouter adapter with rate limit handling.

        Args:
            base_url: Base URL for API (e.g., "https://openrouter.ai/api/v1")
            timeout: Request timeout in seconds (default: 60)
            max_retries: Maximum retry attempts for transient failures (default: 3)
            api_key: Optional API key for authentication
            headers: Optional default headers to include in all requests
            max_prompt_length: Maximum prompt length in characters
            rate_limit_retries: Max retries specifically for 429 rate limits (default: 3)
            base_retry_delay: Base delay for exponential backoff in seconds (default: 1.0)
        """
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
            headers=headers,
            max_prompt_length=max_prompt_length,
        )
        self.rate_limit_retries = (
            rate_limit_retries
            if rate_limit_retries is not None
            else self.DEFAULT_RATE_LIMIT_RETRIES
        )
        self.base_retry_delay = (
            base_retry_delay
            if base_retry_delay is not None
            else self.DEFAULT_BASE_RETRY_DELAY
        )

    def build_request(
        self, model: str, prompt: str
    ) -> Tuple[str, dict[str, str], dict]:
        """
        Build OpenRouter API request (OpenAI-compatible format with auth).

        OpenRouter uses the OpenAI chat completions API format with Bearer token auth:
        POST /chat/completions
        Authorization: Bearer <api_key>

        Args:
            model: Model identifier (e.g., "anthropic/claude-3.5-sonnet", "openai/gpt-4")
            prompt: The prompt to send

        Returns:
            Tuple of (endpoint, headers, body)
        """
        endpoint = "/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Convert prompt to OpenAI chat format
        # max_tokens: 4096 ensures responses aren't truncated
        # Some models default to very low token limits
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,  # Use non-streaming for simplicity
            "max_tokens": 4096,  # Prevent truncation - models need ~2000-4000 for votes
        }

        return (endpoint, headers, body)

    def parse_response(self, response_json: dict) -> str:
        """
        Parse OpenRouter API response (OpenAI format).

        OpenRouter returns OpenAI-compatible chat completions format:
        {
          "id": "gen-abc123",
          "model": "anthropic/claude-3.5-sonnet",
          "created": 1234567890,
          "choices": [{
            "index": 0,
            "message": {
              "role": "assistant",
              "content": "The model's response"
            },
            "finish_reason": "stop"
          }]
        }

        Args:
            response_json: Parsed JSON response from OpenRouter

        Returns:
            Extracted response text from first choice

        Raises:
            KeyError: If response doesn't contain expected fields
            IndexError: If choices array is empty
        """
        if "choices" not in response_json:
            raise KeyError(
                f"OpenRouter response missing 'choices' field. "
                f"Received keys: {list(response_json.keys())}"
            )

        if len(response_json["choices"]) == 0:
            raise IndexError("OpenRouter response has empty 'choices' array")

        choice = response_json["choices"][0]

        # Log warning if response was truncated due to token limit
        finish_reason = choice.get("finish_reason", "unknown")
        if finish_reason == "length":
            model = response_json.get("model", "unknown")
            logger.warning(
                f"OpenRouter response truncated (finish_reason='length') for model {model}. "
                f"Consider increasing max_tokens or using a model with higher limits."
            )

        if "message" not in choice:
            raise KeyError(
                f"OpenRouter choice missing 'message' field. "
                f"Received keys: {list(choice.keys())}"
            )

        message = choice["message"]

        if "content" not in message:
            raise KeyError(
                f"OpenRouter message missing 'content' field. "
                f"Received keys: {list(message.keys())}"
            )

        return message["content"]

    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        """
        Parse Retry-After header from response.

        Handles both formats:
        - Seconds (numeric): "60"
        - HTTP date: "Wed, 21 Oct 2015 07:28:00 GMT"

        Args:
            response: HTTP response object

        Returns:
            Retry delay in seconds, or None if header not present/parseable
        """
        retry_after = response.headers.get("Retry-After") or response.headers.get(
            "retry-after"
        )
        if not retry_after:
            return None

        try:
            # Try numeric format first
            return float(retry_after)
        except ValueError:
            pass

        # Try HTTP date format
        try:
            from email.utils import parsedate_to_datetime

            retry_date = parsedate_to_datetime(retry_after)
            delta = retry_date - datetime.now(retry_date.tzinfo)
            return max(0.0, delta.total_seconds())
        except Exception:
            pass

        return None

    def _calculate_backoff_delay(
        self, attempt: int, retry_after: Optional[float] = None
    ) -> float:
        """
        Calculate backoff delay with exponential growth and jitter.

        Formula: min(base_delay * 2^attempt + random(0,1), MAX_RETRY_DELAY)

        If Retry-After header is present, use that instead (but add small jitter).

        Args:
            attempt: Current retry attempt (0-indexed)
            retry_after: Optional Retry-After value from server

        Returns:
            Delay in seconds before next retry
        """
        if retry_after is not None:
            # Use server-specified delay with small jitter to avoid thundering herd
            jitter = random.uniform(0.0, 1.0)
            return min(retry_after + jitter, self.MAX_RETRY_DELAY)

        # Exponential backoff: base * 2^attempt
        exponential_delay = self.base_retry_delay * (2**attempt)

        # Add jitter (0-1 second)
        jitter = random.uniform(0.0, 1.0)

        # Cap at maximum delay
        return min(exponential_delay + jitter, self.MAX_RETRY_DELAY)

    async def _execute_with_rate_limit_retry(
        self, url: str, headers: dict[str, str], body: dict, timeout: float
    ) -> dict:
        """
        Execute HTTP request with rate limit retry logic.

        Handles 429 responses with exponential backoff and jitter.
        Uses Retry-After header when available.

        Args:
            url: Full request URL
            headers: Request headers
            body: Request body (will be JSON-encoded)
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON response

        Raises:
            RateLimitError: If rate limit retries exhausted
            httpx.HTTPStatusError: For non-429 HTTP errors
        """
        last_error = None
        last_retry_after = None

        for attempt in range(self.rate_limit_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, headers=headers, json=body)

                    if response.status_code == 429:
                        # Rate limited - extract retry info
                        retry_after = self._parse_retry_after(response)
                        last_retry_after = retry_after

                        # Try to get error message from response
                        try:
                            error_body = response.json()
                            error_msg = error_body.get("error", {})
                            if isinstance(error_msg, dict):
                                error_msg = error_msg.get(
                                    "message", "Rate limit exceeded"
                                )
                            else:
                                error_msg = str(error_msg) or "Rate limit exceeded"
                        except Exception:
                            error_msg = "Rate limit exceeded"

                        if attempt < self.rate_limit_retries:
                            # Calculate delay and retry
                            delay = self._calculate_backoff_delay(attempt, retry_after)
                            retry_after_info = (
                                f" (Retry-After: {retry_after}s)"
                                if retry_after
                                else ""
                            )
                            logger.warning(
                                f"Rate limited (429){retry_after_info}. "
                                f"Attempt {attempt + 1}/{self.rate_limit_retries + 1}. "
                                f"Waiting {delay:.1f}s before retry..."
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            # Retries exhausted
                            last_error = RateLimitError(
                                message=f"Rate limit exceeded after {self.rate_limit_retries + 1} attempts: {error_msg}",
                                retry_after=retry_after,
                                status_code=429,
                            )
                            raise last_error

                    # Raise for other HTTP errors
                    response.raise_for_status()

                    # Success
                    if attempt > 0:
                        logger.info(
                            f"Request succeeded after {attempt + 1} attempts "
                            f"(recovered from rate limiting)"
                        )
                    return response.json()

            except RateLimitError:
                raise
            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                # Network errors etc - let parent class handle retries
                raise

        # Should not reach here, but raise if we do
        raise RateLimitError(
            message="Rate limit retries exhausted",
            retry_after=last_retry_after,
            status_code=429,
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
        Invoke the OpenRouter API with rate limit handling.

        Overrides base invoke to add rate limit retry logic with exponential
        backoff and jitter for free-tier models.

        Args:
            prompt: The prompt to send to the model
            model: Model identifier
            context: Optional additional context to prepend to prompt
            is_deliberation: Whether this is part of a deliberation
            working_directory: Unused for HTTP adapters
            reasoning_effort: Unused for HTTP adapters
            timeout_override: Optional model-specific timeout in seconds

        Returns:
            Parsed response from the model

        Raises:
            TimeoutError: If request exceeds timeout
            RateLimitError: If rate limit retries exhausted
            httpx.HTTPStatusError: If API returns error status
            ValueError: If prompt exceeds max length
        """
        # Use model-specific timeout if provided
        effective_timeout = (
            timeout_override if timeout_override is not None else self.timeout
        )

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

        # Get request components
        endpoint, headers, body = self.build_request(model, full_prompt)
        full_url = f"{self.base_url}{endpoint}"

        logger.debug(
            f"OpenRouter request: model={model}, prompt_length={len(full_prompt)}, "
            f"timeout={effective_timeout}s"
        )

        start_time = datetime.now()

        try:
            # Execute with rate limit retry logic
            response_json = await self._execute_with_rate_limit_retry(
                url=full_url,
                headers=headers,
                body=body,
                timeout=effective_timeout,
            )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.debug(f"OpenRouter request succeeded in {elapsed:.2f}s")

            return self.parse_response(response_json)

        except asyncio.TimeoutError:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"OpenRouter request timed out after {elapsed:.2f}s")
            raise TimeoutError(f"HTTP request timed out after {effective_timeout}s")

        except RateLimitError as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"OpenRouter rate limit error after {elapsed:.2f}s: {e.message}"
            )
            raise

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"OpenRouter request failed after {elapsed:.2f}s: "
                f"{type(e).__name__}: {str(e)}"
            )
            raise

    async def health_check(
        self, model: str, timeout: Optional[float] = None
    ) -> HealthCheckResult:
        """
        Check if an OpenRouter model is available.

        Sends a minimal request to verify the model can respond.
        Handles rate limits gracefully for free-tier models.

        Args:
            model: Model identifier to check
            timeout: Optional timeout in seconds (defaults to 15s for OpenRouter)

        Returns:
            HealthCheckResult with availability status, latency, and any errors
        """
        # Use slightly longer timeout for OpenRouter due to free tier latency
        check_timeout = timeout if timeout is not None else 15.0
        start_time = datetime.now()

        try:
            # Build a minimal health check request
            endpoint, headers, body = self.build_request(model, "Hi")
            body["max_tokens"] = 1  # Minimal response

            full_url = f"{self.base_url}{endpoint}"

            async with httpx.AsyncClient(timeout=check_timeout) as client:
                response = await client.post(full_url, headers=headers, json=body)

                # Special handling for rate limit on health check
                if response.status_code == 429:
                    retry_after = self._parse_retry_after(response)
                    retry_info = f" (retry after {retry_after}s)" if retry_after else ""
                    return HealthCheckResult(
                        available=False,
                        error=f"Rate limited{retry_info}",
                        model=model,
                        adapter=self.ADAPTER_NAME,
                    )

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
