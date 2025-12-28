"""Gemini CLI adapter."""

from typing import Optional

from adapters.base import BaseCLIAdapter


class GeminiAdapter(BaseCLIAdapter):
    """Adapter for gemini CLI tool (Google AI).

    Note:
        Prompt length validation is handled by the base class.
        Default limit is 100,000 characters (~25k tokens at 4 chars/token),
        which prevents "invalid argument" API errors seen in production.
    """

    def __init__(
        self,
        command: str = "gemini",
        args: Optional[list[str]] = None,
        timeout: int = 60,
        default_reasoning_effort: Optional[str] = None,
        max_prompt_length: Optional[int] = None,
    ):
        """
        Initialize Gemini adapter.

        Args:
            command: Command to execute (default: "gemini")
            args: List of argument templates (from config.yaml)
            timeout: Timeout in seconds (default: 60)
            default_reasoning_effort: Ignored (Gemini doesn't support reasoning effort)
            max_prompt_length: Maximum prompt length in characters. If not specified,
                uses default of 100,000 characters from base class.

        Note:
            The gemini CLI uses `gemini -p "prompt"` or `gemini -m model -p "prompt"` syntax.
        """
        if args is None:
            raise ValueError("args must be provided from config.yaml")
        super().__init__(
            command=command,
            args=args,
            timeout=timeout,
            default_reasoning_effort=default_reasoning_effort,
            max_prompt_length=max_prompt_length,
        )

    def parse_output(self, raw_output: str) -> str:
        """
        Parse gemini output.

        Gemini outputs clean responses without header/footer text,
        so we simply strip whitespace.

        Args:
            raw_output: Raw stdout from gemini

        Returns:
            Parsed model response
        """
        return raw_output.strip()
