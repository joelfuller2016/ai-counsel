"""Context compression for large multi-round debates.

Provides intelligent summarization of previous rounds when context exceeds
configurable thresholds, preventing token limit issues while preserving
key points and voting history.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from models.schema import RoundResponse, Vote

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of context compression.

    Attributes:
        compressed_context: The compressed context string
        original_token_estimate: Estimated tokens in original context
        compressed_token_estimate: Estimated tokens after compression
        rounds_summarized: Number of rounds that were summarized
        key_points_preserved: Number of key points preserved
        voting_history_preserved: Whether voting history was preserved
    """

    compressed_context: str
    original_token_estimate: int
    compressed_token_estimate: int
    rounds_summarized: int
    key_points_preserved: int
    voting_history_preserved: bool


@dataclass
class VoteSnapshot:
    """Snapshot of a vote for history preservation."""

    round: int
    participant: str
    option: str
    confidence: float
    rationale: str


@dataclass
class RoundSummary:
    """Summary of a single round for compression."""

    round_number: int
    key_points: List[str] = field(default_factory=list)
    votes: List[VoteSnapshot] = field(default_factory=list)
    consensus_trend: Optional[str] = None


class ContextCompressor:
    """
    Compresses deliberation context when it exceeds token thresholds.

    Uses extractive summarization to preserve key points and voting history
    while reducing overall context size. Configurable via config.yaml.

    Compression Strategy:
    1. Keep recent rounds in full (configurable, default: last 2 rounds)
    2. Summarize older rounds by extracting:
       - Key points (first sentence of each substantive paragraph)
       - Voting history (option, confidence, rationale)
       - Consensus trends
    3. Preserve all voting snapshots for decision continuity

    Token Estimation:
    Uses simple character-based estimation (1 token ≈ 4 chars for English).
    This is approximate but sufficient for compression decisions.
    """

    # Default thresholds
    DEFAULT_COMPRESSION_THRESHOLD = 8000  # tokens
    DEFAULT_RECENT_ROUNDS_TO_KEEP = 2  # keep last N rounds in full
    DEFAULT_MAX_KEY_POINTS_PER_ROUND = 3
    DEFAULT_MAX_RATIONALE_LENGTH = 100

    def __init__(
        self,
        compression_threshold: int = DEFAULT_COMPRESSION_THRESHOLD,
        recent_rounds_to_keep: int = DEFAULT_RECENT_ROUNDS_TO_KEEP,
        max_key_points_per_round: int = DEFAULT_MAX_KEY_POINTS_PER_ROUND,
        max_rationale_length: int = DEFAULT_MAX_RATIONALE_LENGTH,
    ):
        """
        Initialize context compressor.

        Args:
            compression_threshold: Token count threshold to trigger compression
            recent_rounds_to_keep: Number of recent rounds to keep in full
            max_key_points_per_round: Maximum key points to extract per round
            max_rationale_length: Maximum characters for vote rationales
        """
        self.compression_threshold = compression_threshold
        self.recent_rounds_to_keep = recent_rounds_to_keep
        self.max_key_points_per_round = max_key_points_per_round
        self.max_rationale_length = max_rationale_length

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses simple character-based estimation (1 token ≈ 4 chars).
        This is approximate but sufficient for compression decisions.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return len(text) // 4

    def needs_compression(self, responses: List[RoundResponse]) -> bool:
        """
        Check if responses need compression based on token threshold.

        Args:
            responses: List of all responses from deliberation

        Returns:
            True if compression is needed
        """
        total_text = " ".join(r.response for r in responses)
        token_estimate = self.estimate_tokens(total_text)

        needs = token_estimate > self.compression_threshold
        if needs:
            logger.info(
                f"Context compression needed: ~{token_estimate} tokens "
                f"exceeds threshold of {self.compression_threshold}"
            )

        return needs

    def _extract_key_points(self, response_text: str) -> List[str]:
        """
        Extract key points from a response.

        Uses simple extractive approach: first sentence of each substantive
        paragraph, excluding tool requests and vote markers.

        Args:
            response_text: Raw response text

        Returns:
            List of key point strings
        """
        key_points = []

        # Split into paragraphs
        paragraphs = response_text.split("\n\n")

        for para in paragraphs:
            para = para.strip()

            # Skip empty, tool requests, vote markers, and error messages
            if not para:
                continue
            if "TOOL_REQUEST" in para:
                continue
            if para.startswith("VOTE:"):
                continue
            if para.startswith("[ERROR"):
                continue
            if para.startswith("[Vote Retry"):
                continue

            # Skip very short paragraphs (likely formatting)
            if len(para) < 30:
                continue

            # Extract first sentence (up to 200 chars)
            sentences = re.split(r"[.!?]", para)
            if sentences and sentences[0].strip():
                point = sentences[0].strip()[:200]
                # Add period if not present
                if not point.endswith((".", "!", "?")):
                    point += "."
                key_points.append(point)

                # Limit key points per response
                if len(key_points) >= self.max_key_points_per_round:
                    break

        return key_points

    def _extract_vote(
        self, response_text: str, participant: str, round_num: int
    ) -> Optional[VoteSnapshot]:
        """
        Extract vote information from response.

        Args:
            response_text: Raw response text
            participant: Participant identifier
            round_num: Round number

        Returns:
            VoteSnapshot if vote found, None otherwise
        """
        import json

        # Look for VOTE: marker
        vote_pattern = r"VOTE:\s*(\{.+?\})"
        matches = re.findall(vote_pattern, response_text, re.DOTALL)

        if not matches:
            return None

        # Take the last match (actual vote vs example)
        try:
            vote_data = json.loads(matches[-1])

            # Extract rationale, truncating if needed
            rationale = vote_data.get("rationale", "")
            if len(rationale) > self.max_rationale_length:
                rationale = rationale[: self.max_rationale_length] + "..."

            return VoteSnapshot(
                round=round_num,
                participant=participant,
                option=vote_data.get("option", "unknown"),
                confidence=vote_data.get("confidence", 0.0),
                rationale=rationale,
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _summarize_round(self, responses: List[RoundResponse]) -> RoundSummary:
        """
        Create a summary of a single round.

        Args:
            responses: Responses from a single round

        Returns:
            RoundSummary with key points and votes
        """
        if not responses:
            return RoundSummary(round_number=0)

        round_num = responses[0].round
        summary = RoundSummary(round_number=round_num)

        vote_options = []

        for response in responses:
            # Extract key points
            points = self._extract_key_points(response.response)
            for point in points:
                # Add participant attribution
                short_participant = response.participant.split("@")[0]
                summary.key_points.append(f"{short_participant}: {point}")

            # Extract vote
            vote = self._extract_vote(
                response.response, response.participant, round_num
            )
            if vote:
                summary.votes.append(vote)
                vote_options.append(vote.option)

        # Determine consensus trend
        if vote_options:
            unique_options = set(vote_options)
            if len(unique_options) == 1:
                summary.consensus_trend = f"Unanimous: {vote_options[0]}"
            else:
                # Count votes per option
                from collections import Counter

                vote_counts = Counter(vote_options)
                most_common = vote_counts.most_common(1)[0]
                summary.consensus_trend = (
                    f"Split: {most_common[0]} ({most_common[1]}/{len(vote_options)})"
                )

        return summary

    def _format_round_summary(self, summary: RoundSummary) -> str:
        """
        Format a round summary as compressed context.

        Args:
            summary: RoundSummary to format

        Returns:
            Formatted summary string
        """
        lines = [f"### Round {summary.round_number} Summary"]

        # Add consensus trend if available
        if summary.consensus_trend:
            lines.append(f"**Trend:** {summary.consensus_trend}")

        # Add key points
        if summary.key_points:
            lines.append("**Key Points:**")
            for point in summary.key_points[: self.max_key_points_per_round]:
                lines.append(f"- {point}")

        # Add voting snapshot
        if summary.votes:
            lines.append("**Votes:**")
            for vote in summary.votes:
                short_participant = vote.participant.split("@")[0]
                lines.append(
                    f"- {short_participant}: {vote.option} "
                    f"(confidence: {vote.confidence:.1%})"
                )

        return "\n".join(lines)

    def compress(self, responses: List[RoundResponse]) -> CompressionResult:
        """
        Compress deliberation context.

        Strategy:
        1. Group responses by round
        2. Keep recent N rounds in full
        3. Summarize older rounds
        4. Preserve all voting history

        Args:
            responses: All responses from deliberation

        Returns:
            CompressionResult with compressed context
        """
        if not responses:
            return CompressionResult(
                compressed_context="",
                original_token_estimate=0,
                compressed_token_estimate=0,
                rounds_summarized=0,
                key_points_preserved=0,
                voting_history_preserved=True,
            )

        # Calculate original token estimate
        original_text = " ".join(r.response for r in responses)
        original_tokens = self.estimate_tokens(original_text)

        # Group responses by round
        rounds: dict[int, List[RoundResponse]] = {}
        for response in responses:
            if response.round not in rounds:
                rounds[response.round] = []
            rounds[response.round].append(response)

        sorted_round_nums = sorted(rounds.keys())

        # Determine which rounds to summarize vs keep in full
        if len(sorted_round_nums) <= self.recent_rounds_to_keep:
            # Not enough rounds to compress, return original
            context_parts = ["Previous discussion:\n"]
            for resp in responses:
                context_parts.append(
                    f"Round {resp.round} - {resp.participant}: {resp.response}\n"
                )

            return CompressionResult(
                compressed_context="\n".join(context_parts),
                original_token_estimate=original_tokens,
                compressed_token_estimate=original_tokens,
                rounds_summarized=0,
                key_points_preserved=0,
                voting_history_preserved=True,
            )

        # Split into rounds to summarize and rounds to keep
        rounds_to_summarize = sorted_round_nums[: -self.recent_rounds_to_keep]
        rounds_to_keep = sorted_round_nums[-self.recent_rounds_to_keep :]

        context_parts = []
        total_key_points = 0
        all_votes: List[VoteSnapshot] = []

        # Add compressed summaries for older rounds
        if rounds_to_summarize:
            context_parts.append("## Compressed History\n")
            context_parts.append(
                f"*Rounds 1-{max(rounds_to_summarize)} summarized to preserve context.*\n"
            )

            for round_num in rounds_to_summarize:
                summary = self._summarize_round(rounds[round_num])
                formatted = self._format_round_summary(summary)
                context_parts.append(formatted)
                context_parts.append("")  # Blank line between rounds

                total_key_points += len(summary.key_points)
                all_votes.extend(summary.votes)

        # Add full context for recent rounds
        context_parts.append("## Recent Discussion\n")
        for round_num in rounds_to_keep:
            for resp in rounds[round_num]:
                context_parts.append(
                    f"Round {resp.round} - {resp.participant}: {resp.response}\n"
                )

                # Also extract votes from recent rounds
                vote = self._extract_vote(resp.response, resp.participant, round_num)
                if vote:
                    all_votes.append(vote)

        # Add voting history summary
        if all_votes:
            context_parts.append("\n## Voting History\n")

            # Group by round for clarity
            votes_by_round: dict[int, List[VoteSnapshot]] = {}
            for vote in all_votes:
                if vote.round not in votes_by_round:
                    votes_by_round[vote.round] = []
                votes_by_round[vote.round].append(vote)

            for round_num in sorted(votes_by_round.keys()):
                context_parts.append(f"**Round {round_num}:**")
                for vote in votes_by_round[round_num]:
                    short_participant = vote.participant.split("@")[0]
                    context_parts.append(
                        f"- {short_participant}: {vote.option} "
                        f"({vote.confidence:.0%}) - {vote.rationale}"
                    )
                context_parts.append("")

        compressed_context = "\n".join(context_parts)
        compressed_tokens = self.estimate_tokens(compressed_context)

        logger.info(
            f"Context compressed: {original_tokens} -> {compressed_tokens} tokens "
            f"({100 - (compressed_tokens/original_tokens*100):.1f}% reduction). "
            f"Summarized {len(rounds_to_summarize)} rounds, "
            f"preserved {total_key_points} key points and {len(all_votes)} votes."
        )

        return CompressionResult(
            compressed_context=compressed_context,
            original_token_estimate=original_tokens,
            compressed_token_estimate=compressed_tokens,
            rounds_summarized=len(rounds_to_summarize),
            key_points_preserved=total_key_points,
            voting_history_preserved=len(all_votes) > 0,
        )
