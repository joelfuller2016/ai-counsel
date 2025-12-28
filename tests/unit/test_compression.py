"""Unit tests for context compression module."""

import pytest

from deliberation.compression import (
    ContextCompressor,
    CompressionResult,
    VoteSnapshot,
    RoundSummary,
)
from models.schema import RoundResponse


class TestContextCompressor:
    """Tests for ContextCompressor class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        compressor = ContextCompressor()

        assert compressor.compression_threshold == 8000
        assert compressor.recent_rounds_to_keep == 2
        assert compressor.max_key_points_per_round == 3
        assert compressor.max_rationale_length == 100

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        compressor = ContextCompressor(
            compression_threshold=5000,
            recent_rounds_to_keep=3,
            max_key_points_per_round=5,
            max_rationale_length=150,
        )

        assert compressor.compression_threshold == 5000
        assert compressor.recent_rounds_to_keep == 3
        assert compressor.max_key_points_per_round == 5
        assert compressor.max_rationale_length == 150

    def test_estimate_tokens_empty(self):
        """Test token estimation for empty text."""
        compressor = ContextCompressor()

        assert compressor.estimate_tokens("") == 0
        assert compressor.estimate_tokens(None) == 0

    def test_estimate_tokens_basic(self):
        """Test token estimation for basic text."""
        compressor = ContextCompressor()

        # 40 chars / 4 = 10 tokens
        text = "a" * 40
        assert compressor.estimate_tokens(text) == 10

        # 100 chars / 4 = 25 tokens
        text = "x" * 100
        assert compressor.estimate_tokens(text) == 25

    def test_needs_compression_below_threshold(self):
        """Test needs_compression returns False when below threshold."""
        compressor = ContextCompressor(compression_threshold=1000)

        # Short responses (should not need compression)
        responses = [
            RoundResponse(
                round=1,
                participant="model@cli",
                response="Short response.",
                timestamp="2024-01-01T00:00:00",
            ),
        ]

        assert not compressor.needs_compression(responses)

    def test_needs_compression_above_threshold(self):
        """Test needs_compression returns True when above threshold."""
        compressor = ContextCompressor(compression_threshold=100)  # Very low threshold

        # Long responses (should need compression)
        responses = [
            RoundResponse(
                round=1,
                participant="model@cli",
                response="A" * 1000,  # ~250 tokens
                timestamp="2024-01-01T00:00:00",
            ),
        ]

        assert compressor.needs_compression(responses)

    def test_extract_key_points_basic(self):
        """Test key point extraction from response text."""
        compressor = ContextCompressor()

        response_text = """First key point about the topic is important.

Second paragraph with another main idea.

Third paragraph has its own perspective."""

        points = compressor._extract_key_points(response_text)

        assert len(points) == 3
        assert "First key point" in points[0]
        assert "Second paragraph" in points[1]

    def test_extract_key_points_skips_tool_requests(self):
        """Test key point extraction skips tool requests."""
        compressor = ContextCompressor()

        response_text = """This is a valid key point about the topic.

TOOL_REQUEST: {"name": "read_file", "arguments": {"path": "file.py"}}

Another valid paragraph with content."""

        points = compressor._extract_key_points(response_text)

        assert len(points) == 2
        # Should not contain TOOL_REQUEST
        for point in points:
            assert "TOOL_REQUEST" not in point

    def test_extract_key_points_skips_votes(self):
        """Test key point extraction skips vote markers."""
        compressor = ContextCompressor()

        response_text = """This is analysis content worth preserving.

VOTE: {"option": "A", "confidence": 0.9, "rationale": "Good reasons"}"""

        points = compressor._extract_key_points(response_text)

        assert len(points) == 1
        assert "VOTE" not in points[0]

    def test_extract_key_points_limit(self):
        """Test key point extraction respects max limit."""
        compressor = ContextCompressor(max_key_points_per_round=2)

        response_text = """First paragraph has content.

Second paragraph also important.

Third paragraph should be excluded.

Fourth paragraph also excluded."""

        points = compressor._extract_key_points(response_text)

        assert len(points) <= 2

    def test_extract_vote_basic(self):
        """Test vote extraction from response text."""
        compressor = ContextCompressor()

        response_text = """Analysis of the question...

VOTE: {"option": "Option A", "confidence": 0.85, "rationale": "Strong reasons"}"""

        vote = compressor._extract_vote(response_text, "model@cli", 1)

        assert vote is not None
        assert vote.option == "Option A"
        assert vote.confidence == 0.85
        assert "Strong reasons" in vote.rationale
        assert vote.round == 1
        assert vote.participant == "model@cli"

    def test_extract_vote_rationale_truncation(self):
        """Test vote extraction truncates long rationales."""
        compressor = ContextCompressor(max_rationale_length=20)

        response_text = """VOTE: {"option": "A", "confidence": 0.9, "rationale": "This is a very long rationale that should be truncated to preserve context space"}"""

        vote = compressor._extract_vote(response_text, "model@cli", 1)

        assert vote is not None
        assert len(vote.rationale) <= 23  # 20 + "..."
        assert vote.rationale.endswith("...")

    def test_extract_vote_no_vote_returns_none(self):
        """Test vote extraction returns None when no vote present."""
        compressor = ContextCompressor()

        response_text = "Just analysis without any voting."

        vote = compressor._extract_vote(response_text, "model@cli", 1)

        assert vote is None

    def test_extract_vote_invalid_json_returns_none(self):
        """Test vote extraction returns None for invalid JSON."""
        compressor = ContextCompressor()

        response_text = "VOTE: {invalid json here}"

        vote = compressor._extract_vote(response_text, "model@cli", 1)

        assert vote is None

    def test_summarize_round_basic(self):
        """Test round summarization."""
        compressor = ContextCompressor()

        # Use longer, more realistic response text (30+ chars needed for key point extraction)
        responses = [
            RoundResponse(
                round=1,
                participant="claude@cli",
                response='This is a substantive key point from Claude that contains enough content to be considered meaningful.\n\nVOTE: {"option": "A", "confidence": 0.9, "rationale": "Good reasons for this choice"}',
                timestamp="2024-01-01T00:00:00",
            ),
            RoundResponse(
                round=1,
                participant="codex@cli",
                response='This is another important key point from Codex with sufficient content to be extracted as a summary.\n\nVOTE: {"option": "A", "confidence": 0.8, "rationale": "I also agree with this"}',
                timestamp="2024-01-01T00:00:01",
            ),
        ]

        summary = compressor._summarize_round(responses)

        assert summary.round_number == 1
        assert len(summary.key_points) >= 2
        assert len(summary.votes) == 2
        assert "Unanimous" in summary.consensus_trend

    def test_summarize_round_split_votes(self):
        """Test round summarization with split votes."""
        compressor = ContextCompressor()

        responses = [
            RoundResponse(
                round=2,
                participant="claude@cli",
                response='Analysis here.\n\nVOTE: {"option": "A", "confidence": 0.9, "rationale": "Pick A"}',
                timestamp="2024-01-01T00:00:00",
            ),
            RoundResponse(
                round=2,
                participant="codex@cli",
                response='Different view.\n\nVOTE: {"option": "B", "confidence": 0.8, "rationale": "Pick B"}',
                timestamp="2024-01-01T00:00:01",
            ),
        ]

        summary = compressor._summarize_round(responses)

        assert summary.round_number == 2
        assert "Split" in summary.consensus_trend

    def test_compress_no_responses(self):
        """Test compression with empty responses."""
        compressor = ContextCompressor()

        result = compressor.compress([])

        assert result.compressed_context == ""
        assert result.original_token_estimate == 0
        assert result.compressed_token_estimate == 0
        assert result.rounds_summarized == 0

    def test_compress_few_rounds_no_compression(self):
        """Test compression with fewer rounds than threshold."""
        compressor = ContextCompressor(recent_rounds_to_keep=3)

        responses = [
            RoundResponse(
                round=1,
                participant="model@cli",
                response="Response 1",
                timestamp="2024-01-01T00:00:00",
            ),
            RoundResponse(
                round=2,
                participant="model@cli",
                response="Response 2",
                timestamp="2024-01-01T00:00:01",
            ),
        ]

        result = compressor.compress(responses)

        # With only 2 rounds and recent_rounds_to_keep=3, no summarization needed
        assert result.rounds_summarized == 0

    def test_compress_many_rounds(self):
        """Test compression with many rounds."""
        compressor = ContextCompressor(
            compression_threshold=100,  # Low threshold to trigger
            recent_rounds_to_keep=2,
        )

        # Create responses across 5 rounds with much longer content
        # This simulates real deliberation responses that would benefit from compression
        long_content = (
            "This is a very detailed and comprehensive analysis of the topic at hand. "
            "We need to consider multiple factors including technical feasibility, "
            "business impact, resource requirements, and timeline considerations. "
            "After thorough examination of the evidence and discussion with the team, "
            "I have reached the following conclusions based on the available data. "
        )

        responses = []
        for round_num in range(1, 6):
            responses.append(
                RoundResponse(
                    round=round_num,
                    participant="claude@cli",
                    response=f"Round {round_num} Claude analysis: {long_content * 5}\n\n"
                    f'VOTE: {{"option": "A", "confidence": 0.9, "rationale": "Round {round_num} vote with detailed reasoning"}}',
                    timestamp=f"2024-01-01T00:00:{round_num:02d}",
                )
            )
            responses.append(
                RoundResponse(
                    round=round_num,
                    participant="codex@cli",
                    response=f"Round {round_num} Codex analysis: {long_content * 5}\n\n"
                    f'VOTE: {{"option": "A", "confidence": 0.85, "rationale": "Codex round {round_num} detailed vote"}}',
                    timestamp=f"2024-01-01T00:00:{round_num+10:02d}",
                )
            )

        result = compressor.compress(responses)

        # Rounds 1-3 should be summarized, 4-5 kept in full
        assert result.rounds_summarized == 3
        assert result.voting_history_preserved
        # With much longer content, compression should actually reduce size
        assert result.compressed_token_estimate < result.original_token_estimate

    def test_compress_preserves_voting_history(self):
        """Test compression preserves voting history section."""
        compressor = ContextCompressor(recent_rounds_to_keep=1)

        responses = [
            RoundResponse(
                round=1,
                participant="model@cli",
                response='Analysis.\n\nVOTE: {"option": "A", "confidence": 0.9, "rationale": "First vote"}',
                timestamp="2024-01-01T00:00:00",
            ),
            RoundResponse(
                round=2,
                participant="model@cli",
                response='More analysis.\n\nVOTE: {"option": "A", "confidence": 0.95, "rationale": "Second vote"}',
                timestamp="2024-01-01T00:00:01",
            ),
            RoundResponse(
                round=3,
                participant="model@cli",
                response='Final analysis.\n\nVOTE: {"option": "A", "confidence": 0.98, "rationale": "Final vote"}',
                timestamp="2024-01-01T00:00:02",
            ),
        ]

        result = compressor.compress(responses)

        # Check voting history is in the output
        assert "Voting History" in result.compressed_context
        assert result.voting_history_preserved


class TestCompressionResult:
    """Tests for CompressionResult dataclass."""

    def test_compression_result_fields(self):
        """Test CompressionResult has expected fields."""
        result = CompressionResult(
            compressed_context="compressed text",
            original_token_estimate=1000,
            compressed_token_estimate=500,
            rounds_summarized=3,
            key_points_preserved=9,
            voting_history_preserved=True,
        )

        assert result.compressed_context == "compressed text"
        assert result.original_token_estimate == 1000
        assert result.compressed_token_estimate == 500
        assert result.rounds_summarized == 3
        assert result.key_points_preserved == 9
        assert result.voting_history_preserved is True


class TestVoteSnapshot:
    """Tests for VoteSnapshot dataclass."""

    def test_vote_snapshot_fields(self):
        """Test VoteSnapshot has expected fields."""
        vote = VoteSnapshot(
            round=2,
            participant="claude@cli",
            option="Option A",
            confidence=0.9,
            rationale="Strong reasoning",
        )

        assert vote.round == 2
        assert vote.participant == "claude@cli"
        assert vote.option == "Option A"
        assert vote.confidence == 0.9
        assert vote.rationale == "Strong reasoning"


class TestRoundSummary:
    """Tests for RoundSummary dataclass."""

    def test_round_summary_defaults(self):
        """Test RoundSummary default values."""
        summary = RoundSummary(round_number=1)

        assert summary.round_number == 1
        assert summary.key_points == []
        assert summary.votes == []
        assert summary.consensus_trend is None

    def test_round_summary_with_data(self):
        """Test RoundSummary with populated data."""
        summary = RoundSummary(
            round_number=3,
            key_points=["Point 1", "Point 2"],
            votes=[
                VoteSnapshot(
                    round=3,
                    participant="model@cli",
                    option="A",
                    confidence=0.9,
                    rationale="Good",
                )
            ],
            consensus_trend="Unanimous: A",
        )

        assert summary.round_number == 3
        assert len(summary.key_points) == 2
        assert len(summary.votes) == 1
        assert summary.consensus_trend == "Unanimous: A"
