"""Unit tests for summarizer module including BasicSummarizer fallback."""

import pytest
from datetime import datetime

from deliberation.summarizer import BasicSummarizer, DeliberationSummarizer
from models.schema import RoundResponse, Summary, Vote, RoundVote, VotingResult


class TestBasicSummarizer:
    """Tests for BasicSummarizer fallback functionality."""

    @pytest.fixture
    def basic_summarizer(self):
        """Create a BasicSummarizer instance."""
        return BasicSummarizer()

    @pytest.fixture
    def sample_responses(self):
        """Create sample deliberation responses."""
        return [
            RoundResponse(
                round=1,
                participant="claude-sonnet@claude",
                response=(
                    "I believe Option A is the better choice because it offers "
                    "better maintainability and follows established patterns.\n\n"
                    "The key consideration here is long-term sustainability.\n\n"
                    'VOTE: {"option": "Option A", "confidence": 0.85, '
                    '"rationale": "Better maintainability"}'
                ),
                timestamp=datetime.now().isoformat(),
            ),
            RoundResponse(
                round=1,
                participant="gpt-5@codex",
                response=(
                    "After careful analysis, Option A provides the best balance "
                    "of performance and readability.\n\n"
                    "The performance benchmarks support this conclusion.\n\n"
                    'VOTE: {"option": "Option A", "confidence": 0.9, '
                    '"rationale": "Best balance of concerns"}'
                ),
                timestamp=datetime.now().isoformat(),
            ),
        ]

    @pytest.fixture
    def sample_voting_result_consensus(self):
        """Create a voting result with consensus."""
        return VotingResult(
            final_tally={"Option A": 2},
            votes_by_round=[
                RoundVote(
                    round=1,
                    participant="claude-sonnet@claude",
                    vote=Vote(
                        option="Option A",
                        confidence=0.85,
                        rationale="Better maintainability",
                    ),
                    timestamp=datetime.now().isoformat(),
                ),
                RoundVote(
                    round=1,
                    participant="gpt-5@codex",
                    vote=Vote(
                        option="Option A",
                        confidence=0.9,
                        rationale="Best balance",
                    ),
                    timestamp=datetime.now().isoformat(),
                ),
            ],
            consensus_reached=True,
            winning_option="Option A",
        )

    @pytest.fixture
    def sample_voting_result_split(self):
        """Create a voting result with split votes."""
        return VotingResult(
            final_tally={"Option A": 1, "Option B": 1},
            votes_by_round=[
                RoundVote(
                    round=1,
                    participant="claude-sonnet@claude",
                    vote=Vote(option="Option A", confidence=0.8, rationale="Reason A"),
                    timestamp=datetime.now().isoformat(),
                ),
                RoundVote(
                    round=1,
                    participant="gpt-5@codex",
                    vote=Vote(option="Option B", confidence=0.7, rationale="Reason B"),
                    timestamp=datetime.now().isoformat(),
                ),
            ],
            consensus_reached=False,
            winning_option=None,
        )

    def test_generate_summary_with_consensus(
        self, basic_summarizer, sample_responses, sample_voting_result_consensus
    ):
        """Test summary generation when consensus is reached."""
        summary = basic_summarizer.generate_summary(
            question="Which option should we choose?",
            responses=sample_responses,
            voting_result=sample_voting_result_consensus,
        )

        assert isinstance(summary, Summary)
        assert "[Auto-generated fallback summary]" in summary.consensus
        assert "Consensus reached" in summary.consensus
        assert "Option A" in summary.consensus
        assert "2/2 votes" in summary.consensus
        assert len(summary.key_agreements) > 0
        assert "Option A" in summary.final_recommendation

    def test_generate_summary_with_split_votes(
        self, basic_summarizer, sample_responses, sample_voting_result_split
    ):
        """Test summary generation when votes are split."""
        summary = basic_summarizer.generate_summary(
            question="Which option should we choose?",
            responses=sample_responses,
            voting_result=sample_voting_result_split,
        )

        assert isinstance(summary, Summary)
        assert "[Auto-generated fallback summary]" in summary.consensus
        assert "No clear consensus" in summary.consensus
        # Should mention the split in disagreements
        assert any(
            "different options" in d.lower() or "Option A" in d
            for d in summary.key_disagreements
        )
        assert "No clear winner" in summary.final_recommendation

    def test_generate_summary_without_voting_result(
        self, basic_summarizer, sample_responses
    ):
        """Test summary generation when no voting result is available."""
        summary = basic_summarizer.generate_summary(
            question="What should we do?",
            responses=sample_responses,
            voting_result=None,
        )

        assert isinstance(summary, Summary)
        assert "[Auto-generated fallback summary]" in summary.consensus
        assert "No voting data" in summary.consensus
        assert len(summary.key_agreements) > 0

    def test_generate_summary_with_empty_responses(self, basic_summarizer):
        """Test summary generation with empty responses."""
        summary = basic_summarizer.generate_summary(
            question="Empty deliberation?",
            responses=[],
            voting_result=None,
        )

        assert isinstance(summary, Summary)
        assert "[Auto-generated fallback summary]" in summary.consensus

    def test_generate_summary_with_error_responses(self, basic_summarizer):
        """Test summary generation with error responses."""
        responses = [
            RoundResponse(
                round=1,
                participant="model@cli",
                response="[ERROR: Connection timeout]",
                timestamp=datetime.now().isoformat(),
            ),
        ]

        summary = basic_summarizer.generate_summary(
            question="Test question",
            responses=responses,
            voting_result=None,
        )

        assert isinstance(summary, Summary)
        # Error responses should be skipped in point extraction
        assert "[Auto-generated fallback summary]" in summary.consensus

    def test_extract_participant_points_skips_tool_requests(self, basic_summarizer):
        """Test that tool requests are skipped when extracting points."""
        responses = [
            RoundResponse(
                round=1,
                participant="model@cli",
                response=(
                    "Let me check the code.\n\n"
                    'TOOL_REQUEST: {"name": "read_file", "arguments": {}}\n\n'
                    "Based on my analysis, Option A is better because it provides "
                    "better error handling and cleaner separation of concerns."
                ),
                timestamp=datetime.now().isoformat(),
            ),
        ]

        points = basic_summarizer._extract_participant_points(responses)

        # Should have extracted the analysis point, not the tool request
        assert "model@cli" in points
        assert len(points["model@cli"]) > 0
        # Tool request should not be in extracted points
        for point in points["model@cli"]:
            assert "TOOL_REQUEST" not in point

    def test_extract_participant_points_limits_to_three(self, basic_summarizer):
        """Test that only top 3 points are extracted per participant."""
        long_response = "\n\n".join(
            [
                f"Point {i}: This is a longer sentence that should be extracted as a point."
                for i in range(10)
            ]
        )
        responses = [
            RoundResponse(
                round=1,
                participant="model@cli",
                response=long_response,
                timestamp=datetime.now().isoformat(),
            ),
        ]

        points = basic_summarizer._extract_participant_points(responses)

        assert "model@cli" in points
        assert len(points["model@cli"]) <= 3

    def test_find_disagreements_filters_abstain(self, basic_summarizer):
        """Test that ABSTAIN votes are filtered from disagreement reporting."""
        voting_result = VotingResult(
            final_tally={"Option A": 1, "ABSTAIN": 1},
            votes_by_round=[],
            consensus_reached=False,
            winning_option=None,
        )

        disagreements = basic_summarizer._find_disagreements({}, voting_result)

        # ABSTAIN should not be mentioned as a disagreeing option
        for d in disagreements:
            if "different options" in d.lower():
                assert "ABSTAIN" not in d

    def test_consensus_with_majority_vote(self, basic_summarizer):
        """Test consensus statement with majority (not unanimous) vote."""
        voting_result = VotingResult(
            final_tally={"Option A": 2, "Option B": 1},
            votes_by_round=[],
            consensus_reached=True,
            winning_option="Option A",
        )

        consensus = basic_summarizer._build_consensus_statement(voting_result, [])

        assert "Consensus reached" in consensus
        assert "Option A" in consensus
        assert "2/3 votes" in consensus


class TestDeliberationSummarizerFallback:
    """Tests for DeliberationSummarizer error handling and fallback behavior."""

    @pytest.fixture
    def mock_failing_adapter(self, mocker):
        """Create a mock adapter that always fails."""
        adapter = mocker.Mock()
        adapter.invoke = mocker.AsyncMock(
            side_effect=Exception("API connection failed")
        )
        return adapter

    @pytest.fixture
    def sample_responses(self):
        """Create sample responses."""
        return [
            RoundResponse(
                round=1,
                participant="model@cli",
                response="Analysis: Option A is better.",
                timestamp=datetime.now().isoformat(),
            ),
        ]

    @pytest.mark.asyncio
    async def test_summarizer_returns_fallback_on_error(
        self, mock_failing_adapter, sample_responses
    ):
        """Test that DeliberationSummarizer returns fallback summary on error."""
        summarizer = DeliberationSummarizer(mock_failing_adapter, "test-model")

        summary = await summarizer.generate_summary(
            question="Test question",
            responses=sample_responses,
        )

        # Should return error fallback (from DeliberationSummarizer's own error handling)
        assert isinstance(summary, Summary)
        assert (
            "failed" in summary.consensus.lower()
            or "error" in summary.key_agreements[0].lower()
        )


class TestSummaryIntegration:
    """Integration tests for summary generation pipeline."""

    def test_basic_summarizer_produces_valid_summary_structure(self):
        """Test that BasicSummarizer always produces a valid Summary object."""
        summarizer = BasicSummarizer()

        # Test with various edge cases
        test_cases = [
            ([], None),  # Empty responses, no voting
            (
                [
                    RoundResponse(
                        round=1,
                        participant="a@b",
                        response="Short",
                        timestamp="2024-01-01T00:00:00",
                    )
                ],
                None,
            ),  # Very short response
            (
                [
                    RoundResponse(
                        round=1,
                        participant="a@b",
                        response="x" * 10000,
                        timestamp="2024-01-01T00:00:00",
                    )
                ],
                None,
            ),  # Very long response
        ]

        for responses, voting in test_cases:
            summary = summarizer.generate_summary(
                question="Test",
                responses=responses,
                voting_result=voting,
            )

            # Validate structure
            assert isinstance(summary, Summary)
            assert isinstance(summary.consensus, str)
            assert len(summary.consensus) > 0
            assert isinstance(summary.key_agreements, list)
            assert len(summary.key_agreements) > 0
            assert isinstance(summary.key_disagreements, list)
            assert len(summary.key_disagreements) > 0
            assert isinstance(summary.final_recommendation, str)
            assert len(summary.final_recommendation) > 0
