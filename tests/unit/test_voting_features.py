"""Unit tests for new voting features (Issues #27, #28, #33)."""

import pytest
from datetime import datetime

from models.schema import (
    Vote,
    RoundVote,
    VotingResult,
    VoteChange,
    RoundResponse,
)


class TestVoteChange:
    """Tests for Issue #33: VoteChange model."""

    def test_valid_vote_change(self):
        """Test creating a valid VoteChange."""
        change = VoteChange(
            participant="claude@model",
            from_round=1,
            to_round=2,
            previous_option="Option A",
            new_option="Option B",
            previous_confidence=0.8,
            new_confidence=0.9,
            reasoning="Changed my mind based on new evidence",
        )
        assert change.participant == "claude@model"
        assert change.from_round == 1
        assert change.to_round == 2
        assert change.previous_option == "Option A"
        assert change.new_option == "Option B"
        assert change.previous_confidence == 0.8
        assert change.new_confidence == 0.9
        assert change.reasoning == "Changed my mind based on new evidence"

    def test_vote_change_optional_reasoning(self):
        """Test VoteChange with no reasoning."""
        change = VoteChange(
            participant="claude@model",
            from_round=1,
            to_round=2,
            previous_option="Option A",
            new_option="Option B",
            previous_confidence=0.8,
            new_confidence=0.9,
        )
        assert change.reasoning is None


class TestVotingResultEnhancements:
    """Tests for enhanced VotingResult fields (Issues #27, #28, #33)."""

    def test_weighted_tally(self):
        """Test Issue #27: weighted_tally field."""
        result = VotingResult(
            final_tally={"Option A": 2, "Option B": 1},
            weighted_tally={"Option A": 1.7, "Option B": 0.9},
            votes_by_round=[],
            consensus_reached=True,
            winning_option="Option A",
        )
        assert result.weighted_tally == {"Option A": 1.7, "Option B": 0.9}

    def test_weighted_tally_defaults_to_empty(self):
        """Test weighted_tally has default empty dict."""
        result = VotingResult(
            final_tally={"Option A": 1},
            votes_by_round=[],
            consensus_reached=True,
            winning_option="Option A",
        )
        assert result.weighted_tally == {}

    def test_abstain_count(self):
        """Test Issue #28: abstain_count field."""
        result = VotingResult(
            final_tally={"Option A": 2},
            votes_by_round=[],
            consensus_reached=True,
            winning_option="Option A",
            abstain_count=1,
        )
        assert result.abstain_count == 1

    def test_abstain_count_defaults_to_zero(self):
        """Test abstain_count defaults to 0."""
        result = VotingResult(
            final_tally={"Option A": 1},
            votes_by_round=[],
            consensus_reached=True,
            winning_option="Option A",
        )
        assert result.abstain_count == 0

    def test_abstain_rate_by_model(self):
        """Test Issue #28: abstain_rate_by_model field."""
        result = VotingResult(
            final_tally={"Option A": 2},
            votes_by_round=[],
            consensus_reached=True,
            winning_option="Option A",
            abstain_rate_by_model={"claude": 0.5, "codex": 0.0},
        )
        assert result.abstain_rate_by_model["claude"] == 0.5
        assert result.abstain_rate_by_model["codex"] == 0.0

    def test_vote_changes_list(self):
        """Test Issue #33: vote_changes field."""
        change = VoteChange(
            participant="claude",
            from_round=1,
            to_round=2,
            previous_option="A",
            new_option="B",
            previous_confidence=0.8,
            new_confidence=0.9,
        )
        result = VotingResult(
            final_tally={"Option B": 2},
            votes_by_round=[],
            consensus_reached=True,
            winning_option="Option B",
            vote_changes=[change],
        )
        assert len(result.vote_changes) == 1
        assert result.vote_changes[0].participant == "claude"

    def test_vote_stability(self):
        """Test Issue #33: vote_stability field."""
        result = VotingResult(
            final_tally={"Option A": 3},
            votes_by_round=[],
            consensus_reached=True,
            winning_option="Option A",
            vote_stability=0.67,
        )
        assert result.vote_stability == 0.67

    def test_vote_stability_defaults_to_one(self):
        """Test vote_stability defaults to 1.0 (no changes)."""
        result = VotingResult(
            final_tally={"Option A": 1},
            votes_by_round=[],
            consensus_reached=True,
            winning_option="Option A",
        )
        assert result.vote_stability == 1.0


class TestVoteOptionIncludesAbstain:
    """Test that ABSTAIN is documented in Vote option description."""

    def test_vote_allows_abstain_option(self):
        """Test Issue #28: Vote can have ABSTAIN as option."""
        vote = Vote(
            option="ABSTAIN",
            confidence=0.0,
            rationale="Not enough information to make a decision",
        )
        assert vote.option == "ABSTAIN"
        assert vote.confidence == 0.0
