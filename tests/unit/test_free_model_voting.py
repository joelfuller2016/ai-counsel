"""Unit tests for free model vote parsing reliability.

Tests vote parsing from various free model response formats,
including edge cases like malformed votes, missing votes, and ambiguous votes.

Free models (Ollama, LMStudio, local LlamaCpp) often produce less structured
output compared to commercial APIs. This test suite ensures reliable vote
extraction across all response formats.
"""

import pytest

from deliberation.engine import DeliberationEngine
from models.schema import Vote


class TestVoteParsingBasic:
    """Tests for basic vote parsing functionality."""

    def setup_method(self):
        """Set up a minimal engine for vote parsing tests."""
        # Create engine with no adapters (only using _parse_vote)
        self.engine = DeliberationEngine(adapters={})

    def test_parse_standard_vote(self):
        """Test parsing a standard well-formed vote."""
        response = """After careful analysis of the options, I believe Option A is best.

VOTE: {"option": "Option A", "confidence": 0.85, "rationale": "Lower risk and better fit"}
"""
        vote, reason = self.engine._parse_vote(response, "test-model")

        assert vote is not None
        assert reason == ""
        assert vote.option == "Option A"
        assert vote.confidence == 0.85
        assert vote.rationale == "Lower risk and better fit"
        assert vote.continue_debate is True  # Default value

    def test_parse_vote_with_continue_debate_false(self):
        """Test parsing vote with continue_debate set to false."""
        response = """I'm confident in my analysis.

VOTE: {"option": "Yes", "confidence": 0.95, "rationale": "Clear consensus", "continue_debate": false}
"""
        vote, reason = self.engine._parse_vote(response, "test-model")

        assert vote is not None
        assert vote.option == "Yes"
        assert vote.continue_debate is False

    def test_parse_vote_minimum_fields(self):
        """Test parsing vote with only required fields."""
        response = """VOTE: {"option": "Approve", "confidence": 0.7, "rationale": "Acceptable"}"""

        vote, reason = self.engine._parse_vote(response, "test-model")

        assert vote is not None
        assert vote.option == "Approve"
        assert vote.confidence == 0.7

    def test_parse_vote_confidence_boundary_zero(self):
        """Test parsing vote with confidence at 0.0 boundary."""
        response = """VOTE: {"option": "Uncertain", "confidence": 0.0, "rationale": "No idea"}"""

        vote, reason = self.engine._parse_vote(response, "test-model")

        assert vote is not None
        assert vote.confidence == 0.0

    def test_parse_vote_confidence_boundary_one(self):
        """Test parsing vote with confidence at 1.0 boundary."""
        response = """VOTE: {"option": "Certain", "confidence": 1.0, "rationale": "Absolutely sure"}"""

        vote, reason = self.engine._parse_vote(response, "test-model")

        assert vote is not None
        assert vote.confidence == 1.0


class TestVoteParsingFreeModelFormats:
    """Tests for vote parsing from various free model response formats.

    Free models like Ollama (llama2, mistral), LMStudio, and local LlamaCpp
    often produce responses with different formatting, extra whitespace,
    markdown artifacts, or unusual JSON formatting.
    """

    def setup_method(self):
        """Set up engine for testing."""
        self.engine = DeliberationEngine(adapters={})

    def test_parse_vote_extra_whitespace(self):
        """Test parsing vote with extra whitespace around JSON."""
        response = """Here is my analysis...

VOTE:    { "option" : "Option B" , "confidence" : 0.8 , "rationale" : "More practical" }

Some trailing text.
"""
        vote, reason = self.engine._parse_vote(response, "ollama-llama2")

        assert vote is not None
        assert vote.option == "Option B"
        assert vote.confidence == 0.8

    def test_parse_vote_multiline_json(self):
        """Test parsing vote with JSON spread across multiple lines."""
        response = """After analysis:

VOTE: {
    "option": "Refactor",
    "confidence": 0.75,
    "rationale": "Improves maintainability"
}
"""
        vote, reason = self.engine._parse_vote(response, "lmstudio-model")

        assert vote is not None
        assert vote.option == "Refactor"

    def test_parse_vote_with_markdown_formatting(self):
        """Test parsing vote embedded in markdown output."""
        response = """## Analysis

The proposed solution addresses the core issue.

### Decision

**VOTE: {"option": "Approve", "confidence": 0.9, "rationale": "Well designed"}**

### Next Steps
...
"""
        vote, reason = self.engine._parse_vote(response, "mistral-7b")

        assert vote is not None
        assert vote.option == "Approve"

    def test_parse_vote_with_code_block(self):
        """Test parsing vote that appears in a code block (common in free models)."""
        response = """Here is my vote:

```json
VOTE: {"option": "Implement", "confidence": 0.85, "rationale": "Good approach"}
```
"""
        vote, reason = self.engine._parse_vote(response, "codellama")

        assert vote is not None
        assert vote.option == "Implement"

    def test_parse_vote_unicode_in_rationale(self):
        """Test parsing vote with unicode characters in rationale."""
        response = """VOTE: {"option": "Yes", "confidence": 0.8, "rationale": "Good approach - ready to go! \\u2713"}"""

        vote, reason = self.engine._parse_vote(response, "ollama-model")

        assert vote is not None
        assert vote.option == "Yes"

    def test_parse_vote_escaped_quotes_in_rationale(self):
        """Test parsing vote with escaped quotes in rationale."""
        response = r"""VOTE: {"option": "Reject", "confidence": 0.7, "rationale": "The \"quick fix\" approach is too risky"}"""

        vote, reason = self.engine._parse_vote(response, "local-model")

        assert vote is not None
        assert vote.option == "Reject"
        assert '"quick fix"' in vote.rationale

    def test_parse_vote_with_newlines_in_rationale(self):
        """Test parsing vote with newlines in rationale (common in verbose models)."""
        # This tests JSON with escaped newlines
        response = r"""VOTE: {"option": "Option A", "confidence": 0.8, "rationale": "First reason.\nSecond reason."}"""

        vote, reason = self.engine._parse_vote(response, "verbose-model")

        assert vote is not None
        assert vote.option == "Option A"

    def test_parse_vote_last_occurrence_used(self):
        """Test that the last VOTE occurrence is used (skips examples in prompt)."""
        response = """As an example, you might write:
VOTE: {"option": "Example", "confidence": 0.5, "rationale": "This is just an example"}

Now for my actual vote:
VOTE: {"option": "Real Choice", "confidence": 0.9, "rationale": "This is my actual decision"}
"""
        vote, reason = self.engine._parse_vote(response, "free-model")

        assert vote is not None
        assert vote.option == "Real Choice"
        assert vote.confidence == 0.9

    def test_parse_vote_integer_confidence(self):
        """Test parsing vote where confidence is an integer (common in some models)."""
        response = (
            """VOTE: {"option": "Yes", "confidence": 1, "rationale": "Certain"}"""
        )

        vote, reason = self.engine._parse_vote(response, "free-model")

        assert vote is not None
        assert vote.confidence == 1.0


class TestMalformedVotes:
    """Tests for handling malformed vote responses."""

    def setup_method(self):
        """Set up engine for testing."""
        self.engine = DeliberationEngine(adapters={})

    def test_malformed_invalid_json(self):
        """Test handling of invalid JSON syntax."""
        response = (
            """VOTE: {"option": "Yes", "confidence": 0.8, rationale: "Missing quote"}"""
        )

        vote, reason = self.engine._parse_vote(response, "bad-json-model")

        assert vote is None
        assert reason == "invalid_json"

    def test_malformed_missing_closing_brace(self):
        """Test handling of unclosed JSON object."""
        # Pad response to exceed 500 char threshold
        padding = "A" * 500
        response = f'''{padding}

VOTE: {{"option": "Yes", "confidence": 0.8, "rationale": "Oops"'''

        vote, reason = self.engine._parse_vote(response, "truncated-model")

        assert vote is None
        # The regex pattern VOTE: {{.+?}} requires a closing brace
        # An unclosed JSON object won't match, so we get no_vote_marker
        assert reason == "no_vote_marker"

    def test_malformed_missing_required_field_option(self):
        """Test handling of vote missing required 'option' field."""
        response = """VOTE: {"confidence": 0.8, "rationale": "No option specified"}"""

        vote, reason = self.engine._parse_vote(response, "incomplete-model")

        assert vote is None
        assert reason == "validation_error"

    def test_malformed_missing_required_field_confidence(self):
        """Test handling of vote missing required 'confidence' field."""
        response = """VOTE: {"option": "Yes", "rationale": "No confidence"}"""

        vote, reason = self.engine._parse_vote(response, "incomplete-model")

        assert vote is None
        assert reason == "validation_error"

    def test_malformed_missing_required_field_rationale(self):
        """Test handling of vote missing required 'rationale' field."""
        response = """VOTE: {"option": "Yes", "confidence": 0.8}"""

        vote, reason = self.engine._parse_vote(response, "incomplete-model")

        assert vote is None
        assert reason == "validation_error"

    def test_malformed_confidence_out_of_range_high(self):
        """Test handling of confidence > 1.0."""
        response = """VOTE: {"option": "Yes", "confidence": 1.5, "rationale": "Too confident"}"""

        vote, reason = self.engine._parse_vote(response, "overconfident-model")

        assert vote is None
        assert reason == "validation_error"

    def test_malformed_confidence_out_of_range_low(self):
        """Test handling of confidence < 0.0."""
        response = """VOTE: {"option": "Yes", "confidence": -0.5, "rationale": "Negative confidence"}"""

        vote, reason = self.engine._parse_vote(response, "negative-model")

        assert vote is None
        assert reason == "validation_error"

    def test_malformed_confidence_not_number(self):
        """Test handling of non-numeric confidence value."""
        response = """VOTE: {"option": "Yes", "confidence": "high", "rationale": "String confidence"}"""

        vote, reason = self.engine._parse_vote(response, "string-confidence-model")

        assert vote is None
        assert reason == "validation_error"

    def test_malformed_null_values(self):
        """Test handling of null values in vote fields."""
        response = (
            """VOTE: {"option": null, "confidence": 0.8, "rationale": "Null option"}"""
        )

        vote, reason = self.engine._parse_vote(response, "null-model")

        assert vote is None
        assert reason == "validation_error"

    def test_malformed_array_instead_of_object(self):
        """Test handling of array instead of object."""
        # Pad response to exceed 500 char threshold
        padding = "A" * 500
        response = f"""{padding}

VOTE: ["Yes", 0.8, "Rationale"]"""

        vote, reason = self.engine._parse_vote(response, "array-model")

        assert vote is None
        # Array doesn't match the VOTE: {json} pattern, so no vote marker found
        assert reason == "no_vote_marker"

    def test_malformed_extra_fields_ignored(self):
        """Test that extra unexpected fields are gracefully handled."""
        response = """VOTE: {"option": "Yes", "confidence": 0.8, "rationale": "Valid", "extra_field": "ignored", "another": 123}"""

        vote, reason = self.engine._parse_vote(response, "extra-fields-model")

        # Pydantic should ignore extra fields by default
        assert vote is not None
        assert vote.option == "Yes"


class TestMissingVotes:
    """Tests for handling responses with missing votes."""

    def setup_method(self):
        """Set up engine for testing."""
        self.engine = DeliberationEngine(adapters={})

    def test_missing_no_vote_marker(self):
        """Test response with no VOTE marker at all."""
        # Pad to exceed 500 char threshold to get no_vote_marker instead of response_too_short
        padding = "A" * 400
        response = f"""I think Option A is better because it's more practical.
The implementation would be straightforward and the risks are minimal.
Overall, I support moving forward with this approach.
{padding}"""

        vote, reason = self.engine._parse_vote(response, "no-vote-model")

        assert vote is None
        assert reason == "no_vote_marker"

    def test_missing_vote_marker_different_case(self):
        """Test that vote marker is case-sensitive (Vote: won't match VOTE:)."""
        # Pad to exceed 500 char threshold
        padding = "B" * 500
        response = f"""{padding}
Vote: {{"option": "Yes", "confidence": 0.8, "rationale": "Test"}}"""

        vote, reason = self.engine._parse_vote(response, "wrong-case-model")

        # Current implementation uses case-sensitive match
        assert vote is None
        assert reason == "no_vote_marker"

    def test_missing_short_response(self):
        """Test very short response (likely error or truncation)."""
        response = "Error: timeout"

        vote, reason = self.engine._parse_vote(response, "error-model")

        assert vote is None
        assert reason == "response_too_short"

    def test_missing_tool_focus_no_vote(self):
        """Test response focused on tool requests without voting."""
        # Pad to exceed 500 char threshold to get tool_focus_no_vote
        padding = "C" * 450
        response = f"""Let me gather more information.

TOOL_REQUEST: {{"name": "read_file", "arguments": {{"path": "src/main.py"}}}}

I'll analyze the results before making a decision.
{padding}"""

        vote, reason = self.engine._parse_vote(response, "tool-focus-model")

        assert vote is None
        assert reason == "tool_focus_no_vote"

    def test_missing_empty_response(self):
        """Test empty response handling."""
        response = ""

        vote, reason = self.engine._parse_vote(response, "empty-model")

        assert vote is None
        assert reason == "response_too_short"

    def test_missing_whitespace_only(self):
        """Test whitespace-only response handling."""
        response = "   \n\n   \t   "

        vote, reason = self.engine._parse_vote(response, "whitespace-model")

        assert vote is None
        assert reason == "response_too_short"


class TestAmbiguousVotes:
    """Tests for handling ambiguous vote responses."""

    def setup_method(self):
        """Set up engine for testing."""
        self.engine = DeliberationEngine(adapters={})

    def test_ambiguous_vote_in_quoted_context(self):
        """Test vote mentioned in quoted context vs actual vote."""
        response = """The user asked about voting, and mentioned "VOTE: ..." as a format.

Here is my actual vote:
VOTE: {"option": "Option B", "confidence": 0.85, "rationale": "Best choice after analysis"}
"""
        vote, reason = self.engine._parse_vote(response, "quoted-model")

        assert vote is not None
        assert vote.option == "Option B"

    def test_ambiguous_multiple_valid_votes(self):
        """Test response with multiple valid vote markers (takes last)."""
        response = """First attempt:
VOTE: {"option": "Option A", "confidence": 0.6, "rationale": "Initial thought"}

After more consideration:
VOTE: {"option": "Option B", "confidence": 0.9, "rationale": "Better after reflection"}
"""
        vote, reason = self.engine._parse_vote(response, "reconsidering-model")

        # Should take the last vote
        assert vote is not None
        assert vote.option == "Option B"
        assert vote.confidence == 0.9

    def test_ambiguous_vote_marker_in_explanation(self):
        """Test VOTE: mentioned but followed by invalid content, then valid vote."""
        response = """You should format your response like VOTE: followed by JSON.

Here's my vote:
VOTE: {"option": "Approve", "confidence": 0.8, "rationale": "Good proposal"}
"""
        vote, reason = self.engine._parse_vote(response, "explanation-model")

        assert vote is not None
        assert vote.option == "Approve"

    def test_ambiguous_option_with_special_characters(self):
        """Test vote with special characters in option name."""
        response = """VOTE: {"option": "Option A (v2.0)", "confidence": 0.75, "rationale": "Latest version"}"""

        vote, reason = self.engine._parse_vote(response, "special-char-model")

        assert vote is not None
        assert vote.option == "Option A (v2.0)"

    def test_ambiguous_long_rationale(self):
        """Test vote with very long rationale (common in verbose models)."""
        long_rationale = "A" * 500  # 500 character rationale
        response = f"""VOTE: {{"option": "Yes", "confidence": 0.8, "rationale": "{long_rationale}"}}"""

        vote, reason = self.engine._parse_vote(response, "verbose-model")

        assert vote is not None
        assert len(vote.rationale) == 500

    def test_ambiguous_empty_option(self):
        """Test vote with empty string option."""
        response = """VOTE: {"option": "", "confidence": 0.8, "rationale": "Empty option test"}"""

        vote, reason = self.engine._parse_vote(response, "empty-option-model")

        # Pydantic allows empty strings, validation should pass
        assert vote is not None
        assert vote.option == ""


class TestAbstainVotes:
    """Tests for abstain vote creation."""

    def setup_method(self):
        """Set up engine for testing."""
        self.engine = DeliberationEngine(adapters={})

    def test_create_abstain_vote_no_vote_marker(self):
        """Test abstain vote created for missing vote marker."""
        vote = self.engine._create_abstain_vote("test-model", "no_vote_marker")

        assert vote.option == "ABSTAIN"
        assert vote.confidence == 0.0
        assert "Did not include a VOTE section" in vote.rationale
        assert vote.continue_debate is True

    def test_create_abstain_vote_invalid_json(self):
        """Test abstain vote created for invalid JSON."""
        vote = self.engine._create_abstain_vote("test-model", "invalid_json")

        assert vote.option == "ABSTAIN"
        assert vote.confidence == 0.0
        assert "malformed" in vote.rationale

    def test_create_abstain_vote_validation_error(self):
        """Test abstain vote created for validation error."""
        vote = self.engine._create_abstain_vote("test-model", "validation_error")

        assert vote.option == "ABSTAIN"
        assert "failed validation" in vote.rationale

    def test_create_abstain_vote_tool_focus(self):
        """Test abstain vote created for tool-focused response."""
        vote = self.engine._create_abstain_vote("test-model", "tool_focus_no_vote")

        assert vote.option == "ABSTAIN"
        assert "tool requests" in vote.rationale

    def test_create_abstain_vote_short_response(self):
        """Test abstain vote created for short response."""
        vote = self.engine._create_abstain_vote("test-model", "response_too_short")

        assert vote.option == "ABSTAIN"
        assert "too short" in vote.rationale

    def test_create_abstain_vote_unknown_reason(self):
        """Test abstain vote with unknown reason."""
        vote = self.engine._create_abstain_vote("test-model", "some_new_reason")

        assert vote.option == "ABSTAIN"
        assert "some_new_reason" in vote.rationale


class TestVoteRetry:
    """Tests for vote retry logic."""

    def setup_method(self):
        """Set up engine for testing."""
        self.engine = DeliberationEngine(adapters={})

    def test_needs_vote_retry_no_vote(self):
        """Test response needs retry when no vote present."""
        response = (
            """This is a substantial analysis of the problem.
We should consider multiple factors including performance,
maintainability, and user experience. The proposed solution
addresses most concerns effectively."""
            * 3
        )  # Make it long enough

        result = self.engine._needs_vote_retry(response)

        assert result is True

    def test_needs_vote_retry_has_vote(self):
        """Test response does not need retry when vote present."""
        response = """Analysis here...
VOTE: {"option": "Yes", "confidence": 0.8, "rationale": "Good approach"}"""

        result = self.engine._needs_vote_retry(response)

        assert result is False

    def test_needs_vote_retry_short_response(self):
        """Test short response does not trigger retry."""
        response = "Error occurred"

        result = self.engine._needs_vote_retry(response)

        assert result is False

    def test_needs_vote_retry_error_response(self):
        """Test error response does not trigger retry."""
        response = "[ERROR: Timeout after 60 seconds]"

        result = self.engine._needs_vote_retry(response)

        assert result is False

    def test_build_vote_retry_prompt(self):
        """Test building a retry prompt for vote extraction."""
        original = "Long analysis about the problem..." * 20

        prompt = self.engine._build_vote_retry_prompt(original)

        assert "Your previous response" in prompt
        assert "VOTE:" in prompt
        assert '"option"' in prompt
        assert '"confidence"' in prompt
        assert '"rationale"' in prompt


class TestFreeModelVotingIntegration:
    """Integration-style tests simulating free model responses."""

    def setup_method(self):
        """Set up engine for testing."""
        self.engine = DeliberationEngine(adapters={})

    def test_ollama_llama2_style_response(self):
        """Test parsing vote from typical Ollama Llama2 response format."""
        response = """Based on my analysis of the code and the proposed changes, I have the following observations:

1. The refactoring improves code readability
2. The new structure is more maintainable
3. Performance impact appears minimal

After careful consideration of these factors, I believe we should proceed with the refactoring.

VOTE: {"option": "Approve refactoring", "confidence": 0.85, "rationale": "Benefits outweigh the migration effort"}

Let me know if you have any questions about my analysis."""

        vote, reason = self.engine._parse_vote(response, "ollama-llama2-7b")

        assert vote is not None
        assert vote.option == "Approve refactoring"
        assert vote.confidence == 0.85

    def test_mistral_7b_style_response(self):
        """Test parsing vote from typical Mistral 7B response format."""
        response = """# Analysis

## Key Points
- The implementation is solid
- Test coverage is adequate
- Documentation could be improved

## Recommendation

The code quality meets our standards. Minor documentation improvements can be addressed in a follow-up PR.

---

VOTE: {"option": "Merge with suggestions", "confidence": 0.8, "rationale": "Good implementation, minor improvements needed"}

---"""

        vote, reason = self.engine._parse_vote(response, "mistral-7b")

        assert vote is not None
        assert vote.option == "Merge with suggestions"

    def test_codellama_style_response(self):
        """Test parsing vote from typical CodeLlama response format."""
        response = """```python
# Analysis of the proposed changes
# ...
```

Looking at the code:
1. Architecture follows best practices
2. Error handling is comprehensive
3. No security concerns identified

VOTE: {"option": "Ship it", "confidence": 0.9, "rationale": "Production ready"}"""

        vote, reason = self.engine._parse_vote(response, "codellama-13b")

        assert vote is not None
        assert vote.option == "Ship it"

    def test_phi_3_style_response(self):
        """Test parsing vote from typical Phi-3 response format (concise)."""
        response = """Quick assessment: Changes look good. No issues found.
VOTE: {"option": "LGTM", "confidence": 0.88, "rationale": "Clean implementation"}"""

        vote, reason = self.engine._parse_vote(response, "phi-3")

        assert vote is not None
        assert vote.option == "LGTM"
        assert vote.confidence == 0.88

    def test_qwen_style_response(self):
        """Test parsing vote from typical Qwen response format."""
        response = """After reviewing the proposal, I want to share my thoughts:

The approach is well-structured and addresses the core requirements. The implementation details are sound, and the testing strategy covers the main scenarios.

My conclusion:

VOTE: {
  "option": "Proceed with implementation",
  "confidence": 0.82,
  "rationale": "Well-designed solution that meets requirements"
}

I'm available to discuss further if needed."""

        vote, reason = self.engine._parse_vote(response, "qwen-7b")

        assert vote is not None
        assert vote.option == "Proceed with implementation"


class TestCostTracker:
    """Tests for CostTracker cost savings calculation."""

    def test_track_free_model_invocation(self):
        """Test tracking a free model invocation."""
        from deliberation.engine import CostTracker

        tracker = CostTracker()
        tracker.track_invocation(
            adapter_name="ollama",
            model="llama2-7b",
            input_text="A" * 4000,  # 1000 tokens
            output_text="B" * 2000,  # 500 tokens
        )

        assert tracker.free_invocations == 1
        assert tracker.paid_invocations == 0
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500

    def test_track_paid_model_invocation(self):
        """Test tracking a paid model invocation."""
        from deliberation.engine import CostTracker

        tracker = CostTracker()
        tracker.track_invocation(
            adapter_name="claude",
            model="claude-sonnet-4-5-20250929",
            input_text="A" * 4000,  # 1000 tokens
            output_text="B" * 2000,  # 500 tokens
        )

        assert tracker.free_invocations == 0
        assert tracker.paid_invocations == 1
        assert tracker.actual_cost > 0

    def test_savings_calculation(self):
        """Test cost savings calculation for free models."""
        from deliberation.engine import CostTracker

        tracker = CostTracker()
        # Free model
        tracker.track_invocation(
            adapter_name="lmstudio",
            model="local-model",
            input_text="A" * 4000,
            output_text="B" * 2000,
        )

        report = tracker.get_savings_report()

        assert report.actual_cost == 0.0  # Free model
        assert report.estimated_paid_cost > 0  # Would have cost something
        assert report.savings == report.estimated_paid_cost  # All savings
        assert report.free_model_invocations == 1
        assert report.paid_model_invocations == 0

    def test_mixed_model_invocations(self):
        """Test tracking mixed free and paid model invocations."""
        from deliberation.engine import CostTracker

        tracker = CostTracker()

        # Free model
        tracker.track_invocation(
            adapter_name="ollama",
            model="mistral-7b",
            input_text="A" * 4000,
            output_text="B" * 2000,
        )

        # Paid model
        tracker.track_invocation(
            adapter_name="codex",
            model="gpt-5.1-codex",
            input_text="A" * 4000,
            output_text="B" * 2000,
        )

        report = tracker.get_savings_report()

        assert report.free_model_invocations == 1
        assert report.paid_model_invocations == 1
        assert report.actual_cost > 0  # Paid model cost
        assert report.estimated_paid_cost > report.actual_cost  # Both models
        assert report.savings > 0  # Some savings from free model

    def test_cost_breakdown_by_model(self):
        """Test cost breakdown tracks per-model costs."""
        from deliberation.engine import CostTracker

        tracker = CostTracker()

        tracker.track_invocation(
            adapter_name="claude",
            model="opus",
            input_text="A" * 4000,
            output_text="B" * 2000,
        )

        tracker.track_invocation(
            adapter_name="claude",
            model="sonnet",
            input_text="A" * 4000,
            output_text="B" * 2000,
        )

        report = tracker.get_savings_report()

        assert "opus@claude" in report.cost_breakdown
        assert "sonnet@claude" in report.cost_breakdown
        assert (
            report.cost_breakdown["opus@claude"]
            > report.cost_breakdown["sonnet@claude"]
        )

    def test_reset_clears_state(self):
        """Test reset clears all tracking state."""
        from deliberation.engine import CostTracker

        tracker = CostTracker()
        tracker.track_invocation(
            adapter_name="ollama",
            model="llama2",
            input_text="A" * 4000,
            output_text="B" * 2000,
        )

        tracker.reset()

        assert tracker.free_invocations == 0
        assert tracker.paid_invocations == 0
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.actual_cost == 0.0
        assert tracker.estimated_cost == 0.0
        assert len(tracker.cost_breakdown) == 0
