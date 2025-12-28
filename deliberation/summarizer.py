"""AI-powered summary generation for deliberations."""

import logging
import re
from typing import Dict, List, Optional, Union

from adapters.base import BaseCLIAdapter
from adapters.base_http import BaseHTTPAdapter
from models.schema import RoundResponse, Summary, VotingResult

logger = logging.getLogger(__name__)


class DeliberationSummarizer:
    """
    Generates AI-powered summaries of deliberations.

    Uses a designated AI adapter to analyze the full debate and extract:
    - Overall consensus (if any)
    - Key areas of agreement
    - Key areas of disagreement
    - Final recommendation
    """

    def __init__(self, adapter: Union[BaseCLIAdapter, BaseHTTPAdapter], model: str):
        """
        Initialize summarizer.

        Args:
            adapter: Adapter to use for summary generation (CLI or HTTP)
            model: Model identifier to use (e.g., "sonnet" for Claude)
        """
        self.adapter = adapter
        self.model = model

    async def generate_summary(
        self, question: str, responses: List[RoundResponse]
    ) -> Summary:
        """
        Generate summary from deliberation responses.

        Args:
            question: Original deliberation question
            responses: All responses from all rounds

        Returns:
            Summary object with consensus, agreements, disagreements, and recommendation
        """
        # Build formatted debate text
        debate_text = self._format_debate(question, responses)

        # Create summarization prompt
        prompt = self._create_summary_prompt(debate_text)

        try:
            # Get AI summary
            summary_text = await self.adapter.invoke(
                prompt=prompt, model=self.model, context=None
            )

            # Parse the summary
            return self._parse_summary(summary_text)

        except Exception as e:
            logger.error(f"Summary generation failed: {e}", exc_info=True)
            # Return placeholder on error
            return Summary(
                consensus="[Summary generation failed]",
                key_agreements=["Error occurred during summary generation"],
                key_disagreements=[],
                final_recommendation="Please review the full debate below.",
            )

    def _format_debate(self, question: str, responses: List[RoundResponse]) -> str:
        """
        Format debate into readable text.

        Args:
            question: Original question
            responses: All responses

        Returns:
            Formatted debate text
        """
        lines = [f"Question: {question}\n"]

        # Group by round
        rounds: Dict[int, List[RoundResponse]] = {}
        for resp in responses:
            if resp.round not in rounds:
                rounds[resp.round] = []
            rounds[resp.round].append(resp)

        # Format each round
        for round_num in sorted(rounds.keys()):
            lines.append(f"\n--- Round {round_num} ---")
            for resp in rounds[round_num]:
                lines.append(f"\n{resp.participant}:")
                lines.append(resp.response)

        return "\n".join(lines)

    def _create_summary_prompt(self, debate_text: str) -> str:
        """
        Create prompt for summary generation.

        Args:
            debate_text: Formatted debate text

        Returns:
            Prompt for AI model
        """
        return f"""Analyze the following AI deliberation and provide a structured summary.

{debate_text}

Please provide your analysis in the following format:

CONSENSUS:
[A 1-2 sentence statement of the overall consensus, if one was reached.
If no consensus, state "No clear consensus reached" and briefly explain the divide.]

KEY AGREEMENTS:
- [Agreement 1]
- [Agreement 2]
- [Agreement 3]
[List 2-5 key points where all or most participants agreed]

KEY DISAGREEMENTS:
- [Disagreement 1]
- [Disagreement 2]
[List any significant points of disagreement, or state "None" if all agreed]

FINAL RECOMMENDATION:
[1-3 sentences providing the best path forward based on the deliberation]

Please be concise and focus on the substance of the arguments, not formatting or style."""

    def _parse_summary(self, summary_text: str) -> Summary:
        """
        Parse AI-generated summary into Summary object.

        Args:
            summary_text: Raw summary text from AI

        Returns:
            Parsed Summary object
        """
        # Initialize with defaults
        consensus: str = "No consensus information provided"
        agreements: List[str] = []
        disagreements: List[str] = []
        recommendation: str = "No recommendation provided"

        # Split into sections
        sections: Dict[str, str | None] = {
            "consensus": None,
            "agreements": None,
            "disagreements": None,
            "recommendation": None,
        }

        current_section = None
        buffer: List[str] = []

        for line in summary_text.split("\n"):
            line_upper = line.strip().upper()

            # Detect section headers
            if "CONSENSUS:" in line_upper:
                if current_section and buffer:
                    sections[current_section] = "\n".join(buffer).strip()
                current_section = "consensus"
                buffer = []
                # Include any text after the header on the same line
                after_header = line.split(":", 1)[1].strip() if ":" in line else ""
                if after_header:
                    buffer.append(after_header)
            elif "KEY AGREEMENT" in line_upper:
                if current_section and buffer:
                    sections[current_section] = "\n".join(buffer).strip()
                current_section = "agreements"
                buffer = []
            elif "KEY DISAGREEMENT" in line_upper:
                if current_section and buffer:
                    sections[current_section] = "\n".join(buffer).strip()
                current_section = "disagreements"
                buffer = []
            elif (
                "FINAL RECOMMENDATION" in line_upper or "RECOMMENDATION:" in line_upper
            ):
                if current_section and buffer:
                    sections[current_section] = "\n".join(buffer).strip()
                current_section = "recommendation"
                buffer = []
                # Include any text after the header on the same line
                after_header = line.split(":", 1)[1].strip() if ":" in line else ""
                if after_header:
                    buffer.append(after_header)
            elif current_section:
                buffer.append(line)

        # Don't forget the last section
        if current_section and buffer:
            sections[current_section] = "\n".join(buffer).strip()

        # Parse consensus
        if sections["consensus"]:
            consensus = sections["consensus"]

        # Parse agreements (bullet points)
        if sections["agreements"]:
            agreements = self._extract_bullet_points(sections["agreements"])

        # Parse disagreements (bullet points)
        if sections["disagreements"]:
            disagreements = self._extract_bullet_points(sections["disagreements"])

        # Parse recommendation
        if sections["recommendation"]:
            recommendation = sections["recommendation"]

        return Summary(
            consensus=consensus,
            key_agreements=(
                agreements if agreements else ["No specific agreements identified"]
            ),
            key_disagreements=(
                disagreements if disagreements else ["No significant disagreements"]
            ),
            final_recommendation=recommendation,
        )

    def _extract_bullet_points(self, text: str) -> List[str]:
        """
        Extract bullet points from text.

        Args:
            text: Text containing bullet points

        Returns:
            List of bullet point contents
        """
        points = []
        for line in text.split("\n"):
            line = line.strip()
            # Match various bullet formats: -, *, •, 1., etc.
            if line and (
                line.startswith("-")
                or line.startswith("*")
                or line.startswith("•")
                or (len(line) > 2 and line[0].isdigit() and line[1] in ".)")
            ):
                # Remove bullet/number prefix
                if line.startswith(("-", "*", "•")):
                    content = line[1:].strip()
                else:
                    # Remove number prefix (e.g., "1. " or "1) ")
                    content = line.split(None, 1)[1] if " " in line else line

                if content:
                    points.append(content)

        return points


class BasicSummarizer:
    """
    Template-based fallback summarizer that works without AI.

    Generates a structured summary from deliberation data using simple
    text extraction and formatting. Used when all AI summarizers fail
    or are unavailable.
    """

    def generate_summary(
        self,
        question: str,
        responses: List[RoundResponse],
        voting_result: Optional[VotingResult] = None,
    ) -> Summary:
        """
        Generate a basic template-based summary.

        Args:
            question: Original deliberation question
            responses: All responses from all rounds
            voting_result: Optional voting result for consensus info

        Returns:
            Summary object with extracted information
        """
        # Extract main points from each participant
        participant_points = self._extract_participant_points(responses)

        # Build consensus statement
        consensus = self._build_consensus_statement(voting_result, responses)

        # Extract key agreements (points mentioned by multiple participants)
        key_agreements = self._find_agreements(participant_points)

        # Extract key disagreements
        key_disagreements = self._find_disagreements(participant_points, voting_result)

        # Build recommendation
        recommendation = self._build_recommendation(voting_result, question)

        return Summary(
            consensus=f"[Auto-generated fallback summary] {consensus}",
            key_agreements=(
                key_agreements
                if key_agreements
                else ["Unable to extract specific agreements from responses"]
            ),
            key_disagreements=(
                key_disagreements
                if key_disagreements
                else ["Unable to extract specific disagreements from responses"]
            ),
            final_recommendation=recommendation,
        )

    def _extract_participant_points(
        self, responses: List[RoundResponse]
    ) -> Dict[str, List[str]]:
        """
        Extract main points from each participant's responses.

        Args:
            responses: All responses from all rounds

        Returns:
            Dictionary mapping participant IDs to their main points
        """
        points_by_participant: Dict[str, List[str]] = {}

        for response in responses:
            participant = response.participant
            if participant not in points_by_participant:
                points_by_participant[participant] = []

            # Extract key sentences (first sentence of each paragraph)
            paragraphs = response.response.split("\n\n")
            for para in paragraphs:
                para = para.strip()
                if not para or para.startswith("[ERROR"):
                    continue

                # Skip tool requests and vote markers
                if "TOOL_REQUEST" in para or para.startswith("VOTE:"):
                    continue

                # Get first sentence (up to 200 chars)
                sentences = re.split(r"[.!?]", para)
                if sentences and sentences[0].strip():
                    point = sentences[0].strip()[:200]
                    if point and len(point) > 20:  # Skip very short fragments
                        points_by_participant[participant].append(point)

        # Limit to top 3 points per participant
        for participant in points_by_participant:
            points_by_participant[participant] = points_by_participant[participant][:3]

        return points_by_participant

    def _build_consensus_statement(
        self,
        voting_result: Optional[VotingResult],
        responses: List[RoundResponse],
    ) -> str:
        """
        Build a consensus statement based on voting results.

        Args:
            voting_result: Voting result if available
            responses: All responses

        Returns:
            Consensus statement string
        """
        if not voting_result:
            return "No voting data available to determine consensus."

        if voting_result.consensus_reached and voting_result.winning_option:
            total_votes = sum(voting_result.final_tally.values())
            winner_votes = voting_result.final_tally.get(
                voting_result.winning_option, 0
            )
            return (
                f"Consensus reached on '{voting_result.winning_option}' "
                f"with {winner_votes}/{total_votes} votes."
            )
        elif voting_result.final_tally:
            tally_str = ", ".join(
                f"{opt}: {count}" for opt, count in voting_result.final_tally.items()
            )
            return f"No clear consensus. Vote distribution: {tally_str}"
        else:
            return "Deliberation completed without clear voting outcome."

    def _find_agreements(self, participant_points: Dict[str, List[str]]) -> List[str]:
        """
        Find potential areas of agreement based on participant responses.

        Args:
            participant_points: Points extracted from each participant

        Returns:
            List of potential agreement areas
        """
        agreements = []

        # Simple approach: list what each participant emphasized
        for participant, points in participant_points.items():
            if points:
                # Use the first substantive point from each participant
                short_participant = participant.split("@")[0]  # Remove @cli suffix
                agreements.append(f"{short_participant}: {points[0]}")

        if not agreements:
            agreements = [
                "Participants provided analysis but specific agreements could not be extracted"
            ]

        return agreements[:5]  # Limit to 5 items

    def _find_disagreements(
        self,
        participant_points: Dict[str, List[str]],
        voting_result: Optional[VotingResult],
    ) -> List[str]:
        """
        Find potential areas of disagreement.

        Args:
            participant_points: Points from each participant
            voting_result: Voting result if available

        Returns:
            List of potential disagreement areas
        """
        disagreements = []

        # If voting shows different options, there's disagreement
        if voting_result and voting_result.final_tally:
            unique_options = list(voting_result.final_tally.keys())
            if len(unique_options) > 1:
                # Filter out ABSTAIN from disagreement reporting
                substantive_options = [
                    opt for opt in unique_options if opt.upper() != "ABSTAIN"
                ]
                if len(substantive_options) > 1:
                    disagreements.append(
                        f"Participants voted for different options: {', '.join(substantive_options)}"
                    )

        if not disagreements:
            if voting_result and voting_result.consensus_reached:
                disagreements = ["No significant disagreements - consensus was reached"]
            else:
                disagreements = [
                    "Unable to extract specific disagreements from responses"
                ]

        return disagreements

    def _build_recommendation(
        self, voting_result: Optional[VotingResult], question: str
    ) -> str:
        """
        Build a recommendation based on voting outcome.

        Args:
            voting_result: Voting result if available
            question: Original question

        Returns:
            Recommendation string
        """
        if voting_result and voting_result.winning_option:
            return (
                f"Based on the voting outcome, the recommendation is: "
                f"'{voting_result.winning_option}'. "
                f"Review the full debate transcript for detailed reasoning."
            )
        else:
            return (
                "No clear winner emerged from the deliberation. "
                "Review the full debate transcript to understand different perspectives."
            )
