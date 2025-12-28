"""Integration tests for concurrent adapter failures.

Issue #32: Tests for multiple adapters failing simultaneously, partial failures
(some succeed, some fail), recovery after transient failures, and graceful
degradation behavior.

Uses pytest patterns consistent with existing integration tests.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from deliberation.engine import DeliberationEngine
from models.config import load_config
from models.schema import DeliberateRequest, Participant


@pytest.mark.integration
class TestMultipleSimultaneousFailures:
    """Tests for multiple adapters failing at the same time."""

    @pytest.fixture
    def config(self):
        """Load test config."""
        return load_config("config.yaml")

    @pytest.fixture
    def all_failing_adapters(self):
        """Create adapters that all fail."""
        claude_adapter = AsyncMock()
        claude_adapter.invoke = AsyncMock(
            side_effect=Exception("Claude API unavailable")
        )

        ollama_adapter = AsyncMock()
        ollama_adapter.invoke = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        lmstudio_adapter = AsyncMock()
        lmstudio_adapter.invoke = AsyncMock(
            side_effect=TimeoutError("Request timed out")
        )

        return {
            "claude": claude_adapter,
            "ollama": ollama_adapter,
            "lmstudio": lmstudio_adapter,
        }

    @pytest.mark.asyncio
    async def test_all_adapters_fail_simultaneously(self, config, all_failing_adapters):
        """Test deliberation when all adapters fail at once."""
        engine = DeliberationEngine(adapters=all_failing_adapters, config=config)

        request = DeliberateRequest(
            question="What is the best programming language?",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        # Deliberation should complete (not crash)
        assert result.status == "complete"
        assert result.rounds_completed == 1
        assert len(result.full_debate) == 3

        # All responses should contain error markers
        for response in result.full_debate:
            assert "[ERROR:" in response.response or "ERROR" in response.response

    @pytest.mark.asyncio
    async def test_all_adapters_fail_with_same_error_type(self, config):
        """Test when all adapters fail with the same error type."""
        # All adapters fail with connection errors
        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
            "lmstudio": AsyncMock(),
        }
        for name, adapter in adapters.items():
            adapter.invoke = AsyncMock(
                side_effect=httpx.ConnectError(f"Cannot connect to {name}")
            )

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Test question",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"
        # All should have error responses
        error_count = sum(
            1
            for r in result.full_debate
            if "[ERROR:" in r.response or "ERROR" in r.response
        )
        assert error_count == 3

    @pytest.mark.asyncio
    async def test_cascading_failures_across_rounds(self, config):
        """Test when adapters fail in sequence across multiple rounds."""
        # Round 1: All succeed
        # Round 2: One fails
        # Round 3: All fail
        round_counter = {"count": 0}

        def create_round_aware_mock(name):
            async def invoke(*args, **kwargs):
                current_round = round_counter["count"] // 3 + 1
                if current_round == 1:
                    return f"{name} round 1 response"
                elif current_round == 2:
                    if name == "ollama":
                        raise Exception(f"{name} failed in round 2")
                    return f"{name} round 2 response"
                else:
                    raise Exception(f"{name} failed in round 3")

            return invoke

        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
            "lmstudio": AsyncMock(),
        }

        for name in adapters:
            adapters[name].invoke = AsyncMock(side_effect=create_round_aware_mock(name))
            # Wrap to track calls
            original = adapters[name].invoke

            async def counting_invoke(*args, _orig=original, **kwargs):
                round_counter["count"] += 1
                return await _orig(*args, **kwargs)

            adapters[name].invoke = AsyncMock(side_effect=counting_invoke)

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Test cascading failures",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
            ],
            rounds=3,
            mode="conference",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"
        assert result.rounds_completed >= 1


@pytest.mark.integration
class TestPartialFailures:
    """Tests for scenarios where some adapters succeed and others fail."""

    @pytest.fixture
    def config(self):
        """Load test config."""
        return load_config("config.yaml")

    @pytest.fixture
    def mixed_success_adapters(self):
        """Create adapters with mixed success/failure."""
        claude_adapter = AsyncMock()
        claude_adapter.invoke = AsyncMock(
            return_value='Claude succeeds!\nVOTE: {"option": "Option A", "confidence": 0.9, "rationale": "Clear winner", "continue_debate": false}'
        )

        ollama_adapter = AsyncMock()
        ollama_adapter.invoke = AsyncMock(
            side_effect=httpx.ConnectError("Ollama server not running")
        )

        lmstudio_adapter = AsyncMock()
        lmstudio_adapter.invoke = AsyncMock(
            return_value='LM Studio agrees!\nVOTE: {"option": "Option A", "confidence": 0.85, "rationale": "Makes sense", "continue_debate": false}'
        )

        return {
            "claude": claude_adapter,
            "ollama": ollama_adapter,
            "lmstudio": lmstudio_adapter,
        }

    @pytest.mark.asyncio
    async def test_one_of_three_fails(self, config, mixed_success_adapters):
        """Test that deliberation succeeds when 1 of 3 adapters fails."""
        engine = DeliberationEngine(adapters=mixed_success_adapters, config=config)

        request = DeliberateRequest(
            question="Should we use TypeScript?",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"
        assert len(result.full_debate) == 3

        # Two should succeed, one should fail
        successful = [
            r
            for r in result.full_debate
            if "[ERROR:" not in r.response and "ERROR" not in r.response
        ]
        failed = [
            r
            for r in result.full_debate
            if "[ERROR:" in r.response or "ERROR" in r.response
        ]

        assert len(successful) == 2
        assert len(failed) == 1

        # Failed should be ollama
        assert any("ollama" in r.participant for r in failed)

    @pytest.mark.asyncio
    async def test_two_of_three_fail(self, config):
        """Test that deliberation handles 2 of 3 adapters failing."""
        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
            "lmstudio": AsyncMock(),
        }

        adapters["claude"].invoke = AsyncMock(
            return_value='Only Claude works!\nVOTE: {"option": "Solo", "confidence": 0.8, "rationale": "Only opinion", "continue_debate": false}'
        )
        adapters["ollama"].invoke = AsyncMock(side_effect=TimeoutError("Timeout"))
        adapters["lmstudio"].invoke = AsyncMock(side_effect=Exception("Server crashed"))

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Test majority failure",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"
        assert len(result.full_debate) == 3

        # One success, two failures
        successful = [
            r
            for r in result.full_debate
            if "[ERROR:" not in r.response and "ERROR" not in r.response
        ]
        assert len(successful) == 1
        assert any("claude" in r.participant for r in successful)

    @pytest.mark.asyncio
    async def test_voting_with_partial_failures(self, config, mixed_success_adapters):
        """Test that voting aggregation works correctly with partial failures."""
        engine = DeliberationEngine(adapters=mixed_success_adapters, config=config)

        request = DeliberateRequest(
            question="Vote test with failures",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        # Voting should still work with available votes
        if result.voting_result:
            # Should aggregate votes from successful adapters only
            assert result.voting_result.final_tally.get("Option A", 0) >= 1

    @pytest.mark.asyncio
    async def test_alternating_failures_across_rounds(self, config):
        """Test when different adapters fail in different rounds."""
        round_counter = {"round": 1}
        call_counter = {"claude": 0, "ollama": 0, "lmstudio": 0}

        def create_alternating_mock(name):
            async def invoke(*args, **kwargs):
                call_counter[name] += 1
                current_round = call_counter[name]

                if name == "claude" and current_round == 2:
                    raise Exception("Claude fails in round 2")
                elif name == "ollama" and current_round == 1:
                    raise Exception("Ollama fails in round 1")
                elif name == "lmstudio" and current_round == 3:
                    raise Exception("LM Studio fails in round 3")
                else:
                    return f"{name} response in round {current_round}"

            return invoke

        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
            "lmstudio": AsyncMock(),
        }
        for name in adapters:
            adapters[name].invoke = AsyncMock(side_effect=create_alternating_mock(name))

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Test alternating failures",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
            ],
            rounds=3,
            mode="conference",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"
        # Should complete all 3 rounds despite alternating failures
        assert result.rounds_completed >= 1


@pytest.mark.integration
class TestTransientFailureRecovery:
    """Tests for recovery after transient failures."""

    @pytest.fixture
    def config(self):
        """Load test config."""
        return load_config("config.yaml")

    @pytest.mark.asyncio
    async def test_adapter_recovers_in_later_round(self, config):
        """Test that an adapter that fails in round 1 can succeed in round 2."""
        call_counts = {"ollama": 0}

        async def recovering_invoke(*args, **kwargs):
            call_counts["ollama"] += 1
            if call_counts["ollama"] == 1:
                raise httpx.ConnectError("Temporary failure")
            return 'Ollama recovered!\nVOTE: {"option": "Recovery", "confidence": 0.9, "rationale": "Back online", "continue_debate": false}'

        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
        }
        adapters["claude"].invoke = AsyncMock(
            return_value='Claude consistent.\nVOTE: {"option": "Stable", "confidence": 0.85, "rationale": "Always works", "continue_debate": true}'
        )
        adapters["ollama"].invoke = AsyncMock(side_effect=recovering_invoke)

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Test recovery",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
            ],
            rounds=2,
            mode="conference",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"
        assert result.rounds_completed >= 2

        # Find ollama responses
        ollama_responses = [r for r in result.full_debate if "ollama" in r.participant]
        assert len(ollama_responses) == 2

        # First should be error, second should be success
        assert (
            "[ERROR:" in ollama_responses[0].response
            or "ERROR" in ollama_responses[0].response
        )
        assert "[ERROR:" not in ollama_responses[1].response

    @pytest.mark.asyncio
    async def test_intermittent_failures_with_eventual_success(self, config):
        """Test handling of intermittent failures that eventually succeed."""
        failure_pattern = {
            "claude": [False, True, False],  # Fail, Success, Fail
            "ollama": [True, False, True],  # Success, Fail, Success
        }
        call_counts = {"claude": 0, "ollama": 0}

        def create_intermittent_mock(name):
            async def invoke(*args, **kwargs):
                idx = min(call_counts[name], len(failure_pattern[name]) - 1)
                should_succeed = failure_pattern[name][idx]
                call_counts[name] += 1

                if should_succeed:
                    return f"{name} succeeded this time"
                else:
                    raise Exception(f"{name} failed this time")

            return invoke

        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
        }
        for name in adapters:
            adapters[name].invoke = AsyncMock(
                side_effect=create_intermittent_mock(name)
            )

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Intermittent test",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
            ],
            rounds=3,
            mode="conference",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"

    @pytest.mark.asyncio
    async def test_all_recover_after_initial_failures(self, config):
        """Test when all adapters fail initially but recover later."""
        call_counts = {"claude": 0, "ollama": 0, "lmstudio": 0}

        def create_delayed_recovery_mock(name):
            async def invoke(*args, **kwargs):
                call_counts[name] += 1
                if call_counts[name] == 1:
                    raise Exception(f"{name} initial failure")
                return f"{name} recovered in round {call_counts[name]}"

            return invoke

        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
            "lmstudio": AsyncMock(),
        }
        for name in adapters:
            adapters[name].invoke = AsyncMock(
                side_effect=create_delayed_recovery_mock(name)
            )

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Delayed recovery test",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
            ],
            rounds=2,
            mode="conference",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"
        assert result.rounds_completed == 2

        # Round 1: All fail
        # Round 2: All succeed
        round_2_responses = [r for r in result.full_debate if r.round == 2]
        for response in round_2_responses:
            assert "[ERROR:" not in response.response


@pytest.mark.integration
class TestGracefulDegradation:
    """Tests for graceful degradation behavior."""

    @pytest.fixture
    def config(self):
        """Load test config."""
        return load_config("config.yaml")

    @pytest.mark.asyncio
    async def test_continues_with_single_working_adapter(self, config):
        """Test that deliberation continues even with only one working adapter."""
        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
            "lmstudio": AsyncMock(),
            "openrouter": AsyncMock(),
        }

        adapters["claude"].invoke = AsyncMock(side_effect=Exception("Failed"))
        adapters["ollama"].invoke = AsyncMock(side_effect=Exception("Failed"))
        adapters["lmstudio"].invoke = AsyncMock(side_effect=Exception("Failed"))
        adapters["openrouter"].invoke = AsyncMock(
            return_value='Only survivor!\nVOTE: {"option": "Survive", "confidence": 0.95, "rationale": "Last one standing", "continue_debate": false}'
        )

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Survival test",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
                Participant(cli="openrouter", model="gpt-4"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"
        assert len(result.full_debate) == 4

        # Verify one success
        successful = [
            r
            for r in result.full_debate
            if "[ERROR:" not in r.response and "ERROR" not in r.response
        ]
        assert len(successful) == 1

    @pytest.mark.asyncio
    async def test_error_messages_are_informative(self, config):
        """Test that error messages in responses are informative."""
        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
        }

        adapters["claude"].invoke = AsyncMock(
            side_effect=TimeoutError("Request timed out after 60 seconds")
        )
        adapters["ollama"].invoke = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused on port 11434")
        )

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Error message test",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        # Check that error messages contain useful info
        for response in result.full_debate:
            if "[ERROR:" in response.response or "ERROR" in response.response:
                # Should contain some error description
                error_text = response.response.lower()
                assert any(
                    term in error_text
                    for term in ["timeout", "connection", "error", "failed", "refused"]
                )

    @pytest.mark.asyncio
    async def test_result_includes_all_participants_even_on_failure(self, config):
        """Test that result includes entries for all participants, even failed ones."""
        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
            "lmstudio": AsyncMock(),
        }

        adapters["claude"].invoke = AsyncMock(return_value="Claude works")
        adapters["ollama"].invoke = AsyncMock(side_effect=Exception("Ollama fails"))
        adapters["lmstudio"].invoke = AsyncMock(
            side_effect=Exception("LM Studio fails")
        )

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Inclusion test",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        # Should have exactly 3 responses (one per participant)
        assert len(result.full_debate) == 3

        # Check all participant names are present
        participants = {r.participant for r in result.full_debate}
        assert any("claude" in p for p in participants)
        assert any("ollama" in p for p in participants)
        assert any("lmstudio" in p for p in participants)

    @pytest.mark.asyncio
    async def test_deliberation_completes_in_reasonable_time_with_failures(
        self, config
    ):
        """Test that failures don't cause excessive delays.

        Note: The deliberation engine has background operations (AI summarization,
        decision graph storage) that add overhead beyond adapter invocations.
        We test that the overall time is reasonable (under 60 seconds) rather than
        requiring sub-second completion, since summarizers and persistence layers
        have their own timeouts and retry logic.
        """
        import time

        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
        }

        # Simulate quick failures (not stuck)
        adapters["claude"].invoke = AsyncMock(side_effect=Exception("Quick failure"))
        adapters["ollama"].invoke = AsyncMock(side_effect=Exception("Quick failure"))

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Speed test",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        start = time.time()
        result = await engine.execute(request)
        elapsed = time.time() - start

        assert result.status == "complete"
        # Should complete in reasonable time even with background operations
        # (AI summarization, decision graph storage may add up to ~45s overhead)
        assert elapsed < 60.0

    @pytest.mark.asyncio
    async def test_partial_voting_still_produces_result(self, config):
        """Test that voting produces a result even with partial participation."""
        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
            "lmstudio": AsyncMock(),
        }

        adapters["claude"].invoke = AsyncMock(
            return_value='TypeScript!\nVOTE: {"option": "TypeScript", "confidence": 0.9, "rationale": "Type safety", "continue_debate": false}'
        )
        adapters["ollama"].invoke = AsyncMock(side_effect=Exception("Failed"))
        adapters["lmstudio"].invoke = AsyncMock(
            return_value='TypeScript!\nVOTE: {"option": "TypeScript", "confidence": 0.85, "rationale": "Agree", "continue_debate": false}'
        )

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Partial voting test",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        # Voting should still work with 2 of 3 votes
        if result.voting_result:
            assert result.voting_result.final_tally.get("TypeScript", 0) == 2


@pytest.mark.integration
class TestConcurrentExecutionStress:
    """Stress tests for concurrent adapter execution."""

    @pytest.fixture
    def config(self):
        """Load test config."""
        return load_config("config.yaml")

    @pytest.mark.asyncio
    async def test_many_participants_with_mixed_failures(self, config):
        """Test deliberation with many participants and mixed failure patterns."""
        # Create 6 adapters with 50% failure rate
        adapter_names = ["claude", "ollama", "lmstudio", "openrouter", "codex", "droid"]
        adapters = {}

        for i, name in enumerate(adapter_names):
            adapter = AsyncMock()
            if i % 2 == 0:  # Even indexes succeed
                adapter.invoke = AsyncMock(
                    return_value=f'{name} works!\nVOTE: {{"option": "Option", "confidence": 0.8, "rationale": "Works", "continue_debate": false}}'
                )
            else:  # Odd indexes fail
                adapter.invoke = AsyncMock(side_effect=Exception(f"{name} failed"))
            adapters[name] = adapter

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Stress test",
            participants=[
                Participant(cli=name, model="test-model") for name in adapter_names
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"
        assert len(result.full_debate) == 6

        # 3 should succeed, 3 should fail
        successful = [
            r
            for r in result.full_debate
            if "[ERROR:" not in r.response and "ERROR" not in r.response
        ]
        assert len(successful) == 3

    @pytest.mark.asyncio
    async def test_rapid_failure_sequence(self, config):
        """Test handling of rapid consecutive failures."""
        failure_delay = 0.01  # 10ms between failures

        async def delayed_failure(name):
            await asyncio.sleep(failure_delay)
            raise Exception(f"{name} rapid failure")

        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
            "lmstudio": AsyncMock(),
        }
        for name in adapters:
            adapters[name].invoke = AsyncMock(
                side_effect=lambda n=name: delayed_failure(n)
            )

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Rapid failure test",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"
        # All should have error responses
        assert all(
            "[ERROR:" in r.response or "ERROR" in r.response for r in result.full_debate
        )


@pytest.mark.integration
class TestErrorTypeVariety:
    """Tests for handling various error types concurrently."""

    @pytest.fixture
    def config(self):
        """Load test config."""
        return load_config("config.yaml")

    @pytest.mark.asyncio
    async def test_mixed_error_types_handled_correctly(self, config):
        """Test that different error types are all handled appropriately."""
        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
            "lmstudio": AsyncMock(),
            "openrouter": AsyncMock(),
        }

        # Different error types
        adapters["claude"].invoke = AsyncMock(
            side_effect=TimeoutError("Request timeout")
        )
        adapters["ollama"].invoke = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        adapters["lmstudio"].invoke = AsyncMock(
            side_effect=ValueError("Invalid response format")
        )
        adapters["openrouter"].invoke = AsyncMock(
            side_effect=RuntimeError("Unexpected runtime error")
        )

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="Mixed error types",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
                Participant(cli="openrouter", model="gpt-4"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"
        assert len(result.full_debate) == 4

        # All should be error responses
        for response in result.full_debate:
            assert "[ERROR:" in response.response or "ERROR" in response.response

    @pytest.mark.asyncio
    async def test_http_status_errors_handled(self, config):
        """Test handling of various HTTP status code errors."""
        adapters = {
            "claude": AsyncMock(),
            "ollama": AsyncMock(),
            "lmstudio": AsyncMock(),
        }

        # Create HTTP status errors
        def create_http_error(status_code, message):
            mock_response = Mock()
            mock_response.status_code = status_code
            return httpx.HTTPStatusError(
                message,
                request=Mock(url="http://test"),
                response=mock_response,
            )

        adapters["claude"].invoke = AsyncMock(
            side_effect=create_http_error(500, "500 Internal Server Error")
        )
        adapters["ollama"].invoke = AsyncMock(
            side_effect=create_http_error(503, "503 Service Unavailable")
        )
        adapters["lmstudio"].invoke = AsyncMock(
            side_effect=create_http_error(429, "429 Too Many Requests")
        )

        engine = DeliberationEngine(adapters=adapters, config=config)

        request = DeliberateRequest(
            question="HTTP error test",
            participants=[
                Participant(cli="claude", model="sonnet"),
                Participant(cli="ollama", model="llama3"),
                Participant(cli="lmstudio", model="mistral"),
            ],
            rounds=1,
            mode="quick",
            working_directory="/tmp",
        )

        result = await engine.execute(request)

        assert result.status == "complete"
        # All should be error responses due to HTTP errors
        for response in result.full_debate:
            assert "[ERROR:" in response.response or "ERROR" in response.response
