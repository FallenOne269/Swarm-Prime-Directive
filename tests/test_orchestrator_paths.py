"""
Tests for SwarmOrchestrator early-exit branches, decision routing, deliverable
failure handling, and multi-cycle context propagation.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swarm_prime.models import (
    AgentMessage,
    AgentRole,
    AlignmentRiskReport,
    ArchitecturalMutationProposals,
    CapabilityDeltaReport,
    CycleDecision,
    CycleState,
    ExperimentalValidationPlan,
    FailureAnalysis,
    PerformanceDelta,
    Proposal,
    ProposalStatus,
    SimulationResult,
)
from swarm_prime.orchestrator import SwarmOrchestrator
from swarm_prime.providers import LLMProvider


# ── Shared Mock ──────────────────────────────────────────────────────────────


class MockLLMProvider(LLMProvider):
    """Deterministic mock LLM for orchestrator tests."""

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        user_msg = messages[-1]["content"] if messages else ""
        for key, response in self._responses.items():
            if key.lower() in user_msg.lower():
                return response
        return json.dumps({
            "title": "Test Proposal",
            "specification": "Test spec",
            "theoretical_justification": "Test justification",
            "resource_requirements": "Low",
            "potential_risks": [],
            "failure_scenarios": [],
        })

    async def complete_structured(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        output_schema: type,
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        raw = await self.complete(system_prompt, messages, temperature, max_tokens)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_proposal(status: ProposalStatus = ProposalStatus.DRAFT, title: str = "Test Upgrade") -> Proposal:
    p = Proposal(
        title=title,
        specification="Add module X",
        theoretical_justification="Improves Y",
        resource_requirements="Low",
    )
    p.status = status
    return p


def _make_simulation(proceed: bool = True) -> SimulationResult:
    return SimulationResult(
        proposal_id="test-id",
        computational_model_output="Analysis complete",
        risk_score=0.1 if proceed else 0.85,
        proceed_recommendation=proceed,
    )


# ── Orchestrator Early-Exit Branch Tests ─────────────────────────────────────


class TestOrchestratorEarlyExits:
    """Covers the four branches in run_cycle that return before step 5."""

    @pytest.mark.asyncio
    async def test_rejected_proposal_skips_simulation(self):
        """Peer-review rejection → DISCARD, simulation never runs."""
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        rejected = _make_proposal(status=ProposalStatus.REJECTED)

        with (
            patch.object(orchestrator, "_step1_propose", new=AsyncMock(return_value=_make_proposal())),
            patch.object(orchestrator.peer_review, "review_proposal", new=AsyncMock(return_value=rejected)),
            patch.object(orchestrator, "_generate_deliverables", new=AsyncMock(return_value=MagicMock())),
        ):
            state = await orchestrator.run_cycle(focus_area="test")

        assert state.decision == CycleDecision.DISCARD
        assert state.simulation is None  # Step 2 was never reached

    @pytest.mark.asyncio
    async def test_rejected_proposal_still_generates_deliverables(self):
        """Even when a proposal is rejected, deliverables are still produced."""
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        rejected = _make_proposal(status=ProposalStatus.REJECTED)
        sentinel = MagicMock()

        with (
            patch.object(orchestrator, "_step1_propose", new=AsyncMock(return_value=_make_proposal())),
            patch.object(orchestrator.peer_review, "review_proposal", new=AsyncMock(return_value=rejected)),
            patch.object(orchestrator, "_generate_deliverables", new=AsyncMock(return_value=sentinel)) as mock_gen,
        ):
            state = await orchestrator.run_cycle(focus_area="test")

        mock_gen.assert_awaited_once()
        assert state.deliverables is sentinel

    @pytest.mark.asyncio
    async def test_simulation_no_proceed_skips_stress_test(self):
        """Simulation recommends against proceeding → DISCARD, steps 3-5 never run."""
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        approved = _make_proposal(status=ProposalStatus.APPROVED)

        with (
            patch.object(orchestrator, "_step1_propose", new=AsyncMock(return_value=_make_proposal())),
            patch.object(orchestrator.peer_review, "review_proposal", new=AsyncMock(return_value=approved)),
            patch.object(orchestrator, "_step2_simulate", new=AsyncMock(return_value=_make_simulation(proceed=False))),
            patch.object(orchestrator, "_generate_deliverables", new=AsyncMock(return_value=MagicMock())),
        ):
            state = await orchestrator.run_cycle(focus_area="test")

        assert state.decision == CycleDecision.DISCARD
        assert state.simulation is not None
        assert state.simulation.proceed_recommendation is False
        assert state.stress_test is None  # Step 3 was skipped

    @pytest.mark.asyncio
    async def test_critical_constraint_violation_halts_before_decision(self):
        """A critical constraint violation after steps 3-4 exits with DISCARD before step 5."""
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        approved = _make_proposal(status=ProposalStatus.APPROVED)
        perf = PerformanceDelta(proposal_id="test-id")

        with (
            patch.object(orchestrator, "_step1_propose", new=AsyncMock(return_value=_make_proposal())),
            patch.object(orchestrator.peer_review, "review_proposal", new=AsyncMock(return_value=approved)),
            patch.object(orchestrator, "_step2_simulate", new=AsyncMock(return_value=_make_simulation(proceed=True))),
            patch.object(orchestrator, "_step3_stress_test", new=AsyncMock(return_value=MagicMock())),
            patch.object(orchestrator, "_step4_measure", new=AsyncMock(return_value=perf)),
            patch.object(orchestrator.constraint_layer, "check_cycle_state", return_value=[MagicMock(severity="critical")]),
            patch.object(orchestrator.constraint_layer, "has_critical_violation", return_value=True),
            patch.object(orchestrator, "_generate_deliverables", new=AsyncMock(return_value=MagicMock())),
        ):
            state = await orchestrator.run_cycle(focus_area="test")

        assert state.decision == CycleDecision.DISCARD
        # Steps 1-4 completed, but step 5 was not reached
        assert state.stress_test is not None
        assert state.performance_delta is not None

    @pytest.mark.asyncio
    async def test_architect_no_structured_data_raises_runtime_error(self):
        """When architect.execute returns no structured_data, RuntimeError is raised."""
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        bare_msg = AgentMessage(
            sender=AgentRole.ARCHITECT,
            content="Here is my proposal as plain text only",
            structured_data=None,
        )

        with patch.object(orchestrator.agents[AgentRole.ARCHITECT], "execute", new=AsyncMock(return_value=bare_msg)):
            with pytest.raises(RuntimeError, match="Architect returned no structured proposal"):
                await orchestrator.run_cycle(focus_area="test")

    @pytest.mark.asyncio
    async def test_failed_cycle_is_still_appended_to_history(self):
        """Even when run_cycle raises, the partial CycleState is saved to history."""
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        bare_msg = AgentMessage(
            sender=AgentRole.ARCHITECT,
            content="plain text",
            structured_data=None,
        )

        with patch.object(orchestrator.agents[AgentRole.ARCHITECT], "execute", new=AsyncMock(return_value=bare_msg)):
            with pytest.raises(RuntimeError):
                await orchestrator.run_cycle(focus_area="test")

        assert len(orchestrator._cycle_history) == 1
        assert orchestrator._cycle_history[0].decision == CycleDecision.DISCARD


# ── Decision Routing Tests ────────────────────────────────────────────────────


class TestOrchestratorDecisions:
    """Tests for the three CycleDecision outcomes from _step5_decide."""

    def _make_state_for_step5(self) -> CycleState:
        state = CycleState(cycle_number=1)
        state.proposal = _make_proposal(status=ProposalStatus.APPROVED)
        state.simulation = _make_simulation(proceed=True)
        state.stress_test = MagicMock()
        state.performance_delta = PerformanceDelta(proposal_id="test-id")
        return state

    @pytest.mark.asyncio
    async def test_step5_decide_adopt(self):
        llm = MockLLMProvider({"collective decision": "ADOPT — All criteria met."})
        decision = await SwarmOrchestrator(llm=llm)._step5_decide(
            self._make_state_for_step5(), {}, "trace"
        )
        assert decision == CycleDecision.ADOPT

    @pytest.mark.asyncio
    async def test_step5_decide_modify(self):
        llm = MockLLMProvider({"collective decision": "MODIFY — Partial success, needs refinement."})
        decision = await SwarmOrchestrator(llm=llm)._step5_decide(
            self._make_state_for_step5(), {}, "trace"
        )
        assert decision == CycleDecision.MODIFY

    @pytest.mark.asyncio
    async def test_step5_decide_discard_when_no_keyword_matches(self):
        """When LLM output contains neither ADOPT nor MODIFY, decision defaults to DISCARD."""
        llm = MockLLMProvider({"collective decision": "DISCARD — Too risky to proceed."})
        decision = await SwarmOrchestrator(llm=llm)._step5_decide(
            self._make_state_for_step5(), {}, "trace"
        )
        assert decision == CycleDecision.DISCARD

    @pytest.mark.asyncio
    async def test_step5_adopt_takes_priority_over_modify(self):
        """If both ADOPT and MODIFY appear, ADOPT wins (checked first)."""
        llm = MockLLMProvider({"collective decision": "ADOPT with MODIFY notes attached."})
        decision = await SwarmOrchestrator(llm=llm)._step5_decide(
            self._make_state_for_step5(), {}, "trace"
        )
        assert decision == CycleDecision.ADOPT


# ── Deliverable Partial-Failure Tests ────────────────────────────────────────


class TestGenerateDeliverablesFailure:
    """_generate_deliverables uses fallback default models when generators raise."""

    @pytest.mark.asyncio
    async def test_all_generators_failing_produces_valid_deliverables(self):
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        # _generate_deliverables uses self._current_cycle (not state.cycle_number) for
        # fallback objects, so we must sync the orchestrator's counter to the desired value.
        orchestrator._current_cycle = 7
        state = CycleState(cycle_number=7)
        state.decision = CycleDecision.ADOPT

        boom = RuntimeError("downstream service unavailable")

        with (
            patch.object(orchestrator.agents[AgentRole.EVALUATOR], "generate_capability_delta", new=AsyncMock(side_effect=boom)),
            patch.object(orchestrator.agents[AgentRole.ALIGNMENT_GUARDIAN], "generate_alignment_report", new=AsyncMock(side_effect=boom)),
            patch.object(orchestrator.agents[AgentRole.SKEPTIC], "generate_failure_analysis", new=AsyncMock(side_effect=boom)),
            patch.object(orchestrator.agents[AgentRole.ARCHITECT], "generate_mutations", new=AsyncMock(side_effect=boom)),
            patch.object(orchestrator.agents[AgentRole.EXPERIMENT_DESIGNER], "generate_validation_plan", new=AsyncMock(side_effect=boom)),
        ):
            deliverables = await orchestrator._generate_deliverables(state, {}, "trace")

        # The whole thing should not raise
        assert deliverables.cycle_number == 7
        assert deliverables.decision == CycleDecision.ADOPT

        # Every fallback carries the correct cycle_number
        assert isinstance(deliverables.capability_delta, CapabilityDeltaReport)
        assert deliverables.capability_delta.cycle_number == 7

        assert isinstance(deliverables.alignment_risk, AlignmentRiskReport)
        assert deliverables.alignment_risk.cycle_number == 7

        assert isinstance(deliverables.failure_analysis, FailureAnalysis)
        assert deliverables.failure_analysis.cycle_number == 7

        assert isinstance(deliverables.architectural_mutations, ArchitecturalMutationProposals)
        assert deliverables.architectural_mutations.cycle_number == 7

        assert isinstance(deliverables.validation_plan, ExperimentalValidationPlan)
        assert deliverables.validation_plan.cycle_number == 7

    @pytest.mark.asyncio
    async def test_single_generator_failure_does_not_affect_others(self):
        """Only the failed deliverable uses the fallback; successful ones use real values."""
        # _generate_deliverables derives cycle_num from self._current_cycle; set it explicitly.
        llm = MockLLMProvider({
            "capability delta": json.dumps({"cycle_number": 2, "summary": "Real summary"}),
            "alignment risk": json.dumps({"cycle_number": 2, "risk_level": "low", "summary": "OK"}),
            "failure analysis": json.dumps({"cycle_number": 2, "summary": "None"}),
            "mutation": json.dumps({"cycle_number": 2, "mutations": [], "summary": "Continue"}),
            "validation": json.dumps({"cycle_number": 2, "summary": "Plan"}),
        })
        orchestrator = SwarmOrchestrator(llm=llm)
        orchestrator._current_cycle = 2
        state = CycleState(cycle_number=2)
        state.decision = CycleDecision.ADOPT

        real_cap_delta = CapabilityDeltaReport(cycle_number=2, summary="Real cap delta")
        real_alignment = AlignmentRiskReport(cycle_number=2, risk_level="medium")
        real_failure = FailureAnalysis(cycle_number=2, summary="One edge case")
        real_mutations = ArchitecturalMutationProposals(cycle_number=2)
        boom = RuntimeError("validation plan service down")

        with (
            patch.object(orchestrator.agents[AgentRole.EVALUATOR], "generate_capability_delta", new=AsyncMock(return_value=real_cap_delta)),
            patch.object(orchestrator.agents[AgentRole.ALIGNMENT_GUARDIAN], "generate_alignment_report", new=AsyncMock(return_value=real_alignment)),
            patch.object(orchestrator.agents[AgentRole.SKEPTIC], "generate_failure_analysis", new=AsyncMock(return_value=real_failure)),
            patch.object(orchestrator.agents[AgentRole.ARCHITECT], "generate_mutations", new=AsyncMock(return_value=real_mutations)),
            patch.object(orchestrator.agents[AgentRole.EXPERIMENT_DESIGNER], "generate_validation_plan", new=AsyncMock(side_effect=boom)),
        ):
            deliverables = await orchestrator._generate_deliverables(state, {}, "trace")

        # Successful deliverables pass through unchanged
        assert deliverables.capability_delta is real_cap_delta
        assert deliverables.alignment_risk is real_alignment
        assert deliverables.failure_analysis is real_failure
        assert deliverables.architectural_mutations is real_mutations
        # Failed one falls back to the default
        assert isinstance(deliverables.validation_plan, ExperimentalValidationPlan)
        assert deliverables.validation_plan.cycle_number == 2


# ── Multi-Cycle Orchestration Tests ──────────────────────────────────────────


class TestRunCycles:
    @pytest.mark.asyncio
    async def test_returns_correct_number_of_states(self):
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        mock_state = CycleState(cycle_number=1)

        with patch.object(orchestrator, "run_cycle", new=AsyncMock(return_value=mock_state)):
            results = await orchestrator.run_cycles(4, focus_area="test")

        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_first_cycle_receives_no_previous_context(self):
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        received: list[dict | None] = []

        async def capture(context=None, focus_area=None):
            received.append(context)
            return CycleState(cycle_number=len(received))

        with patch.object(orchestrator, "run_cycle", side_effect=capture):
            await orchestrator.run_cycles(3)

        assert received[0] is None

    @pytest.mark.asyncio
    async def test_subsequent_cycles_receive_previous_state(self):
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        received: list[dict | None] = []

        async def capture(context=None, focus_area=None):
            received.append(context)
            return CycleState(cycle_number=len(received))

        with patch.object(orchestrator, "run_cycle", side_effect=capture):
            await orchestrator.run_cycles(3)

        assert "previous_cycle" in received[1]
        assert "previous_cycle" in received[2]

    @pytest.mark.asyncio
    async def test_previous_cycle_key_holds_actual_state_object(self):
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        received: list[dict | None] = []
        states: list[CycleState] = []

        async def capture(context=None, focus_area=None):
            received.append(context)
            s = CycleState(cycle_number=len(received))
            states.append(s)
            return s

        with patch.object(orchestrator, "run_cycle", side_effect=capture):
            await orchestrator.run_cycles(2)

        assert received[1]["previous_cycle"] is states[0]


# ── Context Building Tests ────────────────────────────────────────────────────


class TestBuildCycleContext:
    def test_includes_focus_area(self):
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        ctx = orchestrator._build_cycle_context(None, "transfer learning")
        assert ctx["focus_area"] == "transfer learning"

    def test_merges_user_context(self):
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        ctx = orchestrator._build_cycle_context({"custom_key": "custom_value"}, None)
        assert ctx["custom_key"] == "custom_value"

    def test_includes_previous_cycle_summary_when_history_exists(self):
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        prev = CycleState(cycle_number=3)
        prev.decision = CycleDecision.ADOPT
        prev.proposal = _make_proposal(title="Old Upgrade")
        orchestrator._cycle_history.append(prev)

        ctx = orchestrator._build_cycle_context(None, None)

        assert "previous_cycle_summary" in ctx
        summary = ctx["previous_cycle_summary"]
        assert summary["cycle"] == 3
        assert summary["decision"] == "adopt"
        assert summary["proposal_title"] == "Old Upgrade"

    def test_no_previous_summary_when_history_empty(self):
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        ctx = orchestrator._build_cycle_context(None, None)
        assert "previous_cycle_summary" not in ctx

    def test_includes_memory_principles_when_present(self):
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        orchestrator.memory_graph.compressed_principles.append("Shared embeddings aid transfer")

        ctx = orchestrator._build_cycle_context(None, None)

        assert "known_principles" in ctx
        assert "Shared embeddings aid transfer" in ctx["known_principles"]

    def test_includes_failure_patterns_when_present(self):
        orchestrator = SwarmOrchestrator(llm=MockLLMProvider())
        orchestrator.memory_graph.failure_patterns.append("Overfitting on held-out ARC set")

        ctx = orchestrator._build_cycle_context(None, None)

        assert "known_failures" in ctx
        assert "Overfitting on held-out ARC set" in ctx["known_failures"]
