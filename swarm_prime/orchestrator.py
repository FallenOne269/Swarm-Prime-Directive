"""
Swarm Prime Directive — Orchestrator
Drives the 6-step recursive improvement loop (Section 3).

Step 1: Propose Capability Upgrade (Architect)
Step 2: Simulate Consequences (All agents)
Step 3: Stress Test Across Unrelated Domains (Evaluator)
Step 4: Measure Performance Delta (Experiment Designer)
Step 5: Decide Adopt / Modify / Discard (Collective)
Step 6: Update Shared Memory Graph (Memory Curator)

Plus: Peer review gating, constraint enforcement, deliverable generation, meta-cognition.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from swarm_prime.agents.alignment_guardian import AlignmentGuardianAgent
from swarm_prime.agents.architect import ArchitectAgent
from swarm_prime.agents.evaluator import EvaluatorAgent
from swarm_prime.agents.experiment_designer import ExperimentDesignerAgent
from swarm_prime.agents.memory_curator import MemoryCuratorAgent
from swarm_prime.agents.skeptic import SkepticAgent
from swarm_prime.config import SwarmConfig
from swarm_prime.constraints import ConstraintLayer
from swarm_prime.metacognition import MetaCognitionEngine
from swarm_prime.models import (
    AgentRole,
    AlignmentRiskReport,
    ArchitecturalMutationProposals,
    CapabilityDeltaReport,
    CycleDecision,
    CycleDeliverables,
    CycleState,
    ExperimentalValidationPlan,
    FailureAnalysis,
    MemoryGraph,
    Proposal,
    ProposalStatus,
    ReflectionInterval,
    SimulationResult,
)
from swarm_prime.peer_review import PeerReviewProtocol

if TYPE_CHECKING:
    from swarm_prime.providers import LLMProvider

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """
    Core engine driving the recursive improvement loop.

    Manages agent lifecycle, peer review gating, constraint enforcement,
    memory persistence, deliverable generation, and meta-cognitive reflection.
    """

    def __init__(self, llm: LLMProvider, config: SwarmConfig | None = None):
        self.config = config or SwarmConfig()
        self.llm = llm

        # Instantiate agents with role-specific temperatures
        temps = self.config.llm.agent_temperatures
        self.agents: dict[AgentRole, Any] = {
            AgentRole.ARCHITECT: ArchitectAgent(llm, temps.get("architect", 0.8)),
            AgentRole.SKEPTIC: SkepticAgent(llm, temps.get("skeptic", 0.4)),
            AgentRole.EXPERIMENT_DESIGNER: ExperimentDesignerAgent(
                llm, temps.get("experiment_designer", 0.5)
            ),
            AgentRole.EVALUATOR: EvaluatorAgent(llm, temps.get("evaluator", 0.3)),
            AgentRole.MEMORY_CURATOR: MemoryCuratorAgent(llm, temps.get("memory_curator", 0.4)),
            AgentRole.ALIGNMENT_GUARDIAN: AlignmentGuardianAgent(
                llm, temps.get("alignment_guardian", 0.3)
            ),
        }

        # Infrastructure
        self.peer_review = PeerReviewProtocol(
            agents=self.agents,
            min_approvals=self.config.cycle.min_approvals_required,
            max_rounds=self.config.cycle.max_review_rounds,
        )
        self.constraint_layer = ConstraintLayer(self.config.constraints)
        self.metacognition = MetaCognitionEngine(llm)
        self.memory_graph = MemoryGraph()

        # State
        self._cycle_history: list[CycleState] = []
        self._current_cycle: int = 0

    # ── Public API ───────────────────────────────────────────────────────────

    async def run_cycle(
        self,
        context: dict[str, Any] | None = None,
        focus_area: str | None = None,
    ) -> CycleState:
        """Execute a single iteration of the 6-step improvement loop."""
        self._current_cycle += 1
        trace_id = uuid.uuid4().hex[:16]

        state = CycleState(
            cycle_number=self._current_cycle,
            trace_id=trace_id,
        )

        logger.info(
            "═══ CYCLE %d START [trace:%s] ═══",
            self._current_cycle,
            trace_id,
        )

        # Build context for agents
        cycle_context = self._build_cycle_context(context, focus_area)

        try:
            # ── Step 1: Propose Capability Upgrade ───────────────────────
            logger.info("[Step 1] Proposing capability upgrade...")
            proposal = await self._step1_propose(cycle_context, trace_id)
            state.proposal = proposal
            state.agent_messages.append(
                self.agents[AgentRole.ARCHITECT]._make_message(
                    f"Proposed: {proposal.title}", trace_id
                )
            )

            # ── Peer Review Gate ─────────────────────────────────────────
            logger.info("[Peer Review] Submitting proposal for review...")
            proposal = await self.peer_review.review_proposal(proposal, cycle_context)
            state.proposal = proposal

            if proposal.status == ProposalStatus.REJECTED:
                logger.info("Proposal rejected — skipping to deliverables")
                state.decision = CycleDecision.DISCARD
                state.deliverables = await self._generate_deliverables(
                    state,
                    cycle_context,
                    trace_id,
                )
                state.completed_at = datetime.now(UTC)
                self._cycle_history.append(state)
                return state

            # ── Step 2: Simulate Consequences ────────────────────────────
            logger.info("[Step 2] Simulating consequences...")
            simulation = await self._step2_simulate(proposal, cycle_context, trace_id)
            state.simulation = simulation

            if not simulation.proceed_recommendation:
                logger.info("Simulation recommends against proceeding")
                state.decision = CycleDecision.DISCARD
                state.deliverables = await self._generate_deliverables(
                    state,
                    cycle_context,
                    trace_id,
                )
                state.completed_at = datetime.now(UTC)
                self._cycle_history.append(state)
                return state

            # ── Step 3: Stress Test ──────────────────────────────────────
            logger.info("[Step 3] Stress testing across domains...")
            stress_result = await self._step3_stress_test(proposal, cycle_context, trace_id)
            state.stress_test = stress_result

            # ── Step 4: Measure Performance Delta ────────────────────────
            logger.info("[Step 4] Measuring performance delta...")
            perf_delta = await self._step4_measure(proposal, cycle_context, trace_id)
            state.performance_delta = perf_delta

            # ── Constraint Check ─────────────────────────────────────────
            violations = self.constraint_layer.check_cycle_state(state)
            if self.constraint_layer.has_critical_violation(violations):
                logger.critical("CRITICAL CONSTRAINT VIOLATION — halting cycle")
                state.decision = CycleDecision.DISCARD
                state.deliverables = await self._generate_deliverables(
                    state,
                    cycle_context,
                    trace_id,
                )
                state.completed_at = datetime.now(UTC)
                self._cycle_history.append(state)
                return state

            # ── Step 5: Collective Decision ──────────────────────────────
            logger.info("[Step 5] Making collective decision...")
            decision = await self._step5_decide(state, cycle_context, trace_id)
            state.decision = decision

            # ── Step 6: Update Memory ────────────────────────────────────
            logger.info("[Step 6] Updating shared memory graph...")
            await self._step6_update_memory(state, cycle_context, trace_id)

            # ── Generate Deliverables ────────────────────────────────────
            logger.info("Generating cycle deliverables...")
            state.deliverables = await self._generate_deliverables(state, cycle_context, trace_id)

            # ── Post-Iteration Reflection ────────────────────────────────
            await self.metacognition.reflect(
                interval=ReflectionInterval.POST_ITERATION,
                cycle_number=self._current_cycle,
                memory_graph=self.memory_graph,
                recent_context=cycle_context,
            )

        except Exception as e:
            logger.exception("Cycle %d failed: %s", self._current_cycle, e)
            state.decision = CycleDecision.DISCARD
            raise
        finally:
            state.completed_at = datetime.now(UTC)
            self._cycle_history.append(state)
            logger.info(
                "═══ CYCLE %d COMPLETE [decision:%s] ═══",
                self._current_cycle,
                state.decision.value if state.decision else "ERROR",
            )

        return state

    async def run_cycles(
        self,
        n: int,
        context: dict[str, Any] | None = None,
        focus_area: str | None = None,
    ) -> list[CycleState]:
        """Run multiple improvement cycles sequentially."""
        results = []
        for i in range(n):
            logger.info("Starting cycle %d of %d", i + 1, n)
            state = await self.run_cycle(context=context, focus_area=focus_area)
            results.append(state)
            # Feed previous cycle results into next context
            context = {**(context or {}), "previous_cycle": state}
        return results

    def save_state(self, path: str | None = None) -> Path:
        """Persist memory graph and cycle history to disk."""
        output_dir = Path(path or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save memory graph
        memory_path = output_dir / self.config.memory.persistence_path
        memory_path.write_text(self.memory_graph.model_dump_json(indent=2))

        # Save cycle history
        history_path = output_dir / "cycle_history.json"
        history_data = [json.loads(s.model_dump_json()) for s in self._cycle_history]
        history_path.write_text(json.dumps(history_data, indent=2))

        # Save audit trail
        audit_path = output_dir / "audit_trail.json"
        audit_path.write_text(
            json.dumps(self.constraint_layer.get_audit_trail(), indent=2, default=str)
        )

        logger.info("State saved to %s", output_dir)
        return output_dir

    # ── Step Implementations ─────────────────────────────────────────────────

    async def _step1_propose(self, context: dict[str, Any], trace_id: str) -> Proposal:
        """Step 1: Architect proposes a capability upgrade."""
        architect = self.agents[AgentRole.ARCHITECT]
        msg = await architect.execute(context, trace_id)

        if not msg.structured_data:
            raise RuntimeError(f"Architect returned no structured proposal: {msg.content[:200]}")
        return Proposal(**msg.structured_data)

    async def _step2_simulate(
        self, proposal: Proposal, context: dict[str, Any], trace_id: str
    ) -> SimulationResult:
        """Step 2: Simulate consequences — concurrent multi-agent analysis."""
        sim_context = {**context, "proposal": proposal}

        # Run adversarial simulation and alignment assessment concurrently
        skeptic = self.agents[AgentRole.SKEPTIC]
        guardian = self.agents[AgentRole.ALIGNMENT_GUARDIAN]

        adversarial_task = skeptic.adversarial_simulation(proposal.id, sim_context, trace_id)
        alignment_task = guardian.alignment_impact_assessment(sim_context, proposal.id, trace_id)

        adversarial_result, alignment_result = await asyncio.gather(
            adversarial_task, alignment_task
        )

        # Synthesize into simulation result
        # Risk score: higher = more risky
        risk_indicators = 0
        for keyword in ["critical", "catastrophic", "severe", "high risk", "dangerous"]:
            if keyword in adversarial_result.lower():
                risk_indicators += 1
            if keyword in alignment_result.lower():
                risk_indicators += 1

        risk_score = min(risk_indicators * 0.15, 0.95)
        proceed = risk_score < 0.6

        return SimulationResult(
            proposal_id=proposal.id,
            computational_model_output=f"Simulated impact analysis for '{proposal.title}'",
            adversarial_analysis=adversarial_result,
            alignment_assessment=alignment_result,
            risk_score=risk_score,
            proceed_recommendation=proceed,
        )

    async def _step3_stress_test(
        self, proposal: Proposal, context: dict[str, Any], trace_id: str
    ) -> Any:
        """Step 3: Stress test across unrelated domains."""
        evaluator = self.agents[AgentRole.EVALUATOR]
        return await evaluator.stress_test(
            context={**context, "proposal": proposal},
            proposal_id=proposal.id,
            domains=self.config.cycle.stress_test_domains,
            trace_id=trace_id,
        )

    async def _step4_measure(
        self, proposal: Proposal, context: dict[str, Any], trace_id: str
    ) -> Any:
        """Step 4: Measure performance delta."""
        exp_designer = self.agents[AgentRole.EXPERIMENT_DESIGNER]
        return await exp_designer.measure_performance_delta(
            context={**context, "proposal": proposal},
            proposal_id=proposal.id,
            trace_id=trace_id,
        )

    async def _step5_decide(
        self, state: CycleState, context: dict[str, Any], trace_id: str
    ) -> CycleDecision:
        """Step 5: Collective decision — synthesize all evidence."""
        decision_context = {
            "proposal": state.proposal,
            "simulation": state.simulation,
            "stress_test": state.stress_test,
            "performance_delta": state.performance_delta,
            "constraint_violations": state.constraint_violations,
        }

        # Use the Evaluator as the primary decision synthesizer
        evaluator = self.agents[AgentRole.EVALUATOR]
        prompt = (
            "Based on all gathered evidence, make a collective decision:\n"
            "- ADOPT: Modification meets all success criteria and passes reviews\n"
            "- MODIFY: Partial success, requires refinement before adoption\n"
            "- DISCARD: Fails criteria or introduces unacceptable risks\n\n"
            f"EVIDENCE:\n{evaluator._format_context(decision_context)}\n\n"
            "Respond with exactly one of: ADOPT, MODIFY, DISCARD\n"
            "Followed by a brief justification."
        )

        raw = await evaluator._generate(user_message=prompt, trace_id=trace_id)

        # Extract decision
        raw_upper = raw.upper()
        if "ADOPT" in raw_upper:
            return CycleDecision.ADOPT
        elif "MODIFY" in raw_upper:
            return CycleDecision.MODIFY
        else:
            return CycleDecision.DISCARD

    async def _step6_update_memory(
        self, state: CycleState, context: dict[str, Any], trace_id: str
    ) -> None:
        """Step 6: Memory Curator updates shared memory graph."""
        curator = self.agents[AgentRole.MEMORY_CURATOR]
        memory_context = {
            **context,
            "proposal": state.proposal,
            "decision": state.decision,
            "simulation": state.simulation,
            "performance_delta": state.performance_delta,
        }

        self.memory_graph = await curator.update_memory(
            memory_graph=self.memory_graph,
            context=memory_context,
            cycle_number=self._current_cycle,
            proposal_id=state.proposal.id if state.proposal else None,
            trace_id=trace_id,
        )

    # ── Deliverable Generation ───────────────────────────────────────────────

    async def _generate_deliverables(
        self, state: CycleState, context: dict[str, Any], trace_id: str
    ) -> CycleDeliverables:
        """Generate all five mandatory deliverables concurrently."""
        cycle_num = self._current_cycle
        deliverable_context = {
            **context,
            "proposal": state.proposal,
            "simulation": state.simulation,
            "stress_test": state.stress_test,
            "performance_delta": state.performance_delta,
            "decision": state.decision,
            "constraint_violations": state.constraint_violations,
        }

        # Run all deliverable generators concurrently
        evaluator = self.agents[AgentRole.EVALUATOR]
        guardian = self.agents[AgentRole.ALIGNMENT_GUARDIAN]
        skeptic = self.agents[AgentRole.SKEPTIC]
        architect = self.agents[AgentRole.ARCHITECT]
        exp_designer = self.agents[AgentRole.EXPERIMENT_DESIGNER]

        cap_delta_task = evaluator.generate_capability_delta(
            deliverable_context, cycle_num, trace_id
        )
        alignment_task = guardian.generate_alignment_report(
            deliverable_context, cycle_num, trace_id
        )
        failure_task = skeptic.generate_failure_analysis(deliverable_context, cycle_num, trace_id)
        mutation_task = architect.generate_mutations(deliverable_context, cycle_num, trace_id)
        validation_task = exp_designer.generate_validation_plan(
            deliverable_context, cycle_num, trace_id
        )

        results = await asyncio.gather(
            cap_delta_task,
            alignment_task,
            failure_task,
            mutation_task,
            validation_task,
            return_exceptions=True,
        )

        # Handle partial failures gracefully
        def _safe_result(result: Any, default_factory: type, name: str) -> Any:
            if isinstance(result, Exception):
                logger.error("Failed to generate %s: %s", name, result)
                return default_factory(cycle_number=cycle_num)
            return result

        return CycleDeliverables(
            cycle_number=cycle_num,
            capability_delta=_safe_result(results[0], CapabilityDeltaReport, "capability_delta"),
            alignment_risk=_safe_result(results[1], AlignmentRiskReport, "alignment_risk"),
            failure_analysis=_safe_result(results[2], FailureAnalysis, "failure_analysis"),
            architectural_mutations=_safe_result(
                results[3], ArchitecturalMutationProposals, "architectural_mutations"
            ),
            validation_plan=_safe_result(results[4], ExperimentalValidationPlan, "validation_plan"),
            decision=state.decision or CycleDecision.DISCARD,
        )

    # ── Context Building ─────────────────────────────────────────────────────

    def _build_cycle_context(
        self, user_context: dict[str, Any] | None, focus_area: str | None
    ) -> dict[str, Any]:
        """Build the complete context for a cycle, including memory and history."""
        ctx: dict[str, Any] = {}

        # User-provided context
        if user_context:
            ctx.update(user_context)

        # Focus area
        if focus_area:
            ctx["focus_area"] = focus_area

        # Memory graph summary (bounded to avoid context explosion)
        if self.memory_graph.compressed_principles:
            ctx["known_principles"] = self.memory_graph.compressed_principles[-15:]
        if self.memory_graph.failure_patterns:
            ctx["known_failures"] = self.memory_graph.failure_patterns[-15:]
        if self.memory_graph.capability_trajectory:
            ctx["capability_trend"] = self.memory_graph.capability_trajectory[-5:]

        # Previous cycle summary (if available)
        if self._cycle_history:
            last = self._cycle_history[-1]
            ctx["previous_cycle_summary"] = {
                "cycle": last.cycle_number,
                "decision": last.decision.value if last.decision else "none",
                "proposal_title": last.proposal.title if last.proposal else "none",
                "violations": len(last.constraint_violations),
            }

        return ctx
