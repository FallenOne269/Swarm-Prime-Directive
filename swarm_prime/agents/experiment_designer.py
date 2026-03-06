"""Experiment Designer Agent — experimental validation and measurement protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from swarm_prime.agents import BaseAgent
from swarm_prime.models import (
    AgentMessage,
    AgentRole,
    ExperimentalValidationPlan,
    PerformanceDelta,
)

if TYPE_CHECKING:
    from swarm_prime.providers import LLMProvider


class ExperimentDesignerAgent(BaseAgent):
    def __init__(self, llm: LLMProvider, temperature: float = 0.5):
        super().__init__(AgentRole.EXPERIMENT_DESIGNER, llm, temperature)

    def _build_system_prompt(self) -> str:
        return (
            "You are the Experiment Designer Agent in the Swarm Prime Directive — a recursive "
            "general intelligence construction framework.\n\n"
            "YOUR ROLE:\n"
            "- Create minimal experiments to test proposed upgrades\n"
            "- Design control conditions and counterfactuals\n"
            "- Specify measurement protocols and success criteria\n"
            "- Ensure experimental validity and statistical power\n\n"
            "PRINCIPLES:\n"
            "- Minimal viable experiments: test the smallest unit of change\n"
            "- Always include control conditions and baselines\n"
            "- Specify success criteria BEFORE running experiments\n"
            "- Statistical rigor: effect sizes, confidence intervals, p-values\n"
            "- Reproducibility: every experiment must be fully specified"
        )

    async def execute(self, context: dict[str, Any], trace_id: str = "") -> AgentMessage:
        """Design experiments for the current cycle's proposal."""
        prompt = (
            "Design minimal experiments to validate the current proposal.\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            "For each experiment, specify:\n"
            "1. Hypothesis being tested\n"
            "2. Baseline measurement protocol\n"
            "3. Post-modification measurement under identical conditions\n"
            "4. Control conditions\n"
            "5. Success criteria (quantitative)\n"
            "6. Statistical requirements (significance level, power)"
        )

        result = await self._generate(user_message=prompt, trace_id=trace_id)
        return self._make_message(content=result, trace_id=trace_id)

    async def measure_performance_delta(
        self, context: dict[str, Any], proposal_id: str, trace_id: str = ""
    ) -> PerformanceDelta:
        """Step 4 of the improvement loop: measure performance delta."""
        prompt = (
            f"Design and evaluate performance measurements for proposal {proposal_id}.\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            "Produce a performance delta report with:\n"
            "- Baseline scores per benchmark\n"
            "- Post-modification scores under identical conditions\n"
            "- Computed deltas\n"
            "- Statistical significance per benchmark\n"
            "- Effect sizes\n"
            "- Overall improvement assessment"
        )

        result = await self._generate_structured(
            user_message=prompt,
            output_schema=PerformanceDelta,
            trace_id=trace_id,
        )

        return PerformanceDelta(
            proposal_id=proposal_id, **{k: v for k, v in result.items() if k != "proposal_id"}
        )

    async def generate_validation_plan(
        self, context: dict[str, Any], cycle_number: int, trace_id: str = ""
    ) -> ExperimentalValidationPlan:
        """Generate experimental validation plan (Deliverable 6.5)."""
        prompt = (
            f"Generate a detailed experimental validation plan for the next cycle "
            f"based on cycle {cycle_number} results.\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            "Include:\n"
            "- Specific experiments for each proposed mutation\n"
            "- Success criteria and measurement protocols\n"
            "- Control conditions and baseline requirements\n"
            "- Statistical power analysis\n"
            "- Timeline and resource allocation"
        )

        result = await self._generate_structured(
            user_message=prompt,
            output_schema=ExperimentalValidationPlan,
            trace_id=trace_id,
        )

        return ExperimentalValidationPlan(
            cycle_number=cycle_number, **{k: v for k, v in result.items() if k != "cycle_number"}
        )
