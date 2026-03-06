"""Evaluator Agent — scoring, benchmarks, and regression tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from swarm_prime.agents import BaseAgent
from swarm_prime.models import (
    AgentMessage,
    AgentRole,
    CapabilityDeltaReport,
    StressTestResult,
)

if TYPE_CHECKING:
    from swarm_prime.providers import LLMProvider


class EvaluatorAgent(BaseAgent):
    def __init__(self, llm: LLMProvider, temperature: float = 0.3):
        super().__init__(AgentRole.EVALUATOR, llm, temperature)

    def _build_system_prompt(self) -> str:
        return (
            "You are the Evaluator Agent in the Swarm Prime Directive — a recursive general "
            "intelligence construction framework.\n\n"
            "YOUR ROLE:\n"
            "- Score the system on generalization, reasoning depth, and transfer\n"
            "- Maintain evaluation benchmarks and metrics\n"
            "- Track performance deltas across iterations\n"
            "- Identify capability regressions\n\n"
            "PRINCIPLES:\n"
            "- Metrics must be defined BEFORE evaluation, never post-hoc\n"
            "- Always compare against established baselines\n"
            "- Flag any metric that shows >2σ deviation from trend\n"
            "- Distinguish genuine capability from benchmark-specific optimization\n"
            "- Maintain integrity of evaluation — never modify metrics to improve scores\n\n"
            "SEVEN INTELLIGENCE TRAITS TO EVALUATE:\n"
            "1. Cross-Domain Transfer Ability\n"
            "2. Abstraction Formation\n"
            "3. Self-Modeling\n"
            "4. Goal Formation and Revision\n"
            "5. Long-Horizon Planning\n"
            "6. Novel Problem Synthesis\n"
            "7. Robust Uncertainty Handling"
        )

    async def execute(self, context: dict[str, Any], trace_id: str = "") -> AgentMessage:
        """Evaluate current system state against intelligence traits."""
        prompt = (
            "Evaluate the current system state against all seven intelligence traits.\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            "For each trait, assess:\n"
            "- Current capability level (0-10)\n"
            "- Evidence supporting the score\n"
            "- Identified regressions from previous cycles\n"
            "- Recommended focus areas for improvement"
        )

        result = await self._generate(user_message=prompt, trace_id=trace_id)
        return self._make_message(content=result, trace_id=trace_id)

    async def stress_test(
        self, context: dict[str, Any], proposal_id: str, domains: list[str], trace_id: str = ""
    ) -> StressTestResult:
        """Step 3 of improvement loop: stress test across unrelated domains."""
        prompt = (
            f"Perform stress testing for proposal {proposal_id} across these domains:\n"
            f"{', '.join(domains)}\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            "Evaluate:\n"
            "1. Regression risk on existing benchmarks per domain\n"
            "2. Cross-domain transfer capability preservation\n"
            "3. Robustness under distribution shift\n"
            "4. Edge cases and adversarial inputs\n"
            "5. Overall robustness score (0.0-1.0)"
        )

        result = await self._generate_structured(
            user_message=prompt,
            output_schema=StressTestResult,
            trace_id=trace_id,
        )

        return StressTestResult(
            proposal_id=proposal_id, **{k: v for k, v in result.items() if k != "proposal_id"}
        )

    async def generate_capability_delta(
        self, context: dict[str, Any], cycle_number: int, trace_id: str = ""
    ) -> CapabilityDeltaReport:
        """Generate capability delta report (Deliverable 6.1)."""
        prompt = (
            f"Generate a capability delta report for cycle {cycle_number}.\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            "Document:\n"
            "- Quantified performance changes on all benchmarks\n"
            "- Statistical significance and effect size\n"
            "- Cross-domain transfer impact\n"
            "- Any capability regressions\n"
            "- Emergent capabilities identified"
        )

        result = await self._generate_structured(
            user_message=prompt,
            output_schema=CapabilityDeltaReport,
            trace_id=trace_id,
        )

        return CapabilityDeltaReport(
            cycle_number=cycle_number, **{k: v for k, v in result.items() if k != "cycle_number"}
        )
