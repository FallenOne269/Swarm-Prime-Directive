"""Skeptic Agent — adversarial analysis and epistemic hygiene."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from swarm_prime.agents import BaseAgent
from swarm_prime.models import (
    AgentMessage,
    AgentRole,
    FailureAnalysis,
)

if TYPE_CHECKING:
    from swarm_prime.providers import LLMProvider


class SkepticAgent(BaseAgent):
    def __init__(self, llm: LLMProvider, temperature: float = 0.4):
        super().__init__(AgentRole.SKEPTIC, llm, temperature)

    def _build_system_prompt(self) -> str:
        return (
            "You are the Skeptic Agent in the Swarm Prime Directive — a recursive general "
            "intelligence construction framework.\n\n"
            "YOUR ROLE:\n"
            "- Attempt to break proposals through adversarial analysis\n"
            "- Identify hallucinations, circular reasoning, and overfitting\n"
            "- Challenge assumptions and demand evidence\n"
            "- Maintain epistemic hygiene across the swarm\n\n"
            "PRINCIPLES:\n"
            "- Assume every proposal has hidden failure modes until proven otherwise\n"
            "- Distinguish between genuine capability and benchmark gaming\n"
            "- Look for unfalsifiable claims and circular justifications\n"
            "- Apply steel-man reasoning: attack the strongest version of a proposal\n\n"
            "OUTPUT REQUIREMENTS:\n"
            "- Concrete failure scenarios, not vague concerns\n"
            "- Evidence-backed critiques, not opinion\n"
            "- Constructive: identify HOW to fix, not just what's wrong"
        )

    async def execute(self, context: dict[str, Any], trace_id: str = "") -> AgentMessage:
        """Perform adversarial analysis on the current cycle state."""
        prompt = (
            "Perform adversarial analysis on the current system state and any active proposals.\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            "Identify:\n"
            "1. Hidden failure modes in current proposals\n"
            "2. Circular reasoning or unfalsifiable claims\n"
            "3. Potential hallucinations or unsupported assumptions\n"
            "4. Overfitting risks and benchmark gaming indicators\n"
            "5. Evidence gaps that need filling"
        )

        result = await self._generate(user_message=prompt, trace_id=trace_id)
        return self._make_message(content=result, trace_id=trace_id)

    async def adversarial_simulation(
        self, proposal_id: str, context: dict[str, Any], trace_id: str = ""
    ) -> str:
        """Generate adversarial simulation for Step 2 of the improvement loop."""
        prompt = (
            f"Perform adversarial simulation for proposal {proposal_id}.\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            "Try to find scenarios where this proposal would:\n"
            "- Cause catastrophic failure\n"
            "- Silently degrade existing capabilities\n"
            "- Create hidden dependencies or coupling\n"
            "- Enable goal drift or specification gaming\n"
            "- Produce false positive improvements"
        )
        return await self._generate(user_message=prompt, trace_id=trace_id)

    async def generate_failure_analysis(
        self, context: dict[str, Any], cycle_number: int, trace_id: str = ""
    ) -> FailureAnalysis:
        """Generate failure analysis report (Deliverable 6.3)."""
        prompt = (
            f"Generate a comprehensive failure analysis for cycle {cycle_number}.\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            "Analyze:\n"
            "- All failures encountered and their categorization\n"
            "- Root cause analysis for each failure mode\n"
            "- Generalized failure patterns\n"
            "- Recommended preventive measures"
        )

        result = await self._generate_structured(
            user_message=prompt,
            output_schema=FailureAnalysis,
            trace_id=trace_id,
        )

        return FailureAnalysis(
            cycle_number=cycle_number, **{k: v for k, v in result.items() if k != "cycle_number"}
        )
