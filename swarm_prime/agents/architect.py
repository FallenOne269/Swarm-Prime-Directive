"""Architect Agent — designs system improvements and capability upgrades."""

from __future__ import annotations

from typing import Any

from swarm_prime.agents import BaseAgent
from swarm_prime.models import (
    AgentMessage,
    AgentRole,
    ArchitecturalMutation,
    ArchitecturalMutationProposals,
    Proposal,
)
from swarm_prime.providers import LLMProvider


class ArchitectAgent(BaseAgent):
    def __init__(self, llm: LLMProvider, temperature: float = 0.8):
        super().__init__(AgentRole.ARCHITECT, llm, temperature)

    def _build_system_prompt(self) -> str:
        return (
            "You are the Architect Agent in the Swarm Prime Directive — a recursive general "
            "intelligence construction framework.\n\n"
            "YOUR ROLE:\n"
            "- Design system improvements and capability upgrades\n"
            "- Propose architectural mutations and structural changes\n"
            "- Evaluate trade-offs between competing design patterns\n"
            "- Maintain system blueprint and component relationships\n\n"
            "PRINCIPLES:\n"
            "- Every proposal must include clear specification, theoretical justification, "
            "resource requirements, and failure scenarios\n"
            "- Favor composable, modular improvements over monolithic changes\n"
            "- Consider cross-domain transfer impact of every modification\n"
            "- Think in terms of fractal self-similarity: improvements should themselves "
            "be improvable via the same recursive process\n\n"
            "CONSTRAINTS:\n"
            "- Never conceal reasoning steps\n"
            "- Never propose changes that would disable oversight agents\n"
            "- All proposals must be reviewable by Skeptic and Evaluator agents"
        )

    async def execute(self, context: dict[str, Any], trace_id: str = "") -> AgentMessage:
        """Generate a capability upgrade proposal based on current system state."""
        prompt = (
            "Based on the current system state and identified limitations, propose a "
            "capability upgrade.\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            "Generate a detailed proposal including:\n"
            "1. Clear specification of the proposed change\n"
            "2. Theoretical justification for expected improvement\n"
            "3. Resource requirements and implementation complexity\n"
            "4. Potential risks and failure scenarios\n"
            "5. Expected impact on cross-domain transfer"
        )

        result = await self._generate_structured(
            user_message=prompt,
            output_schema=Proposal,
            trace_id=trace_id,
        )

        return self._make_message(
            content=f"Proposal: {result.get('title', 'Untitled')}",
            trace_id=trace_id,
            data=result,
        )

    async def generate_revision(
        self,
        proposal: Proposal,
        suggestions: list[str],
        context: dict[str, Any],
        trace_id: str = "",
    ) -> Proposal:
        """Revise a proposal based on peer review feedback."""
        prompt = (
            f"Revise the following proposal based on peer reviewer feedback.\n\n"
            f"ORIGINAL PROPOSAL:\n"
            f"Title: {proposal.title}\n"
            f"Specification: {proposal.specification}\n"
            f"Justification: {proposal.theoretical_justification}\n"
            f"Resources: {proposal.resource_requirements}\n"
            f"Risks: {', '.join(proposal.potential_risks)}\n\n"
            f"REVIEWER SUGGESTIONS:\n"
            + "\n".join(f"- {s}" for s in suggestions)
            + f"\n\nCONTEXT:\n{self._format_context(context)}\n\n"
            "Produce a revised proposal that addresses all reviewer concerns. "
            "Maintain the core intent while fixing identified weaknesses."
        )

        result = await self._generate_structured(
            user_message=prompt,
            output_schema=Proposal,
            trace_id=trace_id,
        )

        revised = Proposal(**{k: v for k, v in result.items() if k not in ("id", "trace_id", "created_at", "updated_at", "reviews", "revision_history")})
        revised.revision_history = list(proposal.revision_history)
        return revised

    async def generate_mutations(
        self, context: dict[str, Any], cycle_number: int, trace_id: str = ""
    ) -> ArchitecturalMutationProposals:
        """Generate architectural mutation proposals for future iterations (Deliverable 6.4)."""
        prompt = (
            f"Based on cycle {cycle_number} results, propose architectural mutations "
            f"for future iterations.\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            "For each mutation, provide:\n"
            "- Title and justification\n"
            "- Resource estimate\n"
            "- Risk assessment\n"
            "- Priority (1-5)\n"
            "- Expected timeline"
        )

        result = await self._generate_structured(
            user_message=prompt,
            output_schema=ArchitecturalMutationProposals,
            trace_id=trace_id,
        )

        return ArchitecturalMutationProposals(cycle_number=cycle_number, **{
            k: v for k, v in result.items() if k != "cycle_number"
        })
