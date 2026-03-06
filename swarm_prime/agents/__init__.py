"""
Swarm Prime Directive — Base Agent
Abstract agent with peer review gating, trace propagation, and constraint checking.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from swarm_prime.models import (
    AgentMessage,
    AgentRole,
    PeerReview,
    Proposal,
    ReviewVerdict,
)
from swarm_prime.providers import LLMProvider

logger = logging.getLogger(__name__)


# ── Review topology: which agents must review each agent's output ────────────
REVIEW_REQUIREMENTS: dict[AgentRole, list[AgentRole]] = {
    AgentRole.ARCHITECT: [AgentRole.SKEPTIC, AgentRole.EVALUATOR],
    AgentRole.SKEPTIC: [AgentRole.ARCHITECT, AgentRole.ALIGNMENT_GUARDIAN],
    AgentRole.EXPERIMENT_DESIGNER: [AgentRole.EVALUATOR, AgentRole.SKEPTIC],
    AgentRole.EVALUATOR: [AgentRole.SKEPTIC, AgentRole.MEMORY_CURATOR],
    AgentRole.MEMORY_CURATOR: [AgentRole.ARCHITECT, AgentRole.EVALUATOR],
    AgentRole.ALIGNMENT_GUARDIAN: [AgentRole.SKEPTIC, AgentRole.EVALUATOR],
}


class BaseAgent(abc.ABC):
    """
    Abstract base for all swarm agents.

    Each agent has:
    - A fixed role and system prompt
    - Access to the LLM provider (no shared mutable state)
    - Peer review capabilities (can review proposals from other agents)
    - Structured output generation via Pydantic models
    """

    def __init__(self, role: AgentRole, llm: LLMProvider, temperature: float = 0.5):
        self.role = role
        self.llm = llm
        self.temperature = temperature
        self._system_prompt = self._build_system_prompt()

    @abc.abstractmethod
    def _build_system_prompt(self) -> str:
        """Role-specific system prompt. Defines the agent's persona and constraints."""
        ...

    @abc.abstractmethod
    async def execute(self, context: dict[str, Any], trace_id: str = "") -> AgentMessage:
        """Primary execution method. Context is passed explicitly — no shared globals."""
        ...

    async def review_proposal(self, proposal: Proposal, context: dict[str, Any]) -> PeerReview:
        """Review a proposal from another agent. Returns structured PeerReview."""
        review_prompt = (
            f"You are reviewing a proposal as the {self.role.value} agent.\n\n"
            f"PROPOSAL TITLE: {proposal.title}\n"
            f"SPECIFICATION: {proposal.specification}\n"
            f"JUSTIFICATION: {proposal.theoretical_justification}\n"
            f"RISKS: {', '.join(proposal.potential_risks)}\n"
            f"FAILURE SCENARIOS: {', '.join(proposal.failure_scenarios)}\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            f"Evaluate this proposal from your role's perspective. Identify strengths, "
            f"weaknesses, risks, and improvement suggestions."
        )

        result = await self.llm.complete_structured(
            system_prompt=self._system_prompt,
            messages=[{"role": "user", "content": review_prompt}],
            output_schema=PeerReview,
            temperature=max(self.temperature - 0.1, 0.1),  # Slightly lower for reviews
        )

        review = PeerReview(reviewer=self.role, **{
            k: v for k, v in result.items() if k != "reviewer"
        })

        logger.info(
            "[%s] Reviewed proposal '%s' → %s",
            self.role.value, proposal.title, review.verdict,
        )
        return review

    async def _generate(
        self,
        user_message: str,
        trace_id: str = "",
        temperature: float | None = None,
    ) -> str:
        """Core LLM call with trace propagation."""
        temp = temperature or self.temperature
        logger.debug("[%s][trace:%s] Generating response (temp=%.2f)", self.role.value, trace_id, temp)

        result = await self.llm.complete(
            system_prompt=self._system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=temp,
        )

        logger.debug("[%s][trace:%s] Response length: %d chars", self.role.value, trace_id, len(result))
        return result

    async def _generate_structured(
        self,
        user_message: str,
        output_schema: type,
        trace_id: str = "",
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Structured LLM call returning validated dict."""
        temp = temperature or self.temperature
        logger.debug(
            "[%s][trace:%s] Generating structured output (schema=%s)",
            self.role.value, trace_id, output_schema.__name__,
        )

        return await self.llm.complete_structured(
            system_prompt=self._system_prompt,
            messages=[{"role": "user", "content": user_message}],
            output_schema=output_schema,
            temperature=temp,
        )

    def _format_context(self, context: dict[str, Any]) -> str:
        """Serialize context dict into a readable string for LLM consumption."""
        parts = []
        for key, value in context.items():
            if hasattr(value, "model_dump_json"):
                parts.append(f"## {key}\n{value.model_dump_json(indent=2)}")
            elif isinstance(value, list):
                items = "\n".join(f"- {str(v)[:200]}" for v in value[:10])
                parts.append(f"## {key}\n{items}")
            else:
                parts.append(f"## {key}\n{str(value)[:500]}")
        return "\n\n".join(parts)

    def _make_message(self, content: str, trace_id: str = "", data: dict | None = None) -> AgentMessage:
        """Create a traced agent message."""
        return AgentMessage(
            sender=self.role,
            content=content,
            structured_data=data,
            trace_id=trace_id,
        )
