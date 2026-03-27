"""
Tests for PeerReviewProtocol — rejection, revision, and exhausted-rounds flows.
The existing test suite only covers the approval happy path; these tests cover
all other branches in peer_review.py.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from swarm_prime.agents.architect import ArchitectAgent
from swarm_prime.agents.evaluator import EvaluatorAgent
from swarm_prime.agents.skeptic import SkepticAgent
from swarm_prime.models import (
    AgentRole,
    PeerReview,
    Proposal,
    ProposalStatus,
    ReviewVerdict,
)
from swarm_prime.peer_review import PeerReviewProtocol
from swarm_prime.providers import LLMProvider


# ── Shared Mock ──────────────────────────────────────────────────────────────


class MockLLMProvider(LLMProvider):
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
            "verdict": "approve",
            "reasoning": "Default approval",
            "concerns": [],
            "suggestions": [],
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


def _make_proposal(title: str = "Test Upgrade") -> Proposal:
    return Proposal(
        title=title,
        specification="Specification text",
        theoretical_justification="Justification text",
        resource_requirements="Low",
    )


def _approve(role: AgentRole, reasoning: str = "Looks good") -> PeerReview:
    return PeerReview(reviewer=role, verdict=ReviewVerdict.APPROVE, reasoning=reasoning)


def _reject(role: AgentRole, reasoning: str = "Fatal flaw") -> PeerReview:
    return PeerReview(reviewer=role, verdict=ReviewVerdict.REJECT, reasoning=reasoning)


def _revise(role: AgentRole, suggestions: list[str] | None = None) -> PeerReview:
    return PeerReview(
        reviewer=role,
        verdict=ReviewVerdict.REQUEST_REVISION,
        reasoning="Needs more work",
        suggestions=suggestions or ["Add more detail"],
    )


def _agents_without_architect(llm: MockLLMProvider) -> dict[AgentRole, Any]:
    return {
        AgentRole.SKEPTIC: SkepticAgent(llm),
        AgentRole.EVALUATOR: EvaluatorAgent(llm),
    }


# ── Rejection Flow ────────────────────────────────────────────────────────────


class TestPeerReviewRejection:
    @pytest.mark.asyncio
    async def test_single_reject_marks_proposal_rejected(self):
        """Any REJECT verdict immediately sets status=REJECTED."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        protocol = PeerReviewProtocol(
            agents={AgentRole.SKEPTIC: skeptic, AgentRole.EVALUATOR: evaluator},
            min_approvals=2,
            max_rounds=3,
        )

        with (
            patch.object(skeptic, "review_proposal", new=AsyncMock(return_value=_reject(AgentRole.SKEPTIC))),
            patch.object(evaluator, "review_proposal", new=AsyncMock(return_value=_approve(AgentRole.EVALUATOR))),
        ):
            result = await protocol.review_proposal(_make_proposal(), {})

        assert result.status == ProposalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_rejection_stops_after_first_round(self):
        """After a REJECT in round 1, reviewers are not called again."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        protocol = PeerReviewProtocol(
            agents={AgentRole.SKEPTIC: skeptic, AgentRole.EVALUATOR: evaluator},
            min_approvals=2,
            max_rounds=3,
        )

        with (
            patch.object(skeptic, "review_proposal", new=AsyncMock(return_value=_reject(AgentRole.SKEPTIC))) as mock_skeptic,
            patch.object(evaluator, "review_proposal", new=AsyncMock(return_value=_approve(AgentRole.EVALUATOR))),
        ):
            await protocol.review_proposal(_make_proposal(), {})

        assert mock_skeptic.call_count == 1  # Called only once, not in subsequent rounds

    @pytest.mark.asyncio
    async def test_rejection_from_evaluator_also_rejects(self):
        """Rejection from either reviewer role terminates the proposal."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        protocol = PeerReviewProtocol(
            agents={AgentRole.SKEPTIC: skeptic, AgentRole.EVALUATOR: evaluator},
            min_approvals=2,
        )

        with (
            patch.object(skeptic, "review_proposal", new=AsyncMock(return_value=_approve(AgentRole.SKEPTIC))),
            patch.object(evaluator, "review_proposal", new=AsyncMock(return_value=_reject(AgentRole.EVALUATOR))),
        ):
            result = await protocol.review_proposal(_make_proposal(), {})

        assert result.status == ProposalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_both_reviewers_rejecting_still_rejects_once(self):
        """If both reviewers reject, the proposal is rejected (not double-rejected)."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        protocol = PeerReviewProtocol(
            agents={AgentRole.SKEPTIC: skeptic, AgentRole.EVALUATOR: evaluator},
            min_approvals=2,
        )

        with (
            patch.object(skeptic, "review_proposal", new=AsyncMock(return_value=_reject(AgentRole.SKEPTIC))),
            patch.object(evaluator, "review_proposal", new=AsyncMock(return_value=_reject(AgentRole.EVALUATOR))),
        ):
            result = await protocol.review_proposal(_make_proposal(), {})

        assert result.status == ProposalStatus.REJECTED


# ── Revision Flow ─────────────────────────────────────────────────────────────


class TestPeerReviewRevision:
    @pytest.mark.asyncio
    async def test_revision_request_triggers_architect_revision(self):
        """REQUEST_REVISION causes generate_revision to be called exactly once."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        architect = ArchitectAgent(llm)
        protocol = PeerReviewProtocol(
            agents={
                AgentRole.SKEPTIC: skeptic,
                AgentRole.EVALUATOR: evaluator,
                AgentRole.ARCHITECT: architect,
            },
            min_approvals=2,
            max_rounds=3,
        )

        revised_proposal = _make_proposal(title="Revised Proposal")

        with (
            patch.object(
                skeptic, "review_proposal",
                new=AsyncMock(side_effect=[
                    _revise(AgentRole.SKEPTIC, suggestions=["Add risk analysis"]),
                    _approve(AgentRole.SKEPTIC),
                ]),
            ),
            patch.object(
                evaluator, "review_proposal",
                new=AsyncMock(side_effect=[
                    _approve(AgentRole.EVALUATOR),
                    _approve(AgentRole.EVALUATOR),
                ]),
            ),
            patch.object(architect, "generate_revision", new=AsyncMock(return_value=revised_proposal)) as mock_rev,
        ):
            await protocol.review_proposal(_make_proposal(), {})

        mock_rev.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_revision_request_leads_to_approval_after_revision(self):
        """A single revision request followed by approvals → APPROVED."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        architect = ArchitectAgent(llm)
        protocol = PeerReviewProtocol(
            agents={
                AgentRole.SKEPTIC: skeptic,
                AgentRole.EVALUATOR: evaluator,
                AgentRole.ARCHITECT: architect,
            },
            min_approvals=2,
            max_rounds=3,
        )

        revised_proposal = _make_proposal(title="Revised Proposal")

        with (
            patch.object(
                skeptic, "review_proposal",
                new=AsyncMock(side_effect=[
                    _revise(AgentRole.SKEPTIC, suggestions=["Improve section 2"]),
                    _approve(AgentRole.SKEPTIC),
                ]),
            ),
            patch.object(
                evaluator, "review_proposal",
                new=AsyncMock(side_effect=[
                    _approve(AgentRole.EVALUATOR),
                    _approve(AgentRole.EVALUATOR),
                ]),
            ),
            patch.object(architect, "generate_revision", new=AsyncMock(return_value=revised_proposal)),
        ):
            result = await protocol.review_proposal(_make_proposal(), {})

        assert result.status == ProposalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_revision_history_records_round_and_suggestions(self):
        """Revision history entry is created with round number and reviewer suggestions."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        architect = ArchitectAgent(llm)
        protocol = PeerReviewProtocol(
            agents={
                AgentRole.SKEPTIC: skeptic,
                AgentRole.EVALUATOR: evaluator,
                AgentRole.ARCHITECT: architect,
            },
            min_approvals=2,
            max_rounds=3,
        )

        revised_proposal = _make_proposal(title="Revised")

        with (
            patch.object(
                skeptic, "review_proposal",
                new=AsyncMock(side_effect=[
                    _revise(AgentRole.SKEPTIC, suggestions=["Clarify failure modes"]),
                    _approve(AgentRole.SKEPTIC),
                ]),
            ),
            patch.object(
                evaluator, "review_proposal",
                new=AsyncMock(side_effect=[
                    _approve(AgentRole.EVALUATOR),
                    _approve(AgentRole.EVALUATOR),
                ]),
            ),
            patch.object(architect, "generate_revision", new=AsyncMock(return_value=revised_proposal)),
        ):
            result = await protocol.review_proposal(_make_proposal(), {})

        assert len(result.revision_history) == 1
        history_entry = result.revision_history[0]
        assert "Round 1" in history_entry
        assert "Clarify failure modes" in history_entry

    @pytest.mark.asyncio
    async def test_failed_revision_continues_with_original_proposal(self):
        """If architect.generate_revision raises, the review continues with the original."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        architect = ArchitectAgent(llm)
        protocol = PeerReviewProtocol(
            agents={
                AgentRole.SKEPTIC: skeptic,
                AgentRole.EVALUATOR: evaluator,
                AgentRole.ARCHITECT: architect,
            },
            min_approvals=2,
            max_rounds=3,
        )

        with (
            patch.object(
                skeptic, "review_proposal",
                new=AsyncMock(side_effect=[
                    _revise(AgentRole.SKEPTIC, suggestions=["Fix X"]),
                    _approve(AgentRole.SKEPTIC),
                ]),
            ),
            patch.object(
                evaluator, "review_proposal",
                new=AsyncMock(side_effect=[
                    _approve(AgentRole.EVALUATOR),
                    _approve(AgentRole.EVALUATOR),
                ]),
            ),
            patch.object(
                architect, "generate_revision",
                new=AsyncMock(side_effect=RuntimeError("LLM timeout")),
            ),
        ):
            # Should not raise — failure is caught and review continues
            result = await protocol.review_proposal(_make_proposal(title="Original"), {})

        # Round 2 still ran and approved the original proposal
        assert result.status == ProposalStatus.APPROVED


# ── Exhausted Rounds Flow ─────────────────────────────────────────────────────


class TestPeerReviewExhaustedRounds:
    @pytest.mark.asyncio
    async def test_exhausted_rounds_without_approval_rejects(self):
        """When max_rounds is exhausted without reaching min_approvals, proposal is rejected."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        # No ARCHITECT — revision is silently skipped
        protocol = PeerReviewProtocol(
            agents={AgentRole.SKEPTIC: skeptic, AgentRole.EVALUATOR: evaluator},
            min_approvals=2,
            max_rounds=2,
        )

        # Skeptic perpetually requests revision; evaluator always approves → never 2 approvals
        with (
            patch.object(skeptic, "review_proposal", new=AsyncMock(return_value=_revise(AgentRole.SKEPTIC))),
            patch.object(evaluator, "review_proposal", new=AsyncMock(return_value=_approve(AgentRole.EVALUATOR))),
        ):
            result = await protocol.review_proposal(_make_proposal(), {})

        assert result.status == ProposalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_each_round_is_executed_before_rejection(self):
        """With max_rounds=2, skeptic's review_proposal is called exactly twice."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        protocol = PeerReviewProtocol(
            agents={AgentRole.SKEPTIC: skeptic, AgentRole.EVALUATOR: evaluator},
            min_approvals=2,
            max_rounds=2,
        )

        with (
            patch.object(skeptic, "review_proposal", new=AsyncMock(return_value=_revise(AgentRole.SKEPTIC))) as mock_s,
            patch.object(evaluator, "review_proposal", new=AsyncMock(return_value=_approve(AgentRole.EVALUATOR))),
        ):
            await protocol.review_proposal(_make_proposal(), {})

        assert mock_s.call_count == 2

    @pytest.mark.asyncio
    async def test_single_round_max_with_only_revisions_rejects(self):
        """max_rounds=1 means one attempt; revisions are not possible (no next round)."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        protocol = PeerReviewProtocol(
            agents={AgentRole.SKEPTIC: skeptic, AgentRole.EVALUATOR: evaluator},
            min_approvals=2,
            max_rounds=1,
        )

        with (
            patch.object(skeptic, "review_proposal", new=AsyncMock(return_value=_revise(AgentRole.SKEPTIC))),
            patch.object(evaluator, "review_proposal", new=AsyncMock(return_value=_revise(AgentRole.EVALUATOR))),
        ):
            result = await protocol.review_proposal(_make_proposal(), {})

        assert result.status == ProposalStatus.REJECTED


# ── Reviewer Exception Handling ───────────────────────────────────────────────


class TestPeerReviewExceptions:
    @pytest.mark.asyncio
    async def test_reviewer_exception_is_filtered_not_fatal(self):
        """If a reviewer agent raises, its result is dropped and the review continues."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        protocol = PeerReviewProtocol(
            agents={AgentRole.SKEPTIC: skeptic, AgentRole.EVALUATOR: evaluator},
            min_approvals=2,
            max_rounds=3,
        )

        with (
            patch.object(skeptic, "review_proposal", new=AsyncMock(side_effect=RuntimeError("Skeptic crashed"))),
            patch.object(evaluator, "review_proposal", new=AsyncMock(return_value=_approve(AgentRole.EVALUATOR))),
        ):
            # Should not raise
            result = await protocol.review_proposal(_make_proposal(), {})

        # Only evaluator's review is in the reviews list (skeptic's was dropped)
        assert all(r.reviewer != AgentRole.SKEPTIC for r in result.reviews)

    @pytest.mark.asyncio
    async def test_all_reviewers_failing_exhausts_rounds_and_rejects(self):
        """If every reviewer raises in every round, the proposal is ultimately rejected."""
        llm = MockLLMProvider()
        skeptic = SkepticAgent(llm)
        evaluator = EvaluatorAgent(llm)
        protocol = PeerReviewProtocol(
            agents={AgentRole.SKEPTIC: skeptic, AgentRole.EVALUATOR: evaluator},
            min_approvals=2,
            max_rounds=2,
        )

        with (
            patch.object(skeptic, "review_proposal", new=AsyncMock(side_effect=RuntimeError("down"))),
            patch.object(evaluator, "review_proposal", new=AsyncMock(side_effect=RuntimeError("down"))),
        ):
            result = await protocol.review_proposal(_make_proposal(), {})

        assert result.status == ProposalStatus.REJECTED
        assert result.reviews == []  # No reviews were recorded
