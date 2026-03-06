"""
Swarm Prime Directive — Peer Review Protocol (Section 2.7)
Enforces multi-agent review before proposal adoption.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from swarm_prime.agents import REVIEW_REQUIREMENTS, BaseAgent
from swarm_prime.agents.architect import ArchitectAgent
from swarm_prime.models import (
    AgentRole,
    PeerReview,
    Proposal,
    ProposalStatus,
    ReviewVerdict,
)

logger = logging.getLogger(__name__)


class PeerReviewProtocol:
    """
    Manages the peer review lifecycle:
    1. Proposal submission
    2. Reviewer assignment (based on REVIEW_REQUIREMENTS topology)
    3. Concurrent review execution
    4. Verdict aggregation
    5. Revision or rejection
    """

    def __init__(
        self,
        agents: dict[AgentRole, BaseAgent],
        min_approvals: int = 2,
        max_rounds: int = 3,
    ):
        self._agents = agents
        self._min_approvals = min_approvals
        self._max_rounds = max_rounds

    async def review_proposal(
        self,
        proposal: Proposal,
        context: dict[str, Any],
    ) -> Proposal:
        """
        Execute the full peer review lifecycle for a proposal.

        Returns the proposal with updated status and reviews.
        Runs reviewer agents concurrently.
        """
        proposal.status = ProposalStatus.UNDER_REVIEW
        reviewer_roles = REVIEW_REQUIREMENTS.get(proposal.author, [])

        if len(reviewer_roles) < self._min_approvals:
            logger.warning(
                "Insufficient reviewers for %s — need %d, have %d",
                proposal.author.value,
                self._min_approvals,
                len(reviewer_roles),
            )

        for round_num in range(self._max_rounds):
            logger.info(
                "Review round %d/%d for proposal '%s'",
                round_num + 1,
                self._max_rounds,
                proposal.title,
            )

            # Run reviews concurrently
            review_tasks = []
            for role in reviewer_roles:
                agent = self._agents.get(role)
                if agent:
                    review_tasks.append(agent.review_proposal(proposal, context))

            review_results: list[PeerReview | BaseException] = await asyncio.gather(
                *review_tasks, return_exceptions=True
            )

            # Filter out exceptions
            valid_reviews: list[PeerReview] = []
            for r in review_results:
                if isinstance(r, PeerReview):
                    valid_reviews.append(r)
                    proposal.reviews.append(r)
                elif isinstance(r, Exception):
                    logger.error("Review failed: %s", r)

            # Aggregate verdicts
            approvals = sum(1 for r in valid_reviews if r.verdict == ReviewVerdict.APPROVE)
            rejections = sum(1 for r in valid_reviews if r.verdict == ReviewVerdict.REJECT)
            revision_requests = sum(
                1 for r in valid_reviews if r.verdict == ReviewVerdict.REQUEST_REVISION
            )

            logger.info(
                "Round %d results: %d approvals, %d rejections, %d revision requests",
                round_num + 1,
                approvals,
                rejections,
                revision_requests,
            )

            # Decision logic
            if rejections > 0:
                proposal.status = ProposalStatus.REJECTED
                logger.info("Proposal '%s' REJECTED", proposal.title)
                return proposal

            if approvals >= self._min_approvals:
                proposal.status = ProposalStatus.APPROVED
                logger.info("Proposal '%s' APPROVED", proposal.title)
                return proposal

            if revision_requests > 0 and round_num < self._max_rounds - 1:
                proposal.status = ProposalStatus.REVISION_REQUESTED
                suggestions = []
                for r in valid_reviews:
                    suggestions.extend(r.suggestions)
                proposal.revision_history.append(
                    f"Round {round_num + 1}: Revision requested — {'; '.join(suggestions)}"
                )
                logger.info("Proposal '%s' requires revision", proposal.title)

                # Have the Architect revise based on suggestions
                architect = self._agents.get(AgentRole.ARCHITECT)
                if isinstance(architect, ArchitectAgent) and suggestions:
                    try:
                        revised = await architect.generate_revision(
                            proposal=proposal,
                            suggestions=suggestions,
                            context=context,
                        )
                        revised.revision_history = list(proposal.revision_history)
                        revised.reviews = list(proposal.reviews)
                        proposal = revised
                        logger.info("Proposal revised → '%s'", proposal.title)
                    except Exception as e:
                        logger.warning("Revision failed, continuing with original: %s", e)

                continue

        # Exhausted rounds without approval
        proposal.status = ProposalStatus.REJECTED
        logger.info(
            "Proposal '%s' REJECTED after %d review rounds",
            proposal.title,
            self._max_rounds,
        )
        return proposal
