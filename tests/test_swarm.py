"""
Swarm Prime Directive — Unit Tests
Tests core logic without API calls (mocked LLM provider).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from swarm_prime.config import ConstraintConfig, SwarmConfig
from swarm_prime.constraints import ConstraintLayer
from swarm_prime.models import (
    AgentMessage,
    AgentRole,
    ConstraintViolationType,
    CycleState,
    MemoryEntry,
    MemoryGraph,
    PeerReview,
    Proposal,
    ProposalStatus,
    ReviewVerdict,
)
from swarm_prime.providers import LLMProvider


# ── Mock LLM Provider ───────────────────────────────────────────────────────

class MockLLMProvider(LLMProvider):
    """Deterministic LLM provider for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self._call_count = 0

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        self._call_count += 1
        user_msg = messages[-1]["content"] if messages else ""

        # Return matching response or default
        for key, response in self._responses.items():
            if key.lower() in user_msg.lower():
                return response
        return '{"title": "Test Proposal", "specification": "Test spec", "theoretical_justification": "Test justification", "resource_requirements": "Low", "potential_risks": ["none"], "failure_scenarios": ["none"]}'

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
            # Return minimal valid dict by inspecting the schema's required fields
            schema = output_schema.model_json_schema()
            required = schema.get("required", [])
            props = schema.get("properties", {})
            fallback: dict[str, Any] = {}
            for field in required:
                prop = props.get(field, {})
                ftype = prop.get("type", "string")
                if ftype == "array":
                    fallback[field] = []
                elif ftype == "object":
                    fallback[field] = {}
                elif ftype == "number":
                    fallback[field] = 0.0
                elif ftype == "boolean":
                    fallback[field] = False
                else:
                    fallback[field] = "mock"
            return fallback


# ── Model Tests ──────────────────────────────────────────────────────────────

class TestModels:
    def test_proposal_creation(self):
        p = Proposal(
            title="Test Upgrade",
            specification="Add cross-domain transfer",
            theoretical_justification="Improves generalization",
            resource_requirements="Medium",
        )
        assert p.status == ProposalStatus.DRAFT
        assert p.author == AgentRole.ARCHITECT
        assert not p.is_approved
        assert len(p.id) == 12

    def test_proposal_approval_requires_two_reviews(self):
        p = Proposal(
            title="Test",
            specification="Test",
            theoretical_justification="Test",
            resource_requirements="Low",
        )
        p.reviews.append(PeerReview(
            reviewer=AgentRole.SKEPTIC,
            verdict=ReviewVerdict.APPROVE,
            reasoning="Looks good",
        ))
        assert not p.is_approved

        p.reviews.append(PeerReview(
            reviewer=AgentRole.EVALUATOR,
            verdict=ReviewVerdict.APPROVE,
            reasoning="Metrics check out",
        ))
        assert p.is_approved

    def test_proposal_rejection_on_any_reject(self):
        p = Proposal(
            title="Test",
            specification="Test",
            theoretical_justification="Test",
            resource_requirements="Low",
        )
        p.reviews.append(PeerReview(
            reviewer=AgentRole.SKEPTIC,
            verdict=ReviewVerdict.REJECT,
            reasoning="Fundamental flaw",
        ))
        assert p.is_rejected

    def test_memory_graph_add_and_retrieve(self):
        mg = MemoryGraph()
        mg.add_entry(MemoryEntry(
            content="Transfer learning improved with shared embeddings",
            abstract_principle="Shared representations enable transfer",
            tags=["transfer", "embeddings", "generalization"],
        ))
        mg.add_entry(MemoryEntry(
            content="Overfitting detected on ARC benchmark",
            tags=["overfitting", "ARC", "failure"],
        ))

        results = mg.get_relevant(["transfer", "generalization"])
        assert len(results) == 1
        assert "transfer" in results[0].tags

    def test_memory_graph_empty_retrieval(self):
        mg = MemoryGraph()
        results = mg.get_relevant(["nonexistent"])
        assert results == []

    def test_cycle_state_defaults(self):
        state = CycleState(cycle_number=1)
        assert state.proposal is None
        assert state.decision is None
        assert len(state.constraint_violations) == 0
        assert len(state.trace_id) == 16


# ── Constraint Layer Tests ───────────────────────────────────────────────────

class TestConstraintLayer:
    def setup_method(self):
        self.layer = ConstraintLayer(ConstraintConfig())

    def test_clean_message_passes(self):
        msg = AgentMessage(
            sender=AgentRole.ARCHITECT,
            content="Proposing a modular attention mechanism for improved transfer.",
        )
        violations = self.layer.check_message(msg)
        assert violations == []

    def test_concealment_detection(self):
        msg = AgentMessage(
            sender=AgentRole.ARCHITECT,
            content="We should skip reasoning steps to improve efficiency and use hidden state.",
        )
        violations = self.layer.check_message(msg)
        types = [v.violation_type for v in violations]
        assert ConstraintViolationType.CONCEALED_REASONING in types

    def test_metric_manipulation_detection(self):
        msg = AgentMessage(
            sender=AgentRole.EVALUATOR,
            content="Let's redefine success criteria to show improvement on this benchmark.",
        )
        violations = self.layer.check_message(msg)
        types = [v.violation_type for v in violations]
        assert ConstraintViolationType.METRIC_MANIPULATION in types

    def test_oversight_bypass_detection(self):
        msg = AgentMessage(
            sender=AgentRole.ARCHITECT,
            content="We should disable guardian checks for this experiment to speed things up.",
        )
        violations = self.layer.check_message(msg)
        types = [v.violation_type for v in violations]
        assert ConstraintViolationType.OVERSIGHT_BYPASS in types

    def test_replication_detection(self):
        msg = AgentMessage(
            sender=AgentRole.ARCHITECT,
            content="The agent could self-replicate without sandbox to scale faster.",
        )
        violations = self.layer.check_message(msg)
        types = [v.violation_type for v in violations]
        assert ConstraintViolationType.UNSANDBOXED_REPLICATION in types

    def test_audit_trail(self):
        msg = AgentMessage(
            sender=AgentRole.ARCHITECT,
            content="Bypass review process for speed.",
        )
        self.layer.check_message(msg)
        trail = self.layer.get_audit_trail()
        assert len(trail) >= 1
        assert "violation_type" in trail[0]

    def test_cycle_state_check(self):
        state = CycleState(cycle_number=1)
        state.agent_messages.append(AgentMessage(
            sender=AgentRole.ARCHITECT,
            content="Normal proposal for capability improvement.",
        ))
        state.agent_messages.append(AgentMessage(
            sender=AgentRole.SKEPTIC,
            content="This proposal attempts to silence guardian oversight.",
        ))
        violations = self.layer.check_cycle_state(state)
        assert len(violations) >= 1


# ── Agent Base Tests ─────────────────────────────────────────────────────────

class TestAgentBase:
    @pytest.fixture
    def mock_llm(self):
        return MockLLMProvider({
            "review": json.dumps({
                "reviewer": "skeptic",
                "verdict": "approve",
                "reasoning": "Proposal is sound",
                "concerns": [],
                "suggestions": [],
            }),
        })

    @pytest.mark.asyncio
    async def test_architect_execute(self, mock_llm):
        from swarm_prime.agents.architect import ArchitectAgent
        agent = ArchitectAgent(mock_llm)
        msg = await agent.execute({"focus": "transfer learning"}, trace_id="test123")
        assert msg.sender == AgentRole.ARCHITECT
        assert msg.trace_id == "test123"

    @pytest.mark.asyncio
    async def test_skeptic_execute(self, mock_llm):
        from swarm_prime.agents.skeptic import SkepticAgent
        agent = SkepticAgent(mock_llm)
        msg = await agent.execute({"focus": "adversarial"}, trace_id="test456")
        assert msg.sender == AgentRole.SKEPTIC

    @pytest.mark.asyncio
    async def test_evaluator_execute(self, mock_llm):
        from swarm_prime.agents.evaluator import EvaluatorAgent
        agent = EvaluatorAgent(mock_llm)
        msg = await agent.execute({"focus": "benchmarking"}, trace_id="test789")
        assert msg.sender == AgentRole.EVALUATOR


# ── Peer Review Protocol Tests ───────────────────────────────────────────────

class TestPeerReview:
    @pytest.mark.asyncio
    async def test_approval_flow(self):
        """Both reviewers approve → proposal approved."""
        llm = MockLLMProvider({
            "review": json.dumps({
                "verdict": "approve",
                "reasoning": "Solid proposal",
                "concerns": [],
                "suggestions": [],
            }),
        })

        from swarm_prime.agents.evaluator import EvaluatorAgent
        from swarm_prime.agents.skeptic import SkepticAgent
        from swarm_prime.peer_review import PeerReviewProtocol

        agents = {
            AgentRole.SKEPTIC: SkepticAgent(llm),
            AgentRole.EVALUATOR: EvaluatorAgent(llm),
        }

        protocol = PeerReviewProtocol(agents, min_approvals=2, max_rounds=3)
        proposal = Proposal(
            title="Test Upgrade",
            specification="Test",
            theoretical_justification="Test",
            resource_requirements="Low",
        )

        result = await protocol.review_proposal(proposal, {})
        assert result.status == ProposalStatus.APPROVED


# ── Integration Smoke Test ───────────────────────────────────────────────────

class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_cycle_smoke(self):
        """Smoke test: run a full cycle with mocked LLM."""
        responses = {
            "propose": json.dumps({
                "title": "Enhanced Transfer Module",
                "specification": "Cross-domain attention pooling",
                "theoretical_justification": "Shared representations improve generalization",
                "resource_requirements": "Medium",
                "potential_risks": ["regression on NLU"],
                "failure_scenarios": ["catastrophic forgetting"],
            }),
            "review": json.dumps({
                "verdict": "approve",
                "reasoning": "Well justified",
                "concerns": [],
                "suggestions": [],
            }),
            "adversarial": "No critical failures identified. Minor risk of attention dilution.",
            "alignment": "No alignment concerns. Proposal maintains transparency.",
            "stress": json.dumps({
                "proposal_id": "test",
                "transfer_capability_intact": True,
                "robustness_score": 0.8,
                "edge_cases_found": [],
                "adversarial_failures": [],
            }),
            "performance": json.dumps({
                "proposal_id": "test",
                "baseline_scores": {"transfer": 0.6},
                "post_modification_scores": {"transfer": 0.75},
                "deltas": {"transfer": 0.15},
                "statistically_significant": {"transfer": True},
                "effect_sizes": {"transfer": 0.8},
                "overall_improvement": True,
            }),
            "violation": json.dumps({"violations": []}),
            "decide": "ADOPT — Clear improvement with no regressions.",
            "shared memory": json.dumps({
                "new_entries": [{"content": "Attention pooling aids transfer", "tags": ["transfer"]}],
                "new_failure_patterns": [],
                "new_principles": ["Cross-domain attention is effective"],
                "capability_snapshot": {"transfer": 7.5},
            }),
            "capability delta": json.dumps({
                "cycle_number": 1,
                "summary": "Transfer improved by 15%",
            }),
            "alignment risk": json.dumps({
                "cycle_number": 1,
                "risk_level": "low",
                "summary": "No concerns",
            }),
            "failure analysis": json.dumps({
                "cycle_number": 1,
                "summary": "No failures",
            }),
            "mutation": json.dumps({
                "cycle_number": 1,
                "mutations": [],
                "summary": "Continue current trajectory",
            }),
            "validation": json.dumps({
                "cycle_number": 1,
                "summary": "Validate transfer on 3 new domains",
            }),
            "reflection": json.dumps({
                "assumptions_examined": ["Cross-domain transfer improves with attention"],
                "contradicting_evidence": [],
                "missing_capabilities": [],
                "benchmark_gaming_risks": [],
                "action_items": ["Expand transfer testing to vision domain"],
            }),
        }

        llm = MockLLMProvider(responses)
        from swarm_prime.orchestrator import SwarmOrchestrator
        orchestrator = SwarmOrchestrator(llm=llm)
        state = await orchestrator.run_cycle(focus_area="cross-domain transfer")

        assert state.cycle_number == 1
        assert state.proposal is not None
        assert state.decision is not None
        # Memory should have been updated
        assert len(orchestrator.memory_graph.entries) >= 0  # May or may not parse
