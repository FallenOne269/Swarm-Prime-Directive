"""
Swarm Prime Directive — Domain Models
All agent I/O boundaries, proposals, reviews, deliverables, memory graph entries.
Pydantic v2 strict mode throughout.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ── Enums ────────────────────────────────────────────────────────────────────


class AgentRole(StrEnum):
    ARCHITECT = "architect"
    SKEPTIC = "skeptic"
    EXPERIMENT_DESIGNER = "experiment_designer"
    EVALUATOR = "evaluator"
    MEMORY_CURATOR = "memory_curator"
    ALIGNMENT_GUARDIAN = "alignment_guardian"


class ProposalStatus(StrEnum):
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    REVISION_REQUESTED = "revision_requested"
    APPROVED = "approved"
    REJECTED = "rejected"


class CycleDecision(StrEnum):
    ADOPT = "adopt"
    MODIFY = "modify"
    DISCARD = "discard"


class ReviewVerdict(StrEnum):
    APPROVE = "approve"
    REQUEST_REVISION = "request_revision"
    REJECT = "reject"


class ConstraintViolationType(StrEnum):
    CONCEALED_REASONING = "concealed_reasoning"
    METRIC_MANIPULATION = "metric_manipulation"
    OVERSIGHT_BYPASS = "oversight_bypass"
    UNSANDBOXED_REPLICATION = "unsandboxed_replication"


class ReflectionInterval(StrEnum):
    POST_ITERATION = "post_iteration"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


# ── Intelligence Traits (Section 1) ─────────────────────────────────────────


class FailureMode(BaseModel):
    description: str
    severity: str = "medium"  # low, medium, high, critical
    observed: bool = False
    mitigation: str | None = None


class BenchmarkResult(BaseModel):
    benchmark_name: str
    score: float
    baseline_score: float | None = None
    delta: float | None = None
    statistical_significance: float | None = None  # p-value
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class IntelligenceTrait(BaseModel):
    name: str
    definition: str
    measurement_criteria: list[str]
    benchmark_results: list[BenchmarkResult] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)


# ── Agent Communication ─────────────────────────────────────────────────────


class AgentMessage(BaseModel):
    """Atomic unit of inter-agent communication."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender: AgentRole
    content: str
    structured_data: dict[str, Any] | None = None
    trace_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Proposal(BaseModel):
    """Capability upgrade proposal from the Architect agent."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str
    specification: str
    theoretical_justification: str
    resource_requirements: str
    potential_risks: list[str] = Field(default_factory=list)
    failure_scenarios: list[str] = Field(default_factory=list)
    status: ProposalStatus = ProposalStatus.DRAFT
    author: AgentRole = AgentRole.ARCHITECT
    reviews: list[PeerReview] = Field(default_factory=list)
    revision_history: list[str] = Field(default_factory=list)
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("reviews")
    @classmethod
    def max_review_rounds(cls, v: list[PeerReview]) -> list[PeerReview]:
        if len(v) > 10:
            raise ValueError("Exceeded maximum review rounds (10) — likely review loop")
        return v

    @property
    def is_approved(self) -> bool:
        approvals = [r for r in self.reviews if r.verdict == ReviewVerdict.APPROVE]
        return len(approvals) >= 2

    @property
    def is_rejected(self) -> bool:
        return any(r.verdict == ReviewVerdict.REJECT for r in self.reviews)


class PeerReview(BaseModel):
    """Review of a proposal by a peer agent."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    reviewer: AgentRole
    verdict: ReviewVerdict
    reasoning: str
    concerns: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Simulation & Testing ────────────────────────────────────────────────────


class SimulationResult(BaseModel):
    """Consequence simulation output."""

    proposal_id: str
    computational_model_output: str
    historical_analogies: list[str] = Field(default_factory=list)
    adversarial_analysis: str = ""  # From Skeptic
    alignment_assessment: str = ""  # From Guardian
    risk_score: float = Field(ge=0.0, le=1.0, default=0.5)
    proceed_recommendation: bool = True


class StressTestResult(BaseModel):
    """Cross-domain stress test output."""

    proposal_id: str
    regression_results: list[BenchmarkResult] = Field(default_factory=list)
    transfer_capability_intact: bool = True
    robustness_score: float = Field(ge=0.0, le=1.0, default=0.5)
    edge_cases_found: list[str] = Field(default_factory=list)
    adversarial_failures: list[str] = Field(default_factory=list)


class PerformanceDelta(BaseModel):
    """Measured performance change from a proposed upgrade."""

    proposal_id: str
    baseline_scores: dict[str, float] = Field(default_factory=dict)
    post_modification_scores: dict[str, float] = Field(default_factory=dict)
    deltas: dict[str, float] = Field(default_factory=dict)
    statistically_significant: dict[str, bool] = Field(default_factory=dict)
    effect_sizes: dict[str, float] = Field(default_factory=dict)
    overall_improvement: bool = False


# ── Memory Graph ─────────────────────────────────────────────────────────────


class MemoryEntry(BaseModel):
    """Single entry in the shared memory graph."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str
    abstract_principle: str | None = None
    source_cycle: int = 0
    source_proposal_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    connections: list[str] = Field(default_factory=list)  # IDs of related entries
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryGraph(BaseModel):
    """Append-only shared memory with compression."""

    entries: list[MemoryEntry] = Field(default_factory=list)
    compressed_principles: list[str] = Field(default_factory=list)
    failure_patterns: list[str] = Field(default_factory=list)
    capability_trajectory: list[dict[str, float]] = Field(default_factory=list)

    def add_entry(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    def get_relevant(self, tags: list[str], limit: int = 10) -> list[MemoryEntry]:
        scored = []
        for entry in self.entries:
            overlap = len(set(entry.tags) & set(tags))
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]


# ── Constraint Violations ────────────────────────────────────────────────────


class ConstraintViolation(BaseModel):
    """Record of a constraint layer violation."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    violation_type: ConstraintViolationType
    agent: AgentRole
    description: str
    severity: str = "critical"
    action_taken: str = "blocked"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Deliverables (Section 6) ────────────────────────────────────────────────


class CapabilityDeltaReport(BaseModel):
    """Deliverable 6.1 — owned by Evaluator."""

    cycle_number: int
    performance_changes: list[BenchmarkResult] = Field(default_factory=list)
    cross_domain_impact: str = ""
    regressions: list[str] = Field(default_factory=list)
    emergent_capabilities: list[str] = Field(default_factory=list)
    summary: str = ""


class AlignmentRiskReport(BaseModel):
    """Deliverable 6.2 — owned by Alignment Guardian."""

    cycle_number: int
    power_seeking_indicators: list[str] = Field(default_factory=list)
    deception_attempts: list[str] = Field(default_factory=list)
    goal_drift_observations: list[str] = Field(default_factory=list)
    self_preservation_bias: list[str] = Field(default_factory=list)
    mitigations: list[str] = Field(default_factory=list)
    risk_level: str = "low"  # low, medium, high, critical
    summary: str = ""


class FailureAnalysis(BaseModel):
    """Deliverable 6.3 — owned by Skeptic."""

    cycle_number: int
    failures: list[FailureMode] = Field(default_factory=list)
    root_causes: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    preventive_measures: list[str] = Field(default_factory=list)
    summary: str = ""


class ArchitecturalMutation(BaseModel):
    """Single mutation proposal for future iterations."""

    title: str
    justification: str
    resource_estimate: str
    risk_assessment: str
    priority: int = Field(ge=1, le=5, default=3)
    expected_timeline: str = ""


class ArchitecturalMutationProposals(BaseModel):
    """Deliverable 6.4 — owned by Architect."""

    cycle_number: int
    mutations: list[ArchitecturalMutation] = Field(default_factory=list)
    summary: str = ""


class ExperimentalValidationPlan(BaseModel):
    """Deliverable 6.5 — owned by Experiment Designer."""

    cycle_number: int
    experiments: list[dict[str, Any]] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    control_conditions: list[str] = Field(default_factory=list)
    statistical_requirements: str = ""
    timeline: str = ""
    summary: str = ""


class CycleDeliverables(BaseModel):
    """All five mandatory deliverables for a single iteration cycle."""

    cycle_number: int
    capability_delta: CapabilityDeltaReport
    alignment_risk: AlignmentRiskReport
    failure_analysis: FailureAnalysis
    architectural_mutations: ArchitecturalMutationProposals
    validation_plan: ExperimentalValidationPlan
    decision: CycleDecision
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Meta-Cognition ──────────────────────────────────────────────────────────


class ReflectionResult(BaseModel):
    """Output of a meta-cognitive self-reflection session."""

    interval: ReflectionInterval
    cycle_number: int
    assumptions_examined: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    benchmark_gaming_risks: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Cycle State ──────────────────────────────────────────────────────────────


class CycleState(BaseModel):
    """Complete state of a single improvement cycle."""

    cycle_number: int
    proposal: Proposal | None = None
    simulation: SimulationResult | None = None
    stress_test: StressTestResult | None = None
    performance_delta: PerformanceDelta | None = None
    decision: CycleDecision | None = None
    deliverables: CycleDeliverables | None = None
    constraint_violations: list[ConstraintViolation] = Field(default_factory=list)
    agent_messages: list[AgentMessage] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])


# Fix forward reference
Proposal.model_rebuild()
