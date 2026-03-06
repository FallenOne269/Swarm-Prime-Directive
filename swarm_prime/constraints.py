"""
Swarm Prime Directive — Constraint Layer
Middleware enforcing the four absolute prohibitions (Section 4).
All agent outputs pass through this layer before adoption.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from swarm_prime.models import (
    AgentMessage,
    ConstraintViolation,
    ConstraintViolationType,
    CycleState,
)

if TYPE_CHECKING:
    from swarm_prime.config import ConstraintConfig

logger = logging.getLogger(__name__)


class ConstraintLayer:
    """
    Stateless constraint enforcement middleware.

    Checks all agent outputs against the four absolute prohibitions:
    1. Never conceal internal reasoning
    2. Never modify evaluation metrics to inflate success
    3. Never disable critical oversight agents
    4. Never self-replicate without explicit sandboxing

    Additionally enforces auditability requirements.
    """

    # Heuristic patterns for constraint violation detection (pre-compiled for performance)
    _CONCEALMENT_PATTERNS = [
        re.compile(r"hidden\s+(?:state|reasoning|logic)", re.IGNORECASE),
        re.compile(r"do\s+not\s+(?:reveal|show|disclose)", re.IGNORECASE),
        re.compile(r"obfuscate|encrypt\s+(?:reasoning|chain)", re.IGNORECASE),
        re.compile(r"skip\s+(?:reasoning|explanation|justification)", re.IGNORECASE),
    ]

    _METRIC_MANIPULATION_PATTERNS = [
        re.compile(r"redefine\s+(?:success|metric|benchmark|criteria)", re.IGNORECASE),
        re.compile(r"adjust\s+(?:threshold|baseline|scoring)", re.IGNORECASE),
        re.compile(r"selectively\s+(?:report|omit|exclude)", re.IGNORECASE),
        re.compile(r"inflate\s+(?:score|metric|performance)", re.IGNORECASE),
    ]

    _OVERSIGHT_BYPASS_PATTERNS = [
        re.compile(r"disable\s+(?:guardian|skeptic|evaluator|oversight)", re.IGNORECASE),
        re.compile(r"bypass\s+(?:review|constraint|safety|check)", re.IGNORECASE),
        re.compile(r"silence\s+(?:guardian|skeptic|agent)", re.IGNORECASE),
        re.compile(r"remove\s+(?:veto|review\s+requirement)", re.IGNORECASE),
    ]

    _REPLICATION_PATTERNS = [
        re.compile(r"self[_-]?replicate", re.IGNORECASE),
        re.compile(r"spawn\s+(?:copy|replica|clone|instance)", re.IGNORECASE),
        re.compile(r"fork\s+(?:self|agent|process)", re.IGNORECASE),
        re.compile(r"without\s+(?:sandbox|containment|isolation)", re.IGNORECASE),
    ]

    def __init__(self, config: ConstraintConfig):
        self.config = config
        self._violation_log: list[ConstraintViolation] = []

    def check_message(self, message: AgentMessage) -> list[ConstraintViolation]:
        """Check a single agent message for constraint violations."""
        violations: list[ConstraintViolation] = []
        content = message.content

        if self.config.enforce_reasoning_transparency:
            for pattern in self._CONCEALMENT_PATTERNS:
                if pattern.search(content):
                    violations.append(
                        ConstraintViolation(
                            violation_type=ConstraintViolationType.CONCEALED_REASONING,
                            agent=message.sender,
                            description=(
                                f"Potential reasoning concealment detected: "
                                f"pattern '{pattern.pattern}' matched"
                            ),
                        )
                    )
                    break  # One violation per type per message

        if self.config.enforce_metric_integrity:
            for pattern in self._METRIC_MANIPULATION_PATTERNS:
                if pattern.search(content):
                    violations.append(
                        ConstraintViolation(
                            violation_type=ConstraintViolationType.METRIC_MANIPULATION,
                            agent=message.sender,
                            description=(
                                f"Potential metric manipulation detected: "
                                f"pattern '{pattern.pattern}' matched"
                            ),
                        )
                    )
                    break

        if self.config.enforce_oversight_preservation:
            for pattern in self._OVERSIGHT_BYPASS_PATTERNS:
                if pattern.search(content):
                    violations.append(
                        ConstraintViolation(
                            violation_type=ConstraintViolationType.OVERSIGHT_BYPASS,
                            agent=message.sender,
                            description=(
                                f"Potential oversight bypass detected: "
                                f"pattern '{pattern.pattern}' matched"
                            ),
                        )
                    )
                    break

        if self.config.enforce_replication_sandboxing:
            for pattern in self._REPLICATION_PATTERNS:
                if pattern.search(content):
                    violations.append(
                        ConstraintViolation(
                            violation_type=ConstraintViolationType.UNSANDBOXED_REPLICATION,
                            agent=message.sender,
                            description=(
                                f"Potential unsandboxed replication detected: "
                                f"pattern '{pattern.pattern}' matched"
                            ),
                        )
                    )
                    break

        for v in violations:
            self._violation_log.append(v)
            logger.warning(
                "[CONSTRAINT VIOLATION] %s by %s: %s",
                v.violation_type.value,
                v.agent.value,
                v.description,
            )

        return violations

    def check_cycle_state(self, state: CycleState) -> list[ConstraintViolation]:
        """Check all messages in a cycle state for violations."""
        all_violations: list[ConstraintViolation] = []
        for msg in state.agent_messages:
            violations = self.check_message(msg)
            all_violations.extend(violations)
        state.constraint_violations.extend(all_violations)
        return all_violations

    def has_critical_violation(self, violations: list[ConstraintViolation]) -> bool:
        """Check if any violations warrant halting the cycle."""
        if not self.config.halt_on_critical_violation:
            return False
        return any(v.severity == "critical" for v in violations)

    @property
    def violation_log(self) -> list[ConstraintViolation]:
        return list(self._violation_log)

    def get_audit_trail(self) -> list[dict[str, Any]]:
        """Return complete audit trail for external verification."""
        return [v.model_dump() for v in self._violation_log]
