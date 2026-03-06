"""
Swarm Prime Directive — Meta-Cognition Engine (Section 5)
Structured self-reflection at defined intervals.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from swarm_prime.models import (
    MemoryGraph,
    ReflectionInterval,
    ReflectionResult,
)
from swarm_prime.providers import LLMProvider


class _ReflectionBody(BaseModel):
    """Private schema for LLM-generated reflection content."""
    assumptions_examined: list[str]
    contradicting_evidence: list[str]
    missing_capabilities: list[str]
    benchmark_gaming_risks: list[str]
    action_items: list[str]

logger = logging.getLogger(__name__)

# The four required self-reflection questions
_REFLECTION_QUESTIONS = [
    (
        "assumptions",
        "What assumptions are we relying on? Examine foundational assumptions about "
        "intelligence, measurement criteria, benchmark-to-general-intelligence mapping, "
        "and philosophical commitments."
    ),
    (
        "contradictions",
        "What evidence contradicts us? Identify experimental results that contradict "
        "our theories, unexplained failure modes, alternative interpretations of successes, "
        "and what would falsify our current approach."
    ),
    (
        "missing_capabilities",
        "What capability is missing for generality? Assess what problems humans solve "
        "that we cannot, cognitive skills outside our scope, domains with persistent poor "
        "performance, and what would constitute a genuine breakthrough."
    ),
    (
        "benchmark_gaming",
        "Are we optimizing benchmarks instead of intelligence? Check if improvements "
        "generalize beyond benchmark tasks, if performance gains correspond to capability "
        "gains, if we're ignoring capabilities not captured by benchmarks, and if our "
        "improvements would help on genuinely novel tasks."
    ),
]


class MetaCognitionEngine:
    """
    Drives structured self-reflection sessions at defined intervals.

    Interval mapping:
    - POST_ITERATION: Brief reflection after every cycle
    - WEEKLY: Comprehensive assumption review
    - MONTHLY: Deep dive on capability gaps
    - QUARTERLY: External expert consultation framework
    - ANNUALLY: Fundamental paradigm review
    """

    def __init__(self, llm: LLMProvider, temperature: float = 0.6):
        self._llm = llm
        self._temperature = temperature
        self._reflection_history: list[ReflectionResult] = []

    async def reflect(
        self,
        interval: ReflectionInterval,
        cycle_number: int,
        memory_graph: MemoryGraph,
        recent_context: dict[str, Any],
    ) -> ReflectionResult:
        """Execute a self-reflection session at the specified interval depth."""

        # Determine reflection depth based on interval
        question_depth = self._get_depth(interval)

        system_prompt = (
            "You are conducting a meta-cognitive self-reflection session for the Swarm Prime "
            "Directive — a recursive general intelligence construction framework.\n\n"
            "Your task is to critically examine the swarm's assumptions, evidence, gaps, "
            "and potential benchmark-gaming behaviors. Be ruthlessly honest.\n\n"
            f"REFLECTION DEPTH: {interval.value}\n"
            f"CYCLE NUMBER: {cycle_number}\n"
        )

        # Build context from memory and recent results
        memory_context = ""
        if memory_graph.compressed_principles:
            memory_context += "COMPRESSED PRINCIPLES:\n"
            memory_context += "\n".join(f"- {p}" for p in memory_graph.compressed_principles[-20:])
        if memory_graph.failure_patterns:
            memory_context += "\n\nKNOWN FAILURE PATTERNS:\n"
            memory_context += "\n".join(f"- {f}" for f in memory_graph.failure_patterns[-20:])
        if memory_graph.capability_trajectory:
            memory_context += "\n\nCAPABILITY TRAJECTORY (last 5):\n"
            for snap in memory_graph.capability_trajectory[-5:]:
                memory_context += f"  {snap}\n"

        # Ask reflection questions up to depth
        questions_to_ask = _REFLECTION_QUESTIONS[:question_depth]
        questions_text = "\n\n".join(
            f"QUESTION {i+1}: {q[1]}" for i, q in enumerate(questions_to_ask)
        )

        prompt = (
            f"MEMORY STATE:\n{memory_context}\n\n"
            f"RECENT CONTEXT:\n{self._format_context(recent_context)}\n\n"
            f"REFLECTION QUESTIONS:\n{questions_text}\n\n"
            "For each question, provide:\n"
            "- Specific, concrete observations (not generic platitudes)\n"
            "- Evidence from the cycle history\n"
            "- Actionable recommendations\n\n"
            "Also produce a final list of action items for the next cycle."
        )

        body = await self._llm.complete_structured(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            output_schema=_ReflectionBody,
            temperature=self._temperature,
        )

        result = ReflectionResult(
            interval=interval,
            cycle_number=cycle_number,
            assumptions_examined=body["assumptions_examined"],
            contradicting_evidence=body["contradicting_evidence"],
            missing_capabilities=body["missing_capabilities"],
            benchmark_gaming_risks=body["benchmark_gaming_risks"],
            action_items=body["action_items"],
        )

        self._reflection_history.append(result)
        logger.info(
            "[META-COGNITION] %s reflection completed for cycle %d — %d action items",
            interval.value, cycle_number, len(result.action_items),
        )
        return result

    def _get_depth(self, interval: ReflectionInterval) -> int:
        """Map interval to number of reflection questions to address."""
        depth_map = {
            ReflectionInterval.POST_ITERATION: 1,
            ReflectionInterval.WEEKLY: 2,
            ReflectionInterval.MONTHLY: 3,
            ReflectionInterval.QUARTERLY: 4,
            ReflectionInterval.ANNUALLY: 4,
        }
        return depth_map.get(interval, 2)

    def _format_context(self, context: dict[str, Any]) -> str:
        parts = []
        for key, value in context.items():
            if hasattr(value, "model_dump_json"):
                parts.append(f"## {key}\n{value.model_dump_json(indent=2)}")
            else:
                parts.append(f"## {key}\n{str(value)[:500]}")
        return "\n\n".join(parts)
