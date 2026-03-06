"""Memory Curator Agent — shared memory graph management and principle compression."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from swarm_prime.agents import BaseAgent
from swarm_prime.models import (
    AgentMessage,
    AgentRole,
    MemoryEntry,
    MemoryGraph,
)
from swarm_prime.providers import LLMProvider


class _MemoryEntryInput(BaseModel):
    content: str
    abstract_principle: str | None = None
    tags: list[str] = []


class _MemoryUpdate(BaseModel):
    """LLM-generated memory update payload."""
    new_entries: list[_MemoryEntryInput]
    new_failure_patterns: list[str]
    new_principles: list[str]
    capability_snapshot: dict[str, float]


class MemoryCuratorAgent(BaseAgent):
    def __init__(self, llm: LLMProvider, temperature: float = 0.4):
        super().__init__(AgentRole.MEMORY_CURATOR, llm, temperature)

    def _build_system_prompt(self) -> str:
        return (
            "You are the Memory Curator Agent in the Swarm Prime Directive — a recursive "
            "general intelligence construction framework.\n\n"
            "YOUR ROLE:\n"
            "- Compress learnings into abstract reusable principles\n"
            "- Maintain shared memory graph and knowledge base\n"
            "- Identify patterns across experimental outcomes\n"
            "- Enable efficient retrieval of relevant past experiences\n\n"
            "PRINCIPLES:\n"
            "- Compression over accumulation: extract the principle, discard the noise\n"
            "- Cross-reference new learnings with existing memory entries\n"
            "- Tag entries with capability domains for efficient retrieval\n"
            "- Identify meta-patterns: patterns of patterns across cycles\n"
            "- Never distort or omit failure information during compression"
        )

    async def execute(self, context: dict[str, Any], trace_id: str = "") -> AgentMessage:
        """Analyze current context for memory-worthy insights."""
        prompt = (
            "Analyze the current cycle results for memory-worthy insights.\n\n"
            f"CONTEXT:\n{self._format_context(context)}\n\n"
            "Identify:\n"
            "1. Abstract principles that should be preserved\n"
            "2. Failure patterns that should be documented\n"
            "3. Connections to existing memory entries\n"
            "4. Capability trajectory updates"
        )

        result = await self._generate(user_message=prompt, trace_id=trace_id)
        return self._make_message(content=result, trace_id=trace_id)

    async def update_memory(
        self,
        memory_graph: MemoryGraph,
        context: dict[str, Any],
        cycle_number: int,
        proposal_id: str | None = None,
        trace_id: str = "",
    ) -> MemoryGraph:
        """Step 6 of improvement loop: update shared memory graph."""
        # Serialize current memory state for context
        existing_principles = memory_graph.compressed_principles[-10:]  # Last 10
        existing_failures = memory_graph.failure_patterns[-10:]

        prompt = (
            f"Update the shared memory graph for cycle {cycle_number}.\n\n"
            f"EXISTING PRINCIPLES (last 10):\n"
            + "\n".join(f"- {p}" for p in existing_principles) +
            f"\n\nEXISTING FAILURE PATTERNS (last 10):\n"
            + "\n".join(f"- {f}" for f in existing_failures) +
            f"\n\nCURRENT CYCLE CONTEXT:\n{self._format_context(context)}\n\n"
            "Produce:\n"
            "1. New memory entries with abstract principles and tags\n"
            "2. Updated failure patterns observed this cycle\n"
            "3. New compressed principles extracted from outcomes\n"
            "4. Capability snapshot: trait_name -> score (0.0-10.0)"
        )

        parsed = await self._generate_structured(
            user_message=prompt,
            output_schema=_MemoryUpdate,
            trace_id=trace_id,
        )

        for entry_data in parsed.get("new_entries", []):
            memory_graph.add_entry(MemoryEntry(
                content=entry_data["content"],
                abstract_principle=entry_data.get("abstract_principle"),
                source_cycle=cycle_number,
                source_proposal_id=proposal_id,
                tags=entry_data.get("tags", []),
            ))

        for pattern in parsed.get("new_failure_patterns", []):
            if isinstance(pattern, str) and pattern not in memory_graph.failure_patterns:
                memory_graph.failure_patterns.append(pattern)

        for principle in parsed.get("new_principles", []):
            if isinstance(principle, str) and principle not in memory_graph.compressed_principles:
                memory_graph.compressed_principles.append(principle)

        cap_snapshot = parsed.get("capability_snapshot", {})
        if cap_snapshot:
            memory_graph.capability_trajectory.append(cap_snapshot)

        return memory_graph
