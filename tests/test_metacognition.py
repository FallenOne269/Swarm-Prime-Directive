"""
Tests for MetaCognitionEngine — the structured self-reflection system.
Previously had zero test coverage despite being called after every cycle.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from swarm_prime.metacognition import MetaCognitionEngine
from swarm_prime.models import (
    MemoryGraph,
    ReflectionInterval,
    ReflectionResult,
)
from swarm_prime.providers import LLMProvider


# ── Mock Provider ─────────────────────────────────────────────────────────────


class MockLLMProvider(LLMProvider):
    """Returns configurable JSON responses based on keyword matching."""

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
        # Default: return a valid _ReflectionBody payload
        return json.dumps({
            "assumptions_examined": ["Default assumption"],
            "contradicting_evidence": [],
            "missing_capabilities": [],
            "benchmark_gaming_risks": [],
            "action_items": ["Default action"],
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


# ── Prompt-capturing mock for structural assertions ───────────────────────────


class CapturingMockLLMProvider(MockLLMProvider):
    """Extends MockLLMProvider to record every LLM call for assertion."""

    def __init__(self, responses: dict[str, str] | None = None):
        super().__init__(responses)
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(messages[-1]["content"] if messages else "")
        return await super().complete(system_prompt, messages, temperature, max_tokens, response_format)


# ── Shared fixture data ───────────────────────────────────────────────────────

_REFLECTION_RESPONSE = json.dumps({
    "assumptions_examined": ["Transfer learning is the core bottleneck"],
    "contradicting_evidence": ["ARC benchmark shows no improvement after last cycle"],
    "missing_capabilities": ["Compositional reasoning across modalities"],
    "benchmark_gaming_risks": ["Scores may overfit to held-out test distribution"],
    "action_items": ["Expand evaluation to 3 genuinely novel domains"],
})


# ── Depth Mapping Tests ───────────────────────────────────────────────────────


class TestReflectionDepth:
    """_get_depth maps each ReflectionInterval to the correct question count."""

    def setup_method(self):
        self.engine = MetaCognitionEngine(MockLLMProvider())

    def test_post_iteration_depth_is_one(self):
        assert self.engine._get_depth(ReflectionInterval.POST_ITERATION) == 1

    def test_weekly_depth_is_two(self):
        assert self.engine._get_depth(ReflectionInterval.WEEKLY) == 2

    def test_monthly_depth_is_three(self):
        assert self.engine._get_depth(ReflectionInterval.MONTHLY) == 3

    def test_quarterly_depth_is_four(self):
        assert self.engine._get_depth(ReflectionInterval.QUARTERLY) == 4

    def test_annually_depth_is_four(self):
        assert self.engine._get_depth(ReflectionInterval.ANNUALLY) == 4

    def test_all_intervals_covered(self):
        """Every ReflectionInterval variant returns a value in range [1, 4]."""
        for interval in ReflectionInterval:
            depth = self.engine._get_depth(interval)
            assert 1 <= depth <= 4, f"Unexpected depth {depth} for {interval}"


# ── reflect() Return Value Tests ─────────────────────────────────────────────


class TestReflectReturnValue:
    @pytest.mark.asyncio
    async def test_returns_reflection_result_instance(self):
        engine = MetaCognitionEngine(MockLLMProvider({"reflection questions": _REFLECTION_RESPONSE}))
        result = await engine.reflect(
            interval=ReflectionInterval.POST_ITERATION,
            cycle_number=1,
            memory_graph=MemoryGraph(),
            recent_context={},
        )
        assert isinstance(result, ReflectionResult)

    @pytest.mark.asyncio
    async def test_interval_and_cycle_number_are_preserved(self):
        engine = MetaCognitionEngine(MockLLMProvider({"reflection questions": _REFLECTION_RESPONSE}))
        result = await engine.reflect(
            interval=ReflectionInterval.MONTHLY,
            cycle_number=42,
            memory_graph=MemoryGraph(),
            recent_context={},
        )
        assert result.interval == ReflectionInterval.MONTHLY
        assert result.cycle_number == 42

    @pytest.mark.asyncio
    async def test_llm_fields_are_mapped_correctly(self):
        """All five fields from the LLM response end up in the ReflectionResult."""
        engine = MetaCognitionEngine(MockLLMProvider({"reflection questions": _REFLECTION_RESPONSE}))
        result = await engine.reflect(
            interval=ReflectionInterval.WEEKLY,
            cycle_number=5,
            memory_graph=MemoryGraph(),
            recent_context={},
        )

        assert "Transfer learning is the core bottleneck" in result.assumptions_examined
        assert "ARC benchmark shows no improvement after last cycle" in result.contradicting_evidence
        assert "Compositional reasoning across modalities" in result.missing_capabilities
        assert "Scores may overfit to held-out test distribution" in result.benchmark_gaming_risks
        assert "Expand evaluation to 3 genuinely novel domains" in result.action_items

    @pytest.mark.asyncio
    async def test_result_has_timestamp(self):
        engine = MetaCognitionEngine(MockLLMProvider({"reflection questions": _REFLECTION_RESPONSE}))
        result = await engine.reflect(
            interval=ReflectionInterval.POST_ITERATION,
            cycle_number=1,
            memory_graph=MemoryGraph(),
            recent_context={},
        )
        assert result.timestamp is not None


# ── History Accumulation Tests ────────────────────────────────────────────────


class TestReflectionHistory:
    @pytest.mark.asyncio
    async def test_history_starts_empty(self):
        engine = MetaCognitionEngine(MockLLMProvider())
        assert engine._reflection_history == []

    @pytest.mark.asyncio
    async def test_each_reflect_call_appends_to_history(self):
        engine = MetaCognitionEngine(MockLLMProvider({"reflection questions": _REFLECTION_RESPONSE}))

        await engine.reflect(ReflectionInterval.POST_ITERATION, 1, MemoryGraph(), {})
        assert len(engine._reflection_history) == 1

        await engine.reflect(ReflectionInterval.WEEKLY, 2, MemoryGraph(), {})
        assert len(engine._reflection_history) == 2

    @pytest.mark.asyncio
    async def test_history_preserves_intervals_in_order(self):
        engine = MetaCognitionEngine(MockLLMProvider({"reflection questions": _REFLECTION_RESPONSE}))

        await engine.reflect(ReflectionInterval.POST_ITERATION, 1, MemoryGraph(), {})
        await engine.reflect(ReflectionInterval.WEEKLY, 2, MemoryGraph(), {})
        await engine.reflect(ReflectionInterval.MONTHLY, 3, MemoryGraph(), {})

        assert engine._reflection_history[0].interval == ReflectionInterval.POST_ITERATION
        assert engine._reflection_history[1].interval == ReflectionInterval.WEEKLY
        assert engine._reflection_history[2].interval == ReflectionInterval.MONTHLY


# ── Memory Context Injection Tests ───────────────────────────────────────────


class TestReflectMemoryContext:
    @pytest.mark.asyncio
    async def test_compressed_principles_appear_in_prompt(self):
        llm = CapturingMockLLMProvider({"reflection questions": _REFLECTION_RESPONSE})
        engine = MetaCognitionEngine(llm)

        memory = MemoryGraph()
        memory.compressed_principles.append("Shared embeddings improve cross-domain transfer")

        await engine.reflect(ReflectionInterval.POST_ITERATION, 1, memory, {})

        assert "Shared embeddings improve cross-domain transfer" in llm.user_prompts[0]

    @pytest.mark.asyncio
    async def test_failure_patterns_appear_in_prompt(self):
        llm = CapturingMockLLMProvider({"reflection questions": _REFLECTION_RESPONSE})
        engine = MetaCognitionEngine(llm)

        memory = MemoryGraph()
        memory.failure_patterns.append("Overfitting detected on ARC held-out set")

        await engine.reflect(ReflectionInterval.POST_ITERATION, 1, memory, {})

        assert "Overfitting detected on ARC held-out set" in llm.user_prompts[0]

    @pytest.mark.asyncio
    async def test_empty_memory_does_not_crash(self):
        """An empty MemoryGraph is valid input and produces a result without errors."""
        engine = MetaCognitionEngine(MockLLMProvider({"reflection questions": _REFLECTION_RESPONSE}))
        result = await engine.reflect(
            interval=ReflectionInterval.POST_ITERATION,
            cycle_number=1,
            memory_graph=MemoryGraph(),
            recent_context={},
        )
        assert isinstance(result, ReflectionResult)

    @pytest.mark.asyncio
    async def test_recent_context_appears_in_prompt(self):
        llm = CapturingMockLLMProvider({"reflection questions": _REFLECTION_RESPONSE})
        engine = MetaCognitionEngine(llm)

        await engine.reflect(
            interval=ReflectionInterval.POST_ITERATION,
            cycle_number=1,
            memory_graph=MemoryGraph(),
            recent_context={"focus_area": "cross-domain-transfer-xyz"},
        )

        assert "cross-domain-transfer-xyz" in llm.user_prompts[0]


# ── Question Depth in Prompt Tests ───────────────────────────────────────────


class TestReflectQuestionDepth:
    @pytest.mark.asyncio
    async def test_post_iteration_prompt_contains_only_question_1(self):
        llm = CapturingMockLLMProvider({"reflection questions": _REFLECTION_RESPONSE})
        engine = MetaCognitionEngine(llm)

        await engine.reflect(ReflectionInterval.POST_ITERATION, 1, MemoryGraph(), {})

        prompt = llm.user_prompts[0]
        assert "QUESTION 1" in prompt
        assert "QUESTION 2" not in prompt

    @pytest.mark.asyncio
    async def test_weekly_prompt_contains_questions_1_and_2(self):
        llm = CapturingMockLLMProvider({"reflection questions": _REFLECTION_RESPONSE})
        engine = MetaCognitionEngine(llm)

        await engine.reflect(ReflectionInterval.WEEKLY, 1, MemoryGraph(), {})

        prompt = llm.user_prompts[0]
        assert "QUESTION 1" in prompt
        assert "QUESTION 2" in prompt
        assert "QUESTION 3" not in prompt

    @pytest.mark.asyncio
    async def test_monthly_prompt_contains_questions_1_through_3(self):
        llm = CapturingMockLLMProvider({"reflection questions": _REFLECTION_RESPONSE})
        engine = MetaCognitionEngine(llm)

        await engine.reflect(ReflectionInterval.MONTHLY, 1, MemoryGraph(), {})

        prompt = llm.user_prompts[0]
        assert "QUESTION 1" in prompt
        assert "QUESTION 2" in prompt
        assert "QUESTION 3" in prompt
        assert "QUESTION 4" not in prompt

    @pytest.mark.asyncio
    async def test_quarterly_prompt_contains_all_four_questions(self):
        llm = CapturingMockLLMProvider({"reflection questions": _REFLECTION_RESPONSE})
        engine = MetaCognitionEngine(llm)

        await engine.reflect(ReflectionInterval.QUARTERLY, 1, MemoryGraph(), {})

        prompt = llm.user_prompts[0]
        assert "QUESTION 1" in prompt
        assert "QUESTION 2" in prompt
        assert "QUESTION 3" in prompt
        assert "QUESTION 4" in prompt

    @pytest.mark.asyncio
    async def test_system_prompt_contains_interval_and_cycle_number(self):
        """Interval name and cycle number are injected into the system prompt."""
        llm = CapturingMockLLMProvider({"reflection questions": _REFLECTION_RESPONSE})
        engine = MetaCognitionEngine(llm)

        await engine.reflect(ReflectionInterval.WEEKLY, 7, MemoryGraph(), {})

        system = llm.system_prompts[0]
        assert "weekly" in system.lower()
        assert "7" in system
