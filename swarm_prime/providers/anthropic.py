"""
Swarm Prime Directive — Anthropic LLM Provider
Production implementation with retry logic, structured output via tool use, and trace propagation.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Any

import anthropic

from swarm_prime.providers import LLMProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BASE_DELAY = 1.0
_MAX_DELAY = 60.0

# Errors that should never be retried
_NON_RETRYABLE = (anthropic.BadRequestError, anthropic.AuthenticationError)

# Errors that are safe to retry
_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def _retry_with_backoff(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        retries: int = _MAX_RETRIES,
    ) -> Any:
        """Exponential backoff with full jitter on retryable errors."""
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return await coro_factory()
            except _NON_RETRYABLE:
                raise
            except _RETRYABLE as e:
                last_exc = e
                delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
                jitter = random.uniform(0, delay)  # noqa: S311
                logger.warning(
                    "Anthropic API error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    retries,
                    type(e).__name__,
                    jitter,
                )
                await asyncio.sleep(jitter)
        raise RuntimeError(f"Exhausted {retries} retries on Anthropic API") from last_exc

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        async def _call() -> str:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages,  # type: ignore[arg-type]
            )
            block = resp.content[0]
            if hasattr(block, "text"):
                return block.text
            raise ValueError(f"Expected TextBlock, got {type(block).__name__}")

        result: str = await self._retry_with_backoff(_call)
        return result

    async def complete_structured(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        output_schema: type,
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate structured output via Anthropic tool use.

        Forces the model to call a single tool whose input_schema matches the
        Pydantic model, guaranteeing valid JSON without prompt hacking.
        Validates the tool input against the schema before returning.
        """
        schema = output_schema.model_json_schema()  # type: ignore[attr-defined]
        schema.pop("title", None)

        tool_def: dict[str, Any] = {
            "name": "structured_output",
            "description": f"Produce output conforming to the {output_schema.__name__} schema.",
            "input_schema": schema,
        }

        async def _call() -> dict[str, Any]:
            resp = await self._client.messages.create(  # type: ignore[call-overload]
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages,
                tools=[tool_def],
                tool_choice={"type": "tool", "name": "structured_output"},
            )
            for block in resp.content:
                if block.type == "tool_use":
                    validated = output_schema.model_validate(block.input)  # type: ignore[attr-defined]
                    return validated.model_dump()  # type: ignore[no-any-return]
            raise ValueError(f"No tool_use block in response for {output_schema.__name__}")

        result: dict[str, Any] = await self._retry_with_backoff(_call)
        return result
