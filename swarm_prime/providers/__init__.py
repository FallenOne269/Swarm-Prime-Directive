"""
Swarm Prime Directive — LLM Provider Interface
Abstract base for swappable LLM backends.
"""

from __future__ import annotations

import abc
from typing import Any


class LLMProvider(abc.ABC):
    """Abstract LLM provider — all agent cognition routes through this interface."""

    @abc.abstractmethod
    async def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Generate a completion. Returns raw text content."""
        ...

    @abc.abstractmethod
    async def complete_structured(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        output_schema: type,
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate a structured completion that conforms to a Pydantic model schema.
        Returns a dict that can be validated against output_schema."""
        ...
