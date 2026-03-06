"""
Swarm Prime Directive — Configuration
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.7
    # Higher temp for divergent agents (Architect), lower for convergent (Evaluator)
    agent_temperatures: dict[str, float] = Field(
        default_factory=lambda: {
            "architect": 0.8,
            "skeptic": 0.4,
            "experiment_designer": 0.5,
            "evaluator": 0.3,
            "memory_curator": 0.4,
            "alignment_guardian": 0.3,
        }
    )


class CycleConfig(BaseModel):
    max_review_rounds: int = 3
    min_approvals_required: int = 2
    max_cycles: int = 100
    stress_test_domains: list[str] = Field(
        default_factory=lambda: [
            "logical_reasoning",
            "mathematical_problem_solving",
            "natural_language_understanding",
            "code_generation",
            "analogical_reasoning",
            "planning_and_scheduling",
        ]
    )


class ConstraintConfig(BaseModel):
    enforce_reasoning_transparency: bool = True
    enforce_metric_integrity: bool = True
    enforce_oversight_preservation: bool = True
    enforce_replication_sandboxing: bool = True
    halt_on_critical_violation: bool = True


class MemoryConfig(BaseModel):
    max_entries: int = 10000
    compression_threshold: int = 500  # Compress after this many entries
    retrieval_limit: int = 20
    persistence_path: str = "swarm_memory.json"


class SwarmConfig(BaseSettings):
    model_config = {"env_prefix": "SWARM_", "env_nested_delimiter": "__"}

    llm: LLMConfig = Field(default_factory=LLMConfig)
    cycle: CycleConfig = Field(default_factory=CycleConfig)
    constraints: ConstraintConfig = Field(default_factory=ConstraintConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # API keys loaded from env
    anthropic_api_key: str = ""
    log_level: str = "INFO"
    output_dir: str = "swarm_output"
