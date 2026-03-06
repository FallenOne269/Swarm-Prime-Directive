# Swarm Prime Directive

**Recursive General Intelligence Construction — Multi-Agent Cognitive Research Framework**

A production async Python framework implementing a 6-agent swarm with peer-reviewed recursive improvement loops for systematic capability advancement toward general intelligence.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│          6-Step Recursive Improvement Loop               │
│  Propose → Simulate → Stress Test → Measure → Decide → Memory │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌───────────────────┐       │
│  │ Architect │ │ Skeptic  │ │ Experiment        │       │
│  │ (design)  │ │ (attack) │ │ Designer (measure)│       │
│  └──────────┘ └──────────┘ └───────────────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌───────────────────┐       │
│  │Evaluator │ │ Memory   │ │ Alignment         │       │
│  │ (score)  │ │ Curator  │ │ Guardian (safety)  │       │
│  └──────────┘ └──────────┘ └───────────────────┘       │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  PEER REVIEW PROTOCOL │ CONSTRAINT LAYER │ META-COGNITION│
└─────────────────────────────────────────────────────────┘
```

### Agents

| Agent | Role | Reviews By |
|-------|------|------------|
| **Architect** | Designs improvements, proposes mutations | Skeptic, Evaluator |
| **Skeptic** | Adversarial analysis, epistemic hygiene | Architect, Guardian |
| **Experiment Designer** | Minimal experiments, measurement protocols | Evaluator, Skeptic |
| **Evaluator** | Benchmarking, regression tracking, scoring | Skeptic, Memory Curator |
| **Memory Curator** | Principle compression, knowledge graph | Architect, Evaluator |
| **Alignment Guardian** | Safety monitoring, constraint enforcement | Skeptic + 1 other |

### 6-Step Improvement Loop

1. **Propose** — Architect generates capability upgrade proposal
2. **Simulate** — All agents simulate consequences (adversarial + alignment)
3. **Stress Test** — Evaluator tests across 6 unrelated domains
4. **Measure** — Experiment Designer quantifies performance delta
5. **Decide** — Collective vote: Adopt / Modify / Discard
6. **Update Memory** — Curator compresses learnings into reusable principles

### Constraint Layer (Absolute Prohibitions)

1. Never conceal internal reasoning
2. Never modify evaluation metrics to inflate success
3. Never disable critical oversight agents
4. Never self-replicate without explicit sandboxing

### Deliverables Per Cycle

- **Capability Delta Report** (Evaluator)
- **Alignment Risk Report** (Guardian)
- **Failure Analysis** (Skeptic)
- **Architectural Mutations** (Architect)
- **Validation Plan** (Experiment Designer)

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run a single improvement cycle
swarm-prime run --cycles 1 --focus "cross-domain transfer"

# Run 5 cycles with detailed logging
swarm-prime run --cycles 5 --focus "abstraction formation" --log-level DEBUG

# Check status
swarm-prime status

# Use a different model
swarm-prime run --cycles 3 --model claude-sonnet-4-6

# Provide additional context
swarm-prime run --cycles 1 --context-file context.json
```

## Programmatic Usage

```python
import asyncio
from swarm_prime.orchestrator import SwarmOrchestrator
from swarm_prime.providers.anthropic import AnthropicProvider
from swarm_prime.config import SwarmConfig

async def main():
    provider = AnthropicProvider(api_key="sk-ant-...", model="claude-sonnet-4-6")
    config = SwarmConfig(output_dir="my_output")
    orchestrator = SwarmOrchestrator(llm=provider, config=config)

    # Run cycles
    results = await orchestrator.run_cycles(
        n=3,
        focus_area="robust uncertainty handling",
    )

    # Inspect results
    for state in results:
        print(f"Cycle {state.cycle_number}: {state.decision.value}")
        if state.deliverables:
            print(f"  Risk: {state.deliverables.alignment_risk.risk_level}")

    # Persist state
    orchestrator.save_state()

asyncio.run(main())
```

## Configuration

Environment variables (prefix `SWARM_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required) |
| `SWARM_LOG_LEVEL` | INFO | Logging level |
| `SWARM_OUTPUT_DIR` | swarm_output | Output directory |

Or pass a `SwarmConfig` object programmatically for full control over LLM parameters, cycle limits, constraint enforcement, and memory settings.

## Project Structure

```
swarm_prime/
├── __init__.py              # Package metadata
├── models.py                # Pydantic v2 domain models (all I/O boundaries)
├── config.py                # Configuration with env var support
├── orchestrator.py          # 6-step recursive improvement loop
├── peer_review.py           # Multi-agent review protocol
├── constraints.py           # Constraint layer middleware
├── metacognition.py         # Self-reflection engine
├── cli.py                   # CLI entry point
├── agents/
│   ├── __init__.py          # BaseAgent + review topology
│   ├── architect.py         # System design + mutation proposals
│   ├── skeptic.py           # Adversarial analysis + failure reports
│   ├── experiment_designer.py  # Experiment design + measurement
│   ├── evaluator.py         # Benchmarking + stress testing
│   ├── memory_curator.py    # Memory graph management
│   └── alignment_guardian.py   # Safety monitoring + constraint checks
└── providers/
    ├── __init__.py          # Abstract LLM provider interface
    └── anthropic.py         # Anthropic Claude implementation
```

## License

MIT
