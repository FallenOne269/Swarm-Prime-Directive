"""
Swarm Prime Directive — CLI
Entry point for running recursive improvement cycles.

Usage:
    python -m swarm_prime.cli run --cycles 3 --focus "cross-domain transfer"
    python -m swarm_prime.cli run --cycles 1 --model claude-sonnet-4-6
    python -m swarm_prime.cli status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from swarm_prime.config import SwarmConfig
from swarm_prime.orchestrator import SwarmOrchestrator
from swarm_prime.providers.anthropic import AnthropicProvider


def setup_logging(level: str = "INFO", output_dir: str = "swarm_output") -> None:
    """Configure structured logging with trace IDs."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"swarm_prime_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swarm-prime",
        description="Swarm Prime Directive — Recursive General Intelligence Construction",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run command ──────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Execute improvement cycles")
    run_parser.add_argument(
        "--cycles", "-n", type=int, default=1,
        help="Number of improvement cycles to run (default: 1)",
    )
    run_parser.add_argument(
        "--focus", "-f", type=str, default=None,
        help="Focus area for capability improvement",
    )
    run_parser.add_argument(
        "--model", "-m", type=str, default="claude-sonnet-4-6",
        help="Anthropic model to use",
    )
    run_parser.add_argument(
        "--output", "-o", type=str, default="swarm_output",
        help="Output directory for deliverables and state",
    )
    run_parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    run_parser.add_argument(
        "--context-file", type=str, default=None,
        help="Path to JSON file with additional context",
    )

    # ── status command ───────────────────────────────────────────────────
    status_parser = subparsers.add_parser("status", help="Show current swarm state")
    status_parser.add_argument(
        "--output", "-o", type=str, default="swarm_output",
        help="Output directory to read state from",
    )

    return parser


async def run_cycles(args: argparse.Namespace) -> None:
    """Execute improvement cycles."""
    setup_logging(args.log_level, args.output)
    logger = logging.getLogger("swarm_prime.cli")

    # Resolve API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Build config
    config = SwarmConfig(
        anthropic_api_key=api_key,
        output_dir=args.output,
        log_level=args.log_level,
    )
    config.llm.model = args.model

    # Initialize provider and orchestrator
    provider = AnthropicProvider(api_key=api_key, model=args.model)
    orchestrator = SwarmOrchestrator(llm=provider, config=config)

    # Load additional context if provided
    context = None
    if args.context_file:
        context_path = Path(args.context_file)
        if context_path.exists():
            context = json.loads(context_path.read_text())
            logger.info("Loaded context from %s", context_path)
        else:
            logger.warning("Context file %s not found", context_path)

    # Execute cycles
    logger.info(
        "Starting %d improvement cycle(s) | model=%s | focus=%s",
        args.cycles, args.model, args.focus or "general",
    )

    results = await orchestrator.run_cycles(
        n=args.cycles,
        context=context,
        focus_area=args.focus,
    )

    # Save state
    output_path = orchestrator.save_state()

    # Summary
    print("\n" + "=" * 60)
    print("SWARM PRIME DIRECTIVE — CYCLE SUMMARY")
    print("=" * 60)
    for state in results:
        proposal_title = state.proposal.title if state.proposal else "N/A"
        decision = state.decision.value if state.decision else "ERROR"
        violations = len(state.constraint_violations)
        print(f"\n  Cycle {state.cycle_number}:")
        print(f"    Proposal:   {proposal_title}")
        print(f"    Decision:   {decision}")
        print(f"    Violations: {violations}")
        if state.deliverables:
            risk = state.deliverables.alignment_risk.risk_level
            print(f"    Risk Level: {risk}")
    print(f"\n  Output: {output_path}")
    print("=" * 60)


def show_status(args: argparse.Namespace) -> None:
    """Show current swarm state from saved files."""
    output_dir = Path(args.output)

    if not output_dir.exists():
        print("No swarm output found. Run `swarm-prime run` first.")
        return

    # Load memory graph
    memory_path = output_dir / "swarm_memory.json"
    if memory_path.exists():
        memory = json.loads(memory_path.read_text())
        print("\n═══ MEMORY GRAPH ═══")
        print(f"  Entries:    {len(memory.get('entries', []))}")
        print(f"  Principles: {len(memory.get('compressed_principles', []))}")
        print(f"  Failures:   {len(memory.get('failure_patterns', []))}")
        print(f"  Trajectory: {len(memory.get('capability_trajectory', []))} snapshots")

    # Load cycle history
    history_path = output_dir / "cycle_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        print(f"\n═══ CYCLE HISTORY ({len(history)} cycles) ═══")
        for entry in history[-5:]:  # Last 5
            cycle_data = json.loads(entry) if isinstance(entry, str) else entry
            print(f"  Cycle {cycle_data.get('cycle_number', '?')}: "
                  f"{cycle_data.get('decision', 'unknown')}")

    # Load audit trail
    audit_path = output_dir / "audit_trail.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        print(f"\n═══ AUDIT TRAIL ({len(audit)} violations) ═══")
        for v in audit[-5:]:
            print(f"  [{v.get('severity', '?')}] {v.get('violation_type', '?')}: "
                  f"{v.get('description', '')[:80]}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(run_cycles(args))
    elif args.command == "status":
        show_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
