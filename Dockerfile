FROM python:3.11-slim AS base

# Prevent bytecode + enable unbuffered stdout for container logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Build stage: install deps ────────────────────────────────────────────────
FROM base AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY swarm_prime/ ./swarm_prime/

RUN pip install --prefix=/install .

# ── Production stage ─────────────────────────────────────────────────────────
FROM base AS production

# Non-root user
RUN groupadd --gid 1000 swarm && \
    useradd --uid 1000 --gid swarm --create-home swarm

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy source (for CLI entry point resolution)
WORKDIR /app
COPY --from=builder /build/swarm_prime ./swarm_prime
COPY --from=builder /build/pyproject.toml ./

# Output directory with correct permissions
RUN mkdir -p /app/swarm_output && chown -R swarm:swarm /app

USER swarm

# Health check — verify Python + package importable
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from swarm_prime.orchestrator import SwarmOrchestrator; print('ok')"

ENTRYPOINT ["swarm-prime"]
CMD ["run", "--cycles", "1"]
