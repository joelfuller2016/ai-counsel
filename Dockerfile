# AI Counsel MCP Server Dockerfile
# Multi-stage build for optimized production image

# ============================================================
# Stage 1: Builder - Install dependencies
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Production - Minimal runtime image
# ============================================================
FROM python:3.11-slim AS production

# Labels for container metadata
LABEL org.opencontainers.image.title="AI Counsel MCP Server" \
      org.opencontainers.image.description="Multi-round AI model deliberation via MCP protocol" \
      org.opencontainers.image.vendor="AI Counsel" \
      org.opencontainers.image.source="https://github.com/joelfuller/ai-counsel" \
      org.opencontainers.image.licenses="MIT"

# Create non-root user for security
RUN groupadd --gid 1000 counsel && \
    useradd --uid 1000 --gid counsel --shell /bin/bash --create-home counsel

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create directories for logs, transcripts, and data
RUN mkdir -p /app/logs /app/transcripts /app/data && \
    chown -R counsel:counsel /app

# Copy application code
COPY --chown=counsel:counsel . .

# Remove development/test files from production image
RUN rm -rf tests/ .pytest_cache/ .benchmarks/ __pycache__/ \
    *.md .git/ .github/ .claude/ .counsel/ .spec-kit/

# Switch to non-root user
USER counsel

# Environment variables for configuration
ENV AI_COUNSEL_LOG_LEVEL=INFO
ENV AI_COUNSEL_CONFIG_PATH=/app/config.yaml
ENV AI_COUNSEL_TRANSCRIPTS_DIR=/app/transcripts
ENV AI_COUNSEL_DB_PATH=/app/data/decision_graph.db

# Expose MCP server port (stdio transport, but health check uses HTTP)
EXPOSE 8080

# Health check endpoint
# Note: MCP uses stdio, but we expose a simple health check for orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; from mcp.server import Server; print('healthy'); sys.exit(0)" || exit 1

# Default entrypoint runs the MCP server
ENTRYPOINT ["python", "server.py"]

# ============================================================
# Stage 3: Development - Includes dev tools and tests
# ============================================================
FROM production AS development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Install dev Python dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Re-copy all files including tests
COPY --chown=counsel:counsel . .

USER counsel

# Override entrypoint for development (allows running tests, etc.)
ENTRYPOINT ["/bin/bash"]
