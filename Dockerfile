# ============================================================
# CodeReviewEnv — Dockerfile
# ============================================================
# Multi-stage build:
#   builder  — installs dependencies
#   runtime  — lean production image
#
# Build:
#   docker build -t codereviewenv .
#
# Run API server:
#   docker run --rm -p 7860:7860 codereviewenv
#
# Run with a specific task:
#   curl -X POST http://localhost:7860/reset \
#     -H "Content-Type: application/json" \
#     -d '{"task_id":"medium_flask_api"}'
#
# Interactive shell:
#   docker run --rm -it --entrypoint bash codereviewenv
# ============================================================

# ---- builder stage -------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only runtime dependencies for the API image.
COPY server/requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ---- runtime stage -------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="CodeReviewEnv"
LABEL org.opencontainers.image.description="OpenEnv for PR code review with deterministic grading"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY codereviewenv/ ./codereviewenv/
COPY server/        ./server/
COPY scripts/       ./scripts/
COPY openenv.yaml   ./
COPY pyproject.toml ./
COPY uv.lock        ./
COPY setup.py       ./
COPY README.md      ./
COPY inference.py   ./
COPY app.py         ./

# Install the package itself without re-resolving dependencies already
# installed above. This keeps the image aligned with the runtime deps while
# still preserving validator-facing metadata in pyproject.toml.
RUN pip install --no-cache-dir -e . --no-deps

# Non-root user for security
RUN useradd --create-home --shell /bin/bash codereviewer
USER codereviewer

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
