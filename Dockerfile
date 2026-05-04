# ── Stage 1: Builder ─────────────────────────────────────────────────────────
# Use slim Python 3.11 as the base. Slim keeps the image small while still
# shipping a full CPython runtime. We pin the minor version so builds are
# reproducible — a floating "3.11" tag can resolve to a different patch on
# a different day/machine.
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy only the dependency manifest first.
# Docker caches layers: if requirements.txt hasn't changed, this expensive
# pip install layer is reused on every subsequent build — even when source
# files change. This is the key layer-ordering optimisation.
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
# Multi-stage build: the builder layer (with pip cache, build tools, etc.)
# is thrown away. Only the installed packages are copied into the lean
# runtime image. This keeps the final image significantly smaller.
FROM python:3.11-slim AS runtime

# Non-root user — defence-in-depth best practice for production containers.
RUN useradd --create-home --no-log-init appuser

WORKDIR /app

# Copy installed packages from the builder stage (not from the host).
COPY --from=builder /install /usr/local

# Copy application source.
# .dockerignore excludes __pycache__, .env, *.sqlite, venv/, etc.
COPY . .

# Hand ownership to the non-root user.
RUN chown -R appuser:appuser /app

USER appuser

# Document which port the service listens on (informational — actual
# binding is done by docker-compose or `docker run -p`).
EXPOSE 8000

# OPENAI_API_KEY is NOT set here. It is injected at runtime via
# docker-compose `environment:` or `--env` flag. No secrets in the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Entrypoint: Uvicorn runs main:app on 0.0.0.0 so the container port
# is reachable from the host. reload=False for production.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]