# Pupa backend — cloud image (Railway-ready).
#
# Build context is the repo root so we can copy both `backend/` and
# `deploy/cloud-config.yml`. The baked-in config.yml encodes the
# multi-tenant safety posture (shell off, MCP servers off); Railway env
# vars override everything else at runtime (shell env wins in
# `pupa_config.load_pupa_config`).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# uv handles the lockfile install. `--system`-style isn't used; uv manages
# its own venv at /app/backend/.venv.
RUN pip install --no-cache-dir uv==0.4.30

COPY backend/ /app/backend/

# `uv sync --frozen` honours backend/uv.lock for reproducible installs.
WORKDIR /app/backend
RUN uv sync --frozen --no-dev

# Baked-in cloud defaults. Mounted at the conventional path the YAML
# loader reads from. Railway env vars take precedence — see
# `pupa_config.py`.
RUN mkdir -p /root/.pupa-backend
COPY deploy/cloud-config.yml /root/.pupa-backend/config.yml

EXPOSE 8004

# `uv run` activates the lockfile venv before exec'ing python.
CMD ["uv", "run", "python", "-m", "pupa_backend.app"]
