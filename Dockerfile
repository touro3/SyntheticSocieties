# ─── Stage 1: builder ─────────────────────────────────────────────────
FROM python:3.14-slim AS builder

WORKDIR /app

# Layer 1: dependency install — cached unless requirements-api.txt changes.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Layer 2: install metadata only, so editable-install can run before the
# rest of the source tree invalidates this layer on every source edit.
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e . --no-deps || true

# Layer 3: full source — only this layer is invalidated by source edits.
COPY . .

# ─── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.14-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Only copy runtime source dirs — leaves tests/, docs/, raw CSVs, etc.
# behind so the production image stays small. Add more dirs here when a new
# runtime-required source path lands.
COPY --from=builder /app/api /app/api
COPY --from=builder /app/agents /app/agents
COPY --from=builder /app/decision /app/decision
COPY --from=builder /app/models /app/models
COPY --from=builder /app/environment /app/environment
COPY --from=builder /app/simulation /app/simulation
COPY --from=builder /app/metrics /app/metrics
COPY --from=builder /app/population /app/population
COPY --from=builder /app/bgf_logging /app/bgf_logging
COPY --from=builder /app/analysis /app/analysis
COPY --from=builder /app/tracker /app/tracker
COPY --from=builder /app/utils /app/utils
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/data /app/data

# Default port. HF Spaces honours the `app_port: 5050` README frontmatter
# and routes traffic accordingly — do not change to 7860 unless you also
# update README.md. Render and similar platforms inject $PORT at runtime
# and that override is respected by the gunicorn invocation below.
ENV PORT=5050
EXPOSE 5050

# Health check — explicit timeout so a hung server doesn't block forever.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health', timeout=5)"

# Start API (production) — workers=2 fits in 512 MB RAM
CMD ["sh", "-c", "gunicorn 'api.app:create_app()' --bind 0.0.0.0:${PORT} --workers 2 --timeout 300"]
