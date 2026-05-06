# ─── Stage 1: builder ─────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# ─── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Default port (overridden by Render via $PORT)
ENV PORT=5050
EXPOSE 5050

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

# Start API (production) — workers=2 fits in 512 MB RAM
CMD ["sh", "-c", "gunicorn 'api.app:create_app()' --bind 0.0.0.0:${PORT} --workers 2 --timeout 300"]
