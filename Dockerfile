# ------------------------------------------------------------
# Production-ready Dockerfile for penghu app
# - Default PORT set to 8000
# - Uses gunicorn + gevent for production serving
# - Creates a non-root user and switches to it for runtime
# - Installs fonts and common build deps
# - Includes a simple HEALTHCHECK
# ------------------------------------------------------------

FROM python:3.11.9-slim

# Create a non-root user early so we can chown files to it
RUN useradd -m appuser || true

# Install system dependencies (fonts, tools for building some wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential libssl-dev libffi-dev python3-dev \
       git sqlite3 fontconfig fonts-noto-cjk curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/app

# Copy requirements and install Python deps (as root)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn gevent

# (Optional) Install Locust if you need it inside this image for load tests
# If you don't need locust in production, remove this line to keep image smaller
RUN pip install --no-cache-dir locust==2.29.0 || true

# Ensure PATH contains user-site for completeness (kept from original)
ENV PATH="/root/.local/bin:${PATH}"

# Copy application code
COPY . .

# Try to initialize sqlite DB at build time, ignore errors (keeps parity with old Dockerfile)
# If you prefer runtime init, remove this line.
RUN python init_db.py || true

# Environment variables: default PORT -> 8000 (can be overridden by K8s env)
ENV APP_ENV=docker \
    PORT=8000 \
    PYTHONUNBUFFERED=1

# Make sure the application directory is owned by non-root user
RUN chown -R appuser:appuser /usr/src/app

# Expose the chosen port
EXPOSE 8000

# Switch to non-root user for runtime
USER appuser

# A lightweight healthcheck (container must have curl; we installed it earlier)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://127.0.0.1:${PORT}/healthz || exit 1

# Production entrypoint: gunicorn with gevent workers
# Ensure the module path "app:app" matches your application (app.py -> app)
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:${PORT}", "app:app"]
