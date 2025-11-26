# ---- Production Dockerfile for K8s (Gunicorn + gevent) ----
FROM python:3.11.9-slim

# Create non-root user with stable UID/GID for K8s volumes
RUN useradd -m -u 1000 appuser

# System deps (fonts + sqlite + build essentials)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libssl-dev libffi-dev python3-dev \
    git sqlite3 fontconfig fonts-noto-cjk \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Leverage layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn gevent

# App code
COPY . .

# Writable data dir for SQLite / temp files
RUN mkdir -p /data && chown -R appuser:appuser /usr/src/app /data

# Sensible defaults; override via envs in K8s
ENV APP_ENV=docker \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    D1_BINDING=/data/app.db \
    GUNICORN_WORKERS=2 \
    GUNICORN_TIMEOUT=30

EXPOSE 8000
USER appuser

# Optional: Docker healthcheck (K8s will still use its own probes)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import os,sys,urllib.request as u; url='http://127.0.0.1:'+os.environ.get('PORT','8000')+'/healthz'; sys.exit(0 if u.urlopen(url).getcode()==200 else 1)"

# Run in production with Gunicorn + gevent
CMD ["sh","-c","exec gunicorn -k gevent -w ${GUNICORN_WORKERS} -b 0.0.0.0:${PORT} app:app --timeout ${GUNICORN_TIMEOUT} --access-logfile - --error-logfile -"]
