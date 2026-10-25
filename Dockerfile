# VisionGuard web app (Flask UI + local inference). Model file is mounted at run time, not baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 MODEL_PATH=/models/champion.keras PORT=5000

WORKDIR /srv

# 1) dependencies first: this layer is reused from cache until a requirements file changes
COPY requirements-app.txt requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# 2) source code last: editing code only rebuilds these small layers
COPY src/config.py src/preprocess.py src/
COPY serving/ serving/
COPY app/ app/
COPY api/ api/
COPY wsgi.py .

# 3) unprivileged user
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/health', timeout=4)"
CMD ["python", "wsgi.py", "web"]
