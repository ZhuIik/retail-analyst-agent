FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/ZhuIik/retail-analyst-agent"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt ./
# Keep dependency installation cached when only the application source changes.
RUN sed '/^-e \.$/d' requirements.txt > /tmp/dependencies.txt \
    && pip install --no-cache-dir -r /tmp/dependencies.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir --no-deps -e . \
    && groupadd --gid 10001 analyst \
    && useradd --uid 10001 --gid analyst --no-create-home analyst \
    && mkdir -p data traces \
    && chown -R analyst:analyst data traces
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
# One worker: graph checkpoints and pending confirmations live in this process.
CMD ["uvicorn", "retail_analyst.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
