FROM python:3.12.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN useradd --create-home --uid 10001 app

COPY pyproject.toml README.md ./
COPY assistant ./assistant
COPY evals ./evals
RUN pip install ".[pdf]"

COPY config ./config
COPY data ./data
RUN mkdir -p evals/reports && chown app evals/reports

USER app
EXPOSE 8000
CMD ["uvicorn", "assistant.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
