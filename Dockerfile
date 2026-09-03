# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy source code first, giving ownership to appuser
COPY --chown=appuser:appuser . .

# Ensure runtime-writable dirs exist and are owned by appuser
RUN mkdir -p vectorstore .hf_cache .uploaded_files retrive_docs && \
    touch chat_history.sqlite && \
    chown -R appuser:appuser vectorstore .hf_cache .uploaded_files retrive_docs chat_history.sqlite

# Switch to non-privileged user LAST, right before running the app
USER appuser

EXPOSE 8000

CMD uvicorn main:app --host=0.0.0.0 --port=8000