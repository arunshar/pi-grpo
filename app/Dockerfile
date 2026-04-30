FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3-pip git curl build-essential \
 && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /srv
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install hatchling && pip install --no-cache-dir .

COPY app /srv/app
COPY observability /srv/observability
COPY evaluation /srv/evaluation
ENV PYTHONPATH=/srv

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
