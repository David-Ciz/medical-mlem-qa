FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --upgrade pip \
    && pip install -e .

COPY params.yaml params.lexis.example.yaml dvc.yaml RUN_STEPS.md RESULTS.md /app/
COPY scripts /app/scripts

ENTRYPOINT ["python", "-m"]
CMD ["mlem_qa_medseg.train", "--help"]
