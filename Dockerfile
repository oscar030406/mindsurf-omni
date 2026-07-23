# Inference only. Training runs on the host with its own environment; putting
# both in one image would ship the training dependencies to every server that
# only needs to answer requests.
FROM python:3.12-slim AS base

# libsndfile is needed by soundfile at runtime and is not a Python package, so
# pip cannot supply it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so a code change does not invalidate the layer
# that takes minutes to build. The runtime set comes from pyproject rather than
# being repeated here -- a hand-written list drifts, and the drift only shows
# up as an ImportError inside a running container.
COPY pyproject.toml README.md ./
RUN mkdir -p src/mindsurf_omni \
    && touch src/mindsurf_omni/__init__.py \
    && pip install --no-cache-dir . \
    && rm -rf src

COPY src/ ./src/
COPY assets/tokenizer/ ./assets/tokenizer/
# The licence record is served at /v1/licence, so it ships with the image.
COPY configs/release/ ./configs/release/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# Weights are not baked in: they are 359 MB, carry a non-commercial licence,
# and change independently of this image. Mount them at /app/weights.
VOLUME ["/app/weights"]

# Non-root, because a service that only reads weights and answers HTTP has no
# reason to be able to write anywhere else.
RUN useradd --create-home --uid 10001 omni && chown -R omni /app
USER omni

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "--factory", "mindsurf_omni.service.app:create_app", \
     "--host", "0.0.0.0", "--port", "8000"]
