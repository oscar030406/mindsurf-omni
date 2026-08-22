# Two images out of one file. `base` is the API surface: it starts, answers
# /health, and refuses every engine with the reason. `dictation` is the
# product -- transcribe and polish -- and is the default target, because an
# image that cannot run the product is not a release artifact of it. Training
# still runs on the host: nothing here installs a training set.
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


# The product. Separate stage rather than a fatter base, because this is torch
# and its CUDA wheels -- gigabytes a deployment that only wants the API surface
# has no use for. `docker build --target base .` is that deployment.
#
# It exists because for the whole of phase two the published image could not
# run the phase-two product: `pip install .` takes the runtime set, and the
# runtime set has neither funasr nor torch. The failure was invisible from the
# outside -- the container started and /health answered.
FROM base AS dictation

# Which torch. pip default index carries the CUDA build -- gigabytes of runtime
# a CPU host cannot use -- and PyTorch publishes one index per accelerator and
# expects the builder to choose. Passed through rather than pinned, because the
# right answer is a property of the machine that will run this, not of this file:
#   --build-arg PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIP_EXTRA_INDEX_URL=""

USER root
# src and pyproject are already in place from `base`; only the extra is new, so
# this layer rebuilds when the dependency set changes and not when code does.
RUN pip install --no-cache-dir ".[dictation]" \
    && chown -R omni /app
USER omni

# Same entrypoint. What changed is that MINDSURF_POLISH and MINDSURF_ASR now
# have something to load.
