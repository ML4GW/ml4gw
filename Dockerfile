# ml4gw + ml4gw-buoy runtime image.
#
# Builds a reproducible ml4gw + (CUDA) torch environment and installs ml4gw-buoy
# on top, providing the aframe detector (buoy.Aframe). The targeted-search logic
# is driven by the OSG plugin's bridge (shipped per job), so no analysis script
# is baked into the image.
#
# The PyPI torch wheels bundle the CUDA 12 userspace libraries (see the
# nvidia-*-cu12 entries in uv.lock), so no nvidia/cuda base is needed: the GPU
# host supplies the driver via the container runtime, and the wheels supply the
# rest. For a CPU-only image, override TORCH_INDEX (see below) — that also needs
# a CPU-resolved lock, so prefer a separate lock for that variant.
#
# First draft — not yet built here (no docker on the authoring machine). Most
# likely to need tuning: the uv sync flags, the setuptools-git-versioning step
# (why the context needs .git), and whether ml4gw-buoy's deps (amplfi, etc.)
# stay compatible with the locked torch.
#
# Build:  docker build -t ml4gw:latest .
# Smoke:  docker run --rm ml4gw:latest   # imports ml4gw, buoy, torch

ARG PYTHON_VERSION=3.11

# ---- builder: resolve the locked environment into a self-contained venv ----
FROM python:${PYTHON_VERSION}-slim AS builder

# Standard torch wheels (CUDA 12). Override for CPU with e.g.
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=
# ml4gw-buoy provides the aframe detector (buoy.Aframe) the search wraps.
ARG BUOY_VERSION=0.6.1

ENV PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# git: setuptools-git-versioning reads the tag/sha to compute the version.
# build-essential: some buoy deps (e.g. bilby-cython) build from source where
# no wheel exists for the target platform. Builder-only; runtime stays slim.
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv

WORKDIR /src
# .git is required for versioning; keep it out of .dockerignore.
COPY . /src

# Install only the runtime deps + ml4gw (no dev/docs), non-editable so the
# runtime stage needs no source. --frozen honours uv.lock exactly.
RUN if [ -n "${TORCH_INDEX}" ]; then export UV_INDEX="${TORCH_INDEX}"; fi \
    && uv sync --frozen --no-default-groups --no-editable

# Add ml4gw-buoy into the same venv. ml4gw is already satisfied by the sync
# above, so it is not reinstalled; buoy pulls the rest (amplfi, gwpy,
# ligo-skymap -> lalsuite, huggingface_hub, ...).
RUN uv pip install --python /opt/venv/bin/python "ml4gw-buoy==${BUOY_VERSION}"

# ---- runtime: the venv + the search tool on a clean base ----
FROM python:${PYTHON_VERSION}-slim AS runtime

# libgomp1: OpenMP runtime that numpy/scipy/torch link against.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Default is only for `docker run`; the plugin's bridge overrides the command.
CMD ["python", "-c", "import ml4gw, buoy, torch; print('ml4gw+buoy ready; cuda:', torch.cuda.is_available())"]
