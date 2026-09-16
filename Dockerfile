# ml4gw + ml4gw-buoy runtime image; the OSG plugin bridge drives the search.
# torch's CUDA-12 wheels bundle the CUDA userspace libs, so no nvidia/cuda base
# is needed — the GPU host supplies the driver. Override TORCH_INDEX for CPU.
# Build: docker build -t ml4gw:latest .

ARG PYTHON_VERSION=3.11

# ---- builder: resolve the locked environment into a self-contained venv ----
FROM python:${PYTHON_VERSION}-slim AS builder

# torch wheels (CUDA 12); set to a CPU index for a CPU-only build.
ARG TORCH_INDEX=
# ml4gw-buoy provides the aframe detector (buoy.Aframe).
ARG BUOY_VERSION=0.6.1

ENV PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# git: for setuptools-git-versioning; build-essential: for source-only wheels.
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv

WORKDIR /src
# .git is required for versioning; keep it out of .dockerignore.
COPY . /src

# Install ml4gw + runtime deps only, non-editable, exactly per uv.lock.
RUN if [ -n "${TORCH_INDEX}" ]; then export UV_INDEX="${TORCH_INDEX}"; fi \
    && uv sync --frozen --no-default-groups --no-editable

# Add ml4gw-buoy (pulls amplfi, gwpy, ligo-skymap, ...); ml4gw already present.
RUN uv pip install --python /opt/venv/bin/python "ml4gw-buoy==${BUOY_VERSION}"

# ---- runtime: just the venv on a clean base ----
FROM python:${PYTHON_VERSION}-slim AS runtime

# libgomp1: OpenMP runtime that numpy/scipy/torch link against.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Smoke-check default; the plugin's bridge overrides the command.
CMD ["python", "-c", "import ml4gw, buoy, torch; print('ml4gw+buoy ready; cuda:', torch.cuda.is_available())"]
