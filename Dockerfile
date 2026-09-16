# ml4gw + ml4gw-buoy runtime image; the OSG plugin bridge drives the search.
# torch's CUDA-12 wheels bundle the CUDA libs, so no nvidia base is needed.
# Override TORCH_INDEX for a CPU-only build. Build: docker build -t ml4gw .

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
# Deps only, keyed on the lock so this layer survives ordinary source changes.
COPY pyproject.toml uv.lock README.md ./
RUN if [ -n "${TORCH_INDEX}" ]; then export UV_INDEX="${TORCH_INDEX}"; fi \
    && uv sync --frozen --no-default-groups --no-install-project

# ml4gw-buoy + its compiled deps (bilby-cython): the slow step, now cache-stable.
RUN uv pip install --python /opt/venv/bin/python "ml4gw-buoy==${BUOY_VERSION}"

# Project layer: build the git-versioned local ml4gw over buoy's pulled-in copy.
# The only step that re-runs on a source change; .git supplies the version.
COPY . /src
RUN uv pip install --python /opt/venv/bin/python --no-deps --reinstall-package ml4gw .

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
