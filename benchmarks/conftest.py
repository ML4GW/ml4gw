"""Shared fixtures and hooks for the benchmark suite."""

import sys
from pathlib import Path

import pytest
import torch

# Add benchmarks/ to sys.path so test files in
# subdirectories can import constants.py
sys.path.insert(0, str(Path(__file__).parent))


def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    # Pick the GPU with the most free memory to avoid interfering with other
    # processes. Fall back to CPU if no GPU is accessible.
    best_device = None
    best_free = 0
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best_device = torch.device(f"cuda:{i}")
        except Exception:
            continue
    return best_device if best_device is not None else torch.device("cpu")


@pytest.fixture(scope="session")
def device():
    return _select_device()


@pytest.fixture
def maybe_sync(device):
    """Wrap a callable with torch.cuda.synchronize() for accurate GPU timing.

    pytest-benchmark uses time.perf_counter, which on GPU only measures
    kernel-launch latency, not actual GPU execution time.

    On CPU this returns the callable unchanged.
    """
    if device.type == "cuda":

        def wrap(fn):
            def synced(*args, **kwargs):
                result = fn(*args, **kwargs)
                torch.cuda.synchronize()
                return result

            return synced

    else:

        def wrap(fn):
            return fn

    return wrap


@pytest.fixture(params=[32, 128, 512], ids=lambda x: f"batch_{x}")
def batch_size(request):
    return request.param


@pytest.hookimpl(tryfirst=True)
def pytest_benchmark_compare_machine_info(machine_info, compared_benchmark):
    # hz_actual fluctuates between runs
    for info in (machine_info, compared_benchmark["machine_info"]):
        info["cpu"].pop("hz_actual", None)
        info["cpu"].pop("hz_actual_friendly", None)
