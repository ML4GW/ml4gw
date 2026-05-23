"""Shared fixtures and constants for the benchmark suite."""

import sys
from pathlib import Path

import pytest
import torch

# Make this file importable from subdirectory test files
sys.path.insert(0, str(Path(__file__).parent))

# Default signal parameters used across the benchmark suite.
SAMPLE_RATE = 2048
KERNEL_LEN = 1.0
NUM_SAMPLES = int(KERNEL_LEN * SAMPLE_RATE)

IFOS = ["H1", "L1"]
NUM_CHANNELS = len(IFOS)


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


@pytest.fixture(params=[32, 128, 512], ids=lambda x: f"batch_{x}")
def batch_size(request):
    return request.param


@pytest.hookimpl(tryfirst=True)
def pytest_benchmark_compare_machine_info(machine_info, compared_benchmark):
    # hz_actual fluctuates between runs
    for info in (machine_info, compared_benchmark["machine_info"]):
        info["cpu"].pop("hz_actual", None)
        info["cpu"].pop("hz_actual_friendly", None)
