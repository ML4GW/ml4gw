"""Benchmarks for SpectralDensity."""

import pytest
import torch
from constants import KERNEL_LEN, NUM_CHANNELS, SAMPLE_RATE

from ml4gw.transforms import SpectralDensity


@pytest.fixture(params=["mean", "median"])
def average(request):
    return request.param


def test_spectral_density_forward(
    benchmark, batch_size, average, device, maybe_sync
):
    spectral_density = SpectralDensity(
        sample_rate=SAMPLE_RATE,
        fftlength=KERNEL_LEN,
        average=average,
    ).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, SAMPLE_RATE * 4, device=device)
    benchmark(maybe_sync(spectral_density), x)
