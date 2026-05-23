"""Benchmarks for SpectralDensity."""

import torch
from conftest import KERNEL_LEN, NUM_CHANNELS, SAMPLE_RATE

from ml4gw.transforms import SpectralDensity


def test_spectral_density_forward(benchmark, batch_size, device):
    spectral_density = SpectralDensity(
        sample_rate=SAMPLE_RATE,
        fftlength=KERNEL_LEN,
    ).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, SAMPLE_RATE * 4, device=device)
    benchmark(spectral_density, x)
