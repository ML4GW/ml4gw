"""Benchmarks for Heterodyne."""

import torch
from constants import KERNEL_LEN, NUM_CHANNELS, NUM_SAMPLES, SAMPLE_RATE

from ml4gw.transforms import Heterodyne


def test_heterodyne_forward(benchmark, batch_size, device):
    chirp_mass = torch.tensor([5.0, 10.0, 20.0])
    heterodyner = Heterodyne(
        sample_rate=SAMPLE_RATE,
        kernel_length=KERNEL_LEN,
        chirp_mass=chirp_mass,
        return_type="both",
    ).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(heterodyner, x)
