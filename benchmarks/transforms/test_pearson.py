"""Benchmarks for ShiftedPearsonCorrelation."""

import torch
from constants import NUM_CHANNELS, NUM_SAMPLES

from ml4gw.transforms import ShiftedPearsonCorrelation


def test_shifted_pearson_forward(benchmark, batch_size, device, maybe_sync):
    pearson = ShiftedPearsonCorrelation(max_shift=128).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    y = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(maybe_sync(pearson), x, y)
