"""Benchmarks for IIRFilter."""

import torch
from conftest import NUM_CHANNELS, NUM_SAMPLES, SAMPLE_RATE

from ml4gw.transforms import IIRFilter


def test_iirfilter_forward(benchmark, batch_size, device):
    filter = IIRFilter(N=4, Wn=[32.0, 512.0], btype="band", fs=SAMPLE_RATE).to(
        device
    )
    x = torch.randn(
        batch_size,
        NUM_CHANNELS,
        NUM_SAMPLES,
        dtype=torch.float64,
        device=device,
    )
    benchmark(filter, x)
