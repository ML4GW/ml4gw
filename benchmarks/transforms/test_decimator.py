"""Benchmarks for Decimator."""

import torch
from constants import NUM_CHANNELS, NUM_SAMPLES, SAMPLE_RATE

from ml4gw.transforms import Decimator


def test_decimator_forward(benchmark, batch_size, device):
    schedule = torch.tensor([[0, 1, 256]], dtype=torch.int)
    decimator = Decimator(sample_rate=SAMPLE_RATE, schedule=schedule).to(
        device
    )
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(decimator, x)


def test_decimator_split_forward(benchmark, batch_size, device):
    schedule = torch.tensor([[0, 1, 256]], dtype=torch.int)
    decimator = Decimator(
        sample_rate=SAMPLE_RATE, schedule=schedule, split=True
    ).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(decimator, x)
