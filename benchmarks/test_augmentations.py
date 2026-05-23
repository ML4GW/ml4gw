"""Benchmarks for ml4gw/augmentations.py."""

import torch
from conftest import NUM_CHANNELS, NUM_SAMPLES

from ml4gw.augmentations import SignalInverter, SignalReverser


def test_signal_inverter_forward(benchmark, batch_size, device):
    inverter = SignalInverter(prob=0.5).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(inverter, x)


def test_signal_reverser_forward(benchmark, batch_size, device):
    reverser = SignalReverser(prob=0.5).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(reverser, x)
