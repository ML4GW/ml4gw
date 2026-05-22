"""Benchmarks for ml4gw/augmentations.py."""

import torch

from ml4gw.augmentations import SignalInverter, SignalReverser


def test_signal_inverter_forward(benchmark, batch_size, device):
    inverter = SignalInverter(prob=0.5).to(device)
    x = torch.randn(batch_size, 2, 2048, device=device)
    benchmark(inverter, x)


def test_signal_reverser_forward(benchmark, batch_size, device):
    reverser = SignalReverser(prob=0.5).to(device)
    x = torch.randn(batch_size, 2, 2048, device=device)
    benchmark(reverser, x)
