"""Benchmarks for GroupNorm1D."""

import torch
from conftest import NUM_SAMPLES

from ml4gw.nn.norm import GroupNorm1D


def test_group_norm_forward(benchmark, batch_size, device):
    norm = GroupNorm1D(num_groups=4, num_channels=16).to(device)
    x = torch.randn(batch_size, 16, NUM_SAMPLES, device=device)
    benchmark(norm, x)
