"""Benchmarks for ResNet1D and ResNet2D."""

import torch
from constants import NUM_CHANNELS, NUM_SAMPLES

from ml4gw.nn.resnet.resnet_1d import ResNet1D
from ml4gw.nn.resnet.resnet_2d import ResNet2D


def test_resnet1d_forward(benchmark, batch_size, device, maybe_sync):
    arch = ResNet1D(
        in_channels=NUM_CHANNELS, layers=[2, 2, 2, 2], classes=1
    ).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(maybe_sync(arch), x)


def test_resnet2d_forward(benchmark, batch_size, device, maybe_sync):
    arch = ResNet2D(
        in_channels=NUM_CHANNELS, layers=[2, 2, 2, 2], classes=1
    ).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, 64, 64, device=device)
    benchmark(maybe_sync(arch), x)
