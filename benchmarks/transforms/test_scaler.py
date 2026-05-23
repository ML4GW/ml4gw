"""Benchmarks for ChannelWiseScaler."""

import torch
from constants import NUM_CHANNELS, NUM_SAMPLES

from ml4gw.transforms import ChannelWiseScaler


def test_channel_wise_scaler_forward(
    benchmark, batch_size, device, maybe_sync
):
    scaler = ChannelWiseScaler(num_channels=NUM_CHANNELS)
    scaler.fit(torch.randn(NUM_CHANNELS, NUM_SAMPLES))
    scaler = scaler.to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(maybe_sync(scaler), x)


def test_channel_wise_scaler_reverse(
    benchmark, batch_size, device, maybe_sync
):
    scaler = ChannelWiseScaler(num_channels=NUM_CHANNELS)
    scaler.fit(torch.randn(NUM_CHANNELS, NUM_SAMPLES))
    scaler = scaler.to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(maybe_sync(scaler), x, reverse=True)
