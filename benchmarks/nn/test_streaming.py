"""Benchmarks for Snapshotter and OnlineAverager."""

import torch
from constants import NUM_CHANNELS, NUM_SAMPLES

from ml4gw.nn.streaming import OnlineAverager, Snapshotter

STRIDE_SIZE = 256
BATCH_SIZE = 32
UPDATE_SIZE = 256
NUM_UPDATES = 8


def test_snapshotter_forward(benchmark, device):
    snapshotter = Snapshotter(
        num_channels=NUM_CHANNELS,
        snapshot_size=NUM_SAMPLES,
        stride_size=STRIDE_SIZE,
        batch_size=BATCH_SIZE,
    ).to(device)
    update = torch.randn(NUM_CHANNELS, BATCH_SIZE * STRIDE_SIZE, device=device)
    state = torch.zeros(NUM_CHANNELS, NUM_SAMPLES - STRIDE_SIZE, device=device)
    benchmark(snapshotter, update, state)


def test_online_averager_forward(benchmark, device):
    online_averager = OnlineAverager(
        update_size=UPDATE_SIZE,
        batch_size=BATCH_SIZE,
        num_updates=NUM_UPDATES,
        num_channels=NUM_CHANNELS,
    ).to(device)
    update = torch.randn(
        BATCH_SIZE, NUM_CHANNELS, NUM_UPDATES * UPDATE_SIZE, device=device
    )
    state = online_averager.get_initial_state()
    benchmark(online_averager, update, state)
