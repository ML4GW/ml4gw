"""
Benchmarks for Snapshotter, OnlineAverager, StreamWhitener, and StreamRunner.
"""

import torch
from constants import KERNEL_LEN, NUM_CHANNELS, NUM_SAMPLES, SAMPLE_RATE

from ml4gw.nn.streaming import (
    OnlineAverager,
    Snapshotter,
    StreamRunner,
    StreamWhitener,
)
from ml4gw.transforms import SpectralDensity, Whiten

STRIDE_SIZE = 256
BATCH_SIZE = 32
UPDATE_SIZE = 256
NUM_UPDATES = 8
NUM_STREAMS = 2


def test_snapshotter_forward(benchmark, device, maybe_sync):
    snapshotter = Snapshotter(
        num_channels=NUM_CHANNELS,
        snapshot_size=NUM_SAMPLES,
        stride_size=STRIDE_SIZE,
        batch_size=BATCH_SIZE,
    ).to(device)
    update = torch.randn(NUM_CHANNELS, BATCH_SIZE * STRIDE_SIZE, device=device)
    state = torch.zeros(NUM_CHANNELS, NUM_SAMPLES - STRIDE_SIZE, device=device)
    benchmark(maybe_sync(snapshotter), update, state)


def test_online_averager_forward(benchmark, device, maybe_sync):
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
    benchmark(maybe_sync(online_averager), update, state)


def make_stream_whitener(batch_size):
    fduration = 2
    return StreamWhitener(
        kernel_length=KERNEL_LEN,
        psd_length=64,
        inference_sampling_rate=4,
        batch_size=batch_size,
        spectral_density=SpectralDensity(
            SAMPLE_RATE,
            fftlength=KERNEL_LEN + fduration,
            average="median",
            fast=True,
        ),
        transform=Whiten(fduration=fduration, sample_rate=SAMPLE_RATE),
    )


def test_stream_whitener_forward(benchmark, batch_size, device, maybe_sync):
    stream_whitener = make_stream_whitener(batch_size).to(device)
    x = torch.randn(
        NUM_STREAMS, NUM_CHANNELS, stream_whitener.input_size, device=device
    )
    benchmark(maybe_sync(stream_whitener), x)


def test_stream_runner_step(benchmark, batch_size, device, maybe_sync):
    stream_whitener = make_stream_whitener(batch_size)
    runner = StreamRunner(stream_whitener, torch.nn.Identity()).to(device)
    update = torch.randn(
        NUM_STREAMS,
        NUM_CHANNELS,
        batch_size * stream_whitener.stride_size,
        device=device,
    )
    state = torch.randn(
        NUM_STREAMS, NUM_CHANNELS, runner.buffer.state_size, device=device
    )

    def step(update):
        # Start each call from the same random state so that repeatedly passing
        # the same update doesn't fill the buffer with copies.
        runner.state = state
        return runner.step(update)

    benchmark(maybe_sync(step), update)
