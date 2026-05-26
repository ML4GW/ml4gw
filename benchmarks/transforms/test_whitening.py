"""Benchmarks for Whiten and FixedWhiten."""

import torch
from constants import KERNEL_LEN, NUM_CHANNELS, NUM_SAMPLES, SAMPLE_RATE

from ml4gw.transforms import FixedWhiten, Whiten

FDURATION = 0.5
NUM_SAMPLES_WHITEN = SAMPLE_RATE * 4


def test_whiten_forward(benchmark, batch_size, device, maybe_sync):
    whitener = Whiten(fduration=FDURATION, sample_rate=SAMPLE_RATE).to(device)
    num_freqs = NUM_SAMPLES_WHITEN // 2 + 1
    x = torch.randn(
        batch_size, NUM_CHANNELS, NUM_SAMPLES_WHITEN, device=device
    )
    psds = (
        torch.rand(batch_size, NUM_CHANNELS, num_freqs, device=device).abs()
        + 1e-20
    )
    benchmark(maybe_sync(whitener), x, psds)


def test_fixed_whiten_forward(benchmark, batch_size, device, maybe_sync):
    whitener = FixedWhiten(
        num_channels=NUM_CHANNELS,
        kernel_length=KERNEL_LEN,
        sample_rate=SAMPLE_RATE,
    )
    bg = torch.randn(NUM_CHANNELS, NUM_SAMPLES)
    whitener.fit(FDURATION, bg[0], bg[1], fftlength=KERNEL_LEN)
    whitener = whitener.to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(maybe_sync(whitener), x)
