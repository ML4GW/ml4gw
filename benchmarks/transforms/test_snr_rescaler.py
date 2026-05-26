"""Benchmarks for SnrRescaler."""

import torch
from constants import KERNEL_LEN, NUM_CHANNELS, NUM_SAMPLES, SAMPLE_RATE

from ml4gw.transforms import SnrRescaler


def test_snr_rescaler_forward(benchmark, batch_size, device, maybe_sync):
    rescaler = SnrRescaler(
        num_channels=NUM_CHANNELS,
        sample_rate=SAMPLE_RATE,
        waveform_duration=KERNEL_LEN,
    )
    x = torch.randn(NUM_CHANNELS, SAMPLE_RATE * 4)
    rescaler.fit(x[0], x[1], fftlength=KERNEL_LEN)
    rescaler = rescaler.to(device)
    responses = torch.randn(
        batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device
    )
    benchmark(maybe_sync(rescaler), responses)
