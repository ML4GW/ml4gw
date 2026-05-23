"""Benchmarks for MultiResolutionSpectrogram."""

import torch
from constants import KERNEL_LEN, NUM_CHANNELS, NUM_SAMPLES, SAMPLE_RATE

from ml4gw.transforms import MultiResolutionSpectrogram


def test_multi_resolution_spectrogram_forward(benchmark, batch_size, device):
    spectrogram = MultiResolutionSpectrogram(
        kernel_length=KERNEL_LEN,
        sample_rate=SAMPLE_RATE,
        n_fft=[64, 128, 256],
    ).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(spectrogram, x)
