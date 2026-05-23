"""Benchmarks for ml4gw/spectral.py functions."""

import pytest
import torch
from constants import NUM_CHANNELS, NUM_SAMPLES, SAMPLE_RATE

from ml4gw import spectral

FDURATION = 0.5
SCALE = 1.0 / (SAMPLE_RATE * NUM_SAMPLES)


@pytest.fixture(params=[16, 32, 64], ids=lambda x: f"seg_{x}")
def num_segments(request):
    return request.param


def test_fast_spectral_density(
    benchmark, batch_size, num_segments, device, maybe_sync
):
    window = torch.hann_window(NUM_SAMPLES, device=device)
    x = torch.randn(
        batch_size, NUM_CHANNELS, num_segments * NUM_SAMPLES, device=device
    )
    benchmark(
        maybe_sync(spectral.fast_spectral_density),
        x,
        NUM_SAMPLES,
        NUM_SAMPLES,
        window,
        SCALE,
    )


def test_spectral_density(
    benchmark, batch_size, num_segments, device, maybe_sync
):
    window = torch.hann_window(NUM_SAMPLES, device=device)
    x = torch.randn(
        batch_size, NUM_CHANNELS, num_segments * NUM_SAMPLES, device=device
    )
    benchmark(
        maybe_sync(spectral.spectral_density),
        x,
        NUM_SAMPLES,
        NUM_SAMPLES,
        window,
        SCALE,
    )


def test_truncate_inverse_power_spectrum(benchmark, device, maybe_sync):
    num_freqs = NUM_SAMPLES // 2 + 1
    psd = torch.rand(1, NUM_CHANNELS, num_freqs, device=device) + 1e-20
    benchmark(
        maybe_sync(spectral.truncate_inverse_power_spectrum),
        psd,
        FDURATION,
        SAMPLE_RATE,
    )


def test_normalize_by_psd(benchmark, batch_size, device, maybe_sync):
    n = NUM_SAMPLES * 4
    x = torch.randn(batch_size, NUM_CHANNELS, n, device=device)
    num_freqs = n // 2 + 1
    psd = torch.rand(NUM_CHANNELS, num_freqs, device=device) + 1e-20
    pad = int(FDURATION * SAMPLE_RATE) // 2
    benchmark(maybe_sync(spectral.normalize_by_psd), x, psd, SAMPLE_RATE, pad)


def test_whiten(benchmark, batch_size, device, maybe_sync):
    n = NUM_SAMPLES * 4
    x = torch.randn(batch_size, NUM_CHANNELS, n, device=device)
    num_freqs = n // 2 + 1
    psd = torch.rand(NUM_CHANNELS, num_freqs, device=device) + 1e-20
    fduration = torch.hann_window(
        int(FDURATION * SAMPLE_RATE), dtype=torch.float64, device=device
    )
    benchmark(maybe_sync(spectral.whiten), x, psd, fduration, SAMPLE_RATE)
