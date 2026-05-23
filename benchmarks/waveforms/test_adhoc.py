"""Benchmarks for SineGaussian and Ringdown."""

import torch
from constants import SAMPLE_RATE
from torch.distributions import Uniform

from ml4gw.waveforms import Ringdown, SineGaussian

DURATION = 4.0


def test_sine_gaussian_forward(benchmark, batch_size, device, maybe_sync):
    model = SineGaussian(sample_rate=SAMPLE_RATE, duration=DURATION).to(device)
    dtype = torch.float64

    def uniform(low, high):
        return Uniform(
            torch.as_tensor(low, dtype=dtype, device=device),
            torch.as_tensor(high, dtype=dtype, device=device),
        ).sample((batch_size,))

    quality = uniform(10, 100)
    frequency = uniform(50, 450)
    hrss = uniform(1e-23, 1e-21)
    phase = uniform(0, 2 * torch.pi)
    eccentricity = uniform(0, 1)
    benchmark(maybe_sync(model), quality, frequency, hrss, phase, eccentricity)


def test_ringdown_forward(benchmark, batch_size, device, maybe_sync):
    model = Ringdown(sample_rate=SAMPLE_RATE, duration=DURATION).to(device)
    dtype = torch.float64

    def uniform(low, high):
        return Uniform(
            torch.as_tensor(low, dtype=dtype, device=device),
            torch.as_tensor(high, dtype=dtype, device=device),
        ).sample((batch_size,))

    frequency = uniform(150, 350)
    quality = uniform(2, 17)
    epsilon = uniform(0, 0.1)
    phase = uniform(0, 2 * torch.pi)
    inclination = uniform(0, torch.pi)
    distance = uniform(100, 1000)
    benchmark(
        maybe_sync(model),
        frequency,
        quality,
        epsilon,
        phase,
        inclination,
        distance,
    )
