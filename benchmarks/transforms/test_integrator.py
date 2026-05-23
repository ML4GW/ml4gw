"""Benchmarks for integrators."""

import torch
from conftest import NUM_CHANNELS, NUM_SAMPLES, SAMPLE_RATE

from ml4gw.transforms import LeakyIntegrator, TophatIntegrator


def test_tophat_integrator_forward(benchmark, batch_size, device):
    tophat = TophatIntegrator(
        sample_rate=SAMPLE_RATE, integration_length=1
    ).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(tophat, x)


def test_leaky_integrator_forward(benchmark, batch_size, device):
    leaky = LeakyIntegrator(
        threshold=0.5, decay=0.1, lower_bound=0.0, integrate_value="score"
    ).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device)
    benchmark(leaky, x)
