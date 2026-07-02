"""Benchmarks for WaveformSampler and WaveformProjector."""

import torch
from constants import IFOS, NUM_SAMPLES, SAMPLE_RATE

from ml4gw.transforms import WaveformProjector, WaveformSampler


def test_waveform_sampler_forward(benchmark, batch_size, device, maybe_sync):
    n = batch_size * 2
    plus = torch.randn(n, NUM_SAMPLES, device=device)
    cross = torch.randn(n, NUM_SAMPLES, device=device)
    sampler = WaveformSampler(plus=plus, cross=cross).to(device)
    benchmark(maybe_sync(sampler), batch_size)


def test_waveform_projector_forward(benchmark, batch_size, device, maybe_sync):
    projector = WaveformProjector(ifos=IFOS, sample_rate=SAMPLE_RATE).to(
        device
    )
    dec = torch.rand(batch_size, device=device) * torch.pi - torch.pi / 2
    psi = torch.rand(batch_size, device=device) * torch.pi
    phi = torch.rand(batch_size, device=device) * 2 * torch.pi
    plus = torch.randn(batch_size, NUM_SAMPLES, device=device)
    cross = torch.randn(batch_size, NUM_SAMPLES, device=device)
    benchmark(maybe_sync(projector), dec, psi, phi, plus=plus, cross=cross)
