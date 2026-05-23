"""Benchmarks for ml4gw/gw.py public functions."""

import pytest
import torch
from conftest import IFOS, NUM_CHANNELS, NUM_SAMPLES, SAMPLE_RATE

from ml4gw.gw import (
    compute_antenna_responses,
    compute_ifo_snr,
    compute_network_snr,
    compute_observed_strain,
    get_ifo_geometry,
    reweight_snrs,
    shift_responses,
)


@pytest.fixture(scope="module")
def ifo_geometry(device):
    tensors, vertices = get_ifo_geometry(*IFOS)
    return tensors.to(device), vertices.to(device)


def test_compute_antenna_responses(
    benchmark, batch_size, ifo_geometry, device
):
    tensors, _ = ifo_geometry
    theta = torch.rand(batch_size, device=device) * torch.pi
    psi = torch.rand(batch_size, device=device) * torch.pi
    phi = torch.rand(batch_size, device=device) * 2 * torch.pi
    benchmark(
        compute_antenna_responses, theta, psi, phi, tensors, ["plus", "cross"]
    )


def test_shift_responses(benchmark, batch_size, ifo_geometry, device):
    _, vertices = ifo_geometry
    responses = torch.randn(
        batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device
    )
    theta = torch.rand(batch_size, device=device) * torch.pi
    phi = torch.rand(batch_size, device=device) * 2 * torch.pi
    benchmark(shift_responses, responses, theta, phi, vertices, SAMPLE_RATE)


def test_compute_observed_strain(benchmark, batch_size, ifo_geometry, device):
    tensors, vertices = ifo_geometry
    dec = torch.rand(batch_size, device=device) * torch.pi - torch.pi / 2
    psi = torch.rand(batch_size, device=device) * torch.pi
    phi = torch.rand(batch_size, device=device) * 2 * torch.pi
    hc = torch.randn(batch_size, NUM_SAMPLES, device=device)
    hp = torch.randn(batch_size, NUM_SAMPLES, device=device)
    benchmark(
        compute_observed_strain,
        dec,
        psi,
        phi,
        tensors,
        vertices,
        SAMPLE_RATE,
        plus=hp,
        cross=hc,
    )


def test_compute_ifo_snr(benchmark, batch_size, device):
    num_freqs = NUM_SAMPLES // 2 + 1
    responses = torch.randn(
        batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device
    )
    psd = torch.rand(NUM_CHANNELS, num_freqs, device=device) + 1e-20
    benchmark(compute_ifo_snr, responses, psd, SAMPLE_RATE)


def test_compute_network_snr(benchmark, batch_size, device):
    num_freqs = NUM_SAMPLES // 2 + 1
    responses = torch.randn(
        batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device
    )
    psd = torch.rand(NUM_CHANNELS, num_freqs, device=device) + 1e-20
    benchmark(compute_network_snr, responses, psd, SAMPLE_RATE)


def test_reweight_snrs(benchmark, batch_size, device):
    num_freqs = NUM_SAMPLES // 2 + 1
    responses = torch.randn(
        batch_size, NUM_CHANNELS, NUM_SAMPLES, device=device
    )
    psd = torch.rand(NUM_CHANNELS, num_freqs, device=device) + 1e-20
    target_snrs = torch.rand(batch_size, device=device) * 20 + 5
    benchmark(reweight_snrs, responses, target_snrs, psd, SAMPLE_RATE)
