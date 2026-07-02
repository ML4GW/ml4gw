"""Shared fixtures for waveform benchmarks."""

import pytest
import torch
from torch.distributions import Uniform


def _uniform(n, low, high, dtype, device):
    return Uniform(
        torch.as_tensor(low, dtype=dtype, device=device),
        torch.as_tensor(high, dtype=dtype, device=device),
    ).sample((n,))


@pytest.fixture
def cbc_inputs(batch_size, device):
    dtype = torch.float64
    u = lambda low, high: _uniform(batch_size, low, high, dtype, device)  # noqa: E731
    return {
        "chirp_mass": u(10, 100),
        "mass_ratio": u(0.25, 1.0),
        "chi1": u(-0.99, 0.99),
        "chi2": u(-0.99, 0.99),
        "distance": u(100, 1000),
        "phic": u(0, 2 * torch.pi),
        "inclination": u(0, torch.pi),
    }


@pytest.fixture(params=[2, 4, 8], ids=lambda x: f"dur_{x}")
def duration(request):
    return float(request.param)


@pytest.fixture
def spin_vectors(batch_size, device):
    dtype = torch.float64
    max_chi = 0.99

    def _make_spins():
        u = lambda low, high: _uniform(batch_size, low, high, dtype, device)  # noqa: E731
        chi = u(0, max_chi)
        theta = u(0, torch.pi)
        phi = u(0, 2 * torch.pi)
        return (
            chi * torch.sin(theta) * torch.cos(phi),
            chi * torch.sin(theta) * torch.sin(phi),
            chi * torch.cos(theta),
        )

    return _make_spins(), _make_spins()
