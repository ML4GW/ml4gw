import lalsimulation
import pytest
import torch

from ml4gw.waveforms.cbc.coefficients import (
    taylor_t2_timing_0pn_coeff,
    taylor_t2_timing_2pn_coeff,
    taylor_t2_timing_4pn_coeff,
)

NUM_SAMPLES = 100


@pytest.fixture
def mass_samples():
    # uniform random samples in the range [5, 100]
    m1 = torch.rand(NUM_SAMPLES) * 95 + 5
    m2 = torch.rand(NUM_SAMPLES) * 95 + 5
    total_mass = m1 + m2
    eta = (m1 * m2) / (total_mass**2)
    return total_mass, eta


def test_taylor_t2_timing_0pn_coeff(mass_samples):
    total_mass, eta = mass_samples
    for tm, e in zip(total_mass, eta):
        expected = lalsimulation.SimInspiralTaylorT2Timing_0PNCoeff(
            tm.item(), e.item()
        )
        result = taylor_t2_timing_0pn_coeff(tm.unsqueeze(0), e.unsqueeze(0))
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)


def test_taylor_t2_timing_2pn_coeff(mass_samples):
    total_mass, eta = mass_samples
    for tm, e in zip(total_mass, eta):
        expected = lalsimulation.SimInspiralTaylorT2Timing_2PNCoeff(
            tm.item(), e.item()
        )
        result = taylor_t2_timing_2pn_coeff(tm.unsqueeze(0), e.unsqueeze(0))
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)


def test_taylor_t2_timing_4pn_coeff(mass_samples):
    total_mass, eta = mass_samples
    for tm, e in zip(total_mass, eta):
        expected = lalsimulation.SimInspiralTaylorT2Timing_4PNCoeff(
            tm.item(), e.item()
        )
        result = taylor_t2_timing_4pn_coeff(tm.unsqueeze(0), e.unsqueeze(0))
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)
