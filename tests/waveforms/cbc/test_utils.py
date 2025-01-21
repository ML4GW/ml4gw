import lalsimulation
import pytest
import torch

from ml4gw.constants import MSUN
from ml4gw.waveforms.cbc import utils

NUM_SAMPLES = 100


@pytest.fixture(params=[10.0, 20.0, 30.0, 40.0, 50.0])
def fstart(request):
    return torch.tensor(request.param).double()


@pytest.fixture
def masses():
    m1 = (
        torch.rand(NUM_SAMPLES, dtype=torch.float64) * 45 + 5
    )  # Uniform distribution between 5 and 50
    m2 = (
        torch.rand(NUM_SAMPLES, dtype=torch.float64) * 45 + 5
    )  # Uniform distribution between 5 and 50
    return m1 * MSUN, m2 * MSUN


@pytest.fixture
def spins():
    s1 = (
        torch.rand(NUM_SAMPLES) * 2 - 1
    )  # Uniform distribution between -1 and 1
    s2 = (
        torch.rand(NUM_SAMPLES) * 2 - 1
    )  # Uniform distribution between -1 and 1
    return s1.double(), s2.double()


def test_chirp_time_bound(masses, spins, fstart):
    m1, m2 = masses
    s1, s2 = spins

    result = utils.chirp_time_bound(fstart, m1, m2, s1, s2)
    for i, (m1_val, m2_val, s1_val, s2_val) in enumerate(zip(m1, m2, s1, s2)):
        expected = lalsimulation.SimInspiralChirpTimeBound(
            fstart.item(),
            m1_val.item(),
            m2_val.item(),
            s1_val.item(),
            s2_val.item(),
        )
        print(result[i].float().item(), torch.tensor(expected))
        assert torch.isclose(
            result[i].float(), torch.tensor(expected), rtol=1e-3
        )


def test_final_black_hole_spin_bound(spins):
    s1, s2 = spins

    result = utils.final_black_hole_spin_bound(s1, s2)

    for i, (s1_val, s2_val) in enumerate(zip(s1, s2)):
        expected = lalsimulation.SimInspiralFinalBlackHoleSpinBound(
            s1_val.item(), s2_val.item()
        )
        assert torch.isclose(
            result[i].float(), torch.tensor(expected), rtol=1e-3
        )


def test_merge_time_bound(masses):
    m1, m2 = masses

    result = utils.merge_time_bound(m1, m2)

    for i, (m1_val, m2_val) in enumerate(zip(m1, m2)):
        expected = lalsimulation.SimInspiralMergeTimeBound(
            m1_val.item(), m2_val.item()
        )
        assert torch.isclose(
            result[i].float(), torch.tensor(expected), rtol=1e-3
        )


def test_ringdown_time_bound(masses, spins):
    m1, m2 = masses
    s1, s2 = spins
    s = utils.final_black_hole_spin_bound(s1, s2)
    total_mass = m1 + m2

    result = utils.ringdown_time_bound(total_mass, s)

    for i, (total_mass_val, s_val) in enumerate(zip(total_mass, s)):
        expected = lalsimulation.SimInspiralRingdownTimeBound(
            total_mass_val.item(), s_val.item()
        )
        assert torch.isclose(
            result[i].float(), torch.tensor(expected), rtol=1e-3
        )


def test_chirp_start_frequency_bound(fstart, spins, masses):
    m1, m2 = masses
    s1, s2 = spins
    tchirp = utils.chirp_time_bound(fstart, m1, m2, s1, s2)
    result = utils.chirp_start_frequency_bound(tchirp, m1, m2)

    for i, (tchirp_val, m1_val, m2_val) in enumerate(zip(tchirp, m1, m2)):
        expected = lalsimulation.SimInspiralChirpStartFrequencyBound(
            tchirp_val.item(), m1_val.item(), m2_val.item()
        )
        assert torch.isclose(
            result[i].float(), torch.tensor(expected), rtol=1e-3
        )
