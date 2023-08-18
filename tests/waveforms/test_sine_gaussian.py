import numpy as np
import pytest
import torch
from lalinference import BurstSineGaussian

from ml4gw.waveforms import SineGaussian


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[2.0, 4.0, 8.0])
def duration(request):
    return request.param


@pytest.fixture(params=[3.0, 10.0, 100.0, 55.0])
def quality(request):
    return torch.tensor(request.param, dtype=torch.float64)


@pytest.fixture(params=[100.0, 500.0, 800.0, 961.0])
def frequency(request):
    return torch.tensor(request.param, dtype=torch.float64)


# for amplitudes above ~7e-20, the difference between torch imp
# and lalsim is > ~1e-24. Our implementations are 1 to 1, so
# discrep must be from numerical issues?
@pytest.fixture(params=[1e-23, 1e-22, 1e-21, 1e-20, 7e-20])
def hrss(request):
    return torch.tensor(request.param, dtype=torch.float64)


@pytest.fixture(params=[0.0, np.pi / 2.0, np.pi, 2 * np.pi])
def phase(request):
    return torch.tensor(request.param, dtype=torch.float64)


@pytest.fixture(
    params=[
        0.0,
        0.5,
        1.0,
        0.1,
    ]
)
def eccentricity(request):
    return torch.tensor(request.param, dtype=torch.float64)


def test_sine_gaussian(
    duration,
    sample_rate,
    quality,
    frequency,
    hrss,
    phase,
    eccentricity,
):
    sine_gaussian = SineGaussian(sample_rate, duration)

    # calculate waveforms with torch implementation
    cross, plus = sine_gaussian(quality, frequency, hrss, phase, eccentricity)
    cross, plus = cross[0].numpy(), plus[0].numpy()

    quality = quality.item()
    frequency = frequency.item()
    phase = phase.item()
    eccentricity = eccentricity.item()
    hrss = hrss.item()

    # calculate waveform with lalsimulation
    hplus, hcross = BurstSineGaussian(
        Q=quality,
        centre_frequency=frequency,
        hrss=hrss,
        eccentricity=eccentricity,
        phase=phase,
        delta_t=1 / sample_rate,
    )
    hplus = hplus.data.data
    hcross = hcross.data.data

    # compare cross and plus polarizations
    n_samples = len(hplus)
    start, stop = (
        len(cross) // 2 - n_samples // 2,
        len(cross) // 2 + n_samples // 2 + 1,
    )
    cross, plus = cross[start:stop], plus[start:stop]

    assert np.allclose(
        cross, hcross, atol=1e-24
    )  # factor of 10 smaller than ligo noise floor
    assert np.allclose(
        plus, hplus, atol=1e-24
    )  # factor of 10 smaller than ligo noise floor
