import numpy as np
import pytest
import torch
from lalsimulation import (
    GenerateStringCusp,
    GenerateStringKink,
    GenerateStringKinkKink,
)

from ml4gw.waveforms import GenerateString


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


# The LAL cosmic strain can only generate up to 9 second of data.
@pytest.fixture(params=[2.0, 4.0, 8.0, 9.0, 11.0])
def duration(request):
    return request.param


@pytest.fixture(params=[-4.0 / 3.0, -5.0 / 3.0, -6.0 / 3.0])
def power(request):
    return torch.tensor(request.param, dtype=torch.float64)


# for amplitudes above ~7e-20, the difference between torch imp
# and lalsim is > ~1e-24. Our implementations are 1 to 1, so
# discrep must be from numerical issues?
@pytest.fixture(params=[1e-23, 1e-22, 1e-21, 1e-20, 5e-20])
def amplitude(request):
    return torch.tensor(request.param, dtype=torch.float64)


@pytest.fixture(params=[250.0, 500.0, 1000.0, 2000.0])
def f_high(request):
    return torch.tensor(request.param, dtype=torch.float64)


def test_strings(sample_rate, duration, power, amplitude, f_high):

    # ML4GW
    generate_string = GenerateString(
        sample_rate=sample_rate, duration=duration
    )
    if power == -6.0 / 3.0:
        f_high = torch.Tensor([sample_rate / 2])
    cross, plus = generate_string(
        power=power, amplitude=amplitude, f_high=f_high
    )
    cross, plus = cross[0].numpy(), plus[0].numpy()
    ml4gw_samples = len(plus)

    # Data type conversion
    power = power.item()
    amplitude = amplitude.item()
    f_high = f_high.item()

    # LAL
    if power == -4.0 / 3.0:
        hplus, hcross = GenerateStringCusp(
            amplitude=amplitude, f_high=f_high, delta_t=1 / sample_rate
        )
    if power == -5.0 / 3.0:
        hplus, hcross = GenerateStringKink(
            amplitude=amplitude, f_high=f_high, delta_t=1 / sample_rate
        )
    if power == -6.0 / 3.0:
        hplus, hcross = GenerateStringKinkKink(
            amplitude=amplitude, delta_t=1 / sample_rate
        )
    hplus = hplus.data.data
    hcross = hcross.data.data
    lal_samples = len(hplus)

    if lal_samples < ml4gw_samples:

        start, stop = (
            ml4gw_samples // 2 - lal_samples // 2,
            ml4gw_samples // 2 + lal_samples // 2 + 1,
        )
        cross, plus = cross[start:stop], plus[start:stop]
    else:
        start, stop = (
            len(hplus) // 2 - ml4gw_samples // 2,
            len(hplus) // 2 + ml4gw_samples // 2,
        )
        hcross, hplus = hcross[start:stop], hplus[start:stop]

    assert np.allclose(
        cross, hcross, atol=1e-24
    )  # factor of 10 smaller than ligo noise floor
    assert np.allclose(
        plus, hplus, atol=1e-24
    )  # factor of 10 smaller than ligo noise floor