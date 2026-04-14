import numpy as np
import pytest
import torch
from lalsimulation import SimBurstGaussian

from ml4gw.waveforms import Gaussian


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[2.0, 3.0, 4.0, 6.0])
def duration(request):
    return request.param


@pytest.fixture(params=[1.0e-23, 1.0e-21, 1.0e-19, 1.0e-17])
def hrss(request):
    return torch.tensor(request.param, dtype=torch.float64)


@pytest.fixture(params=[0.01, 0.1, 1, 5])
def gaussian_width(request):
    return torch.tensor(request.param, dtype=torch.float64)


def test_gaussian(sample_rate, duration, hrss, gaussian_width):
    # ML4GW
    gaussian = Gaussian(sample_rate=sample_rate, duration=duration)
    cross, plus = gaussian(hrss=hrss, gaussian_width=gaussian_width)
    cross, plus = cross[0].numpy(), plus[0].numpy()
    ml4gw_samples = len(cross)

    # Conversion
    hrss = hrss.item()
    gaussian_width = gaussian_width.item()

    # LAL
    hplus, hcross = SimBurstGaussian(
        duration=gaussian_width, hrss=hrss, delta_t=1 / sample_rate
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
        # Even length data point.
        start, stop = (
            lal_samples // 2 - ml4gw_samples // 2,
            lal_samples // 2 + ml4gw_samples // 2,
        )
        hcross, hplus = hcross[start:stop], hplus[start:stop]

    assert np.allclose(
        cross, hcross, atol=1e-24
    )  # factor of 10 smaller than ligo noise floor
    assert np.allclose(
        plus, hplus, atol=1e-24
    )  # factor of 10 smaller than ligo noise floor