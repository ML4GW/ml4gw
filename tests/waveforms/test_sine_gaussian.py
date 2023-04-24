import numpy as np
import pytest
import torch
from lalinference import BurstSineGaussian

from ml4gw.waveforms import sine_gaussian


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[2, 4, 8])
def duration(request):
    return request.param


@pytest.fixture(params=[[3, 10, 100]])
def qualities(request):
    return torch.tensor(request.param)


@pytest.fixture(params=[[100, 500, 800]])
def frequencies(request):
    return torch.tensor(request.param)


@pytest.fixture(params=[[1e-22, 1e-21, 1e-20]])
def hrss(request):
    return torch.tensor(request.param)


@pytest.fixture(params=[[0, np.pi / 2, np.pi]])
def phases(request):
    return torch.tensor(request.param)


@pytest.fixture(params=[[0, 0.5, 1]])
def eccentricities(request):
    return torch.tensor(request.param)


def test_sine_gaussian(
    duration,
    sample_rate,
    qualities,
    frequencies,
    hrss,
    phases,
    eccentricities,
):

    # calculate waveforms with torch implementation
    waveforms = sine_gaussian(
        qualities,
        frequencies,
        hrss,
        phases,
        eccentricities,
        sample_rate,
        duration,
    )

    # calculate waveforms with lalsimulation implementation
    # and compare to torch version
    for i in range(len(qualities)):
        quality = qualities[i].item()
        frequency = frequencies[i].item()
        phase = phases[i].item()
        eccentricity = eccentricities[i].item()
        waveform = waveforms[i]

        # calculate waveform with lalsimulation
        hplus, hcross = BurstSineGaussian(
            Q=quality,
            centre_frequency=frequency,
            hrss=hrss[i].item(),
            eccentricity=eccentricity,
            phase=phase,
            delta_t=1 / sample_rate,
        )
        hplus = hplus.data.data
        hcross = hcross.data.data

        # compare hplus and hcross polarizations
        torch_polarizations = waveform.numpy()
        n_samples = len(hplus)
        start, stop = (
            torch_polarizations.shape[-1] // 2 - n_samples // 2,
            torch_polarizations.shape[-1] // 2 + n_samples // 2 + 1,
        )
        torch_polarizations = torch_polarizations[..., start:stop]

        assert np.allclose(torch_polarizations[0], hplus, atol=1e-25)
        assert np.allclose(torch_polarizations[1], hcross, atol=1e-25)
