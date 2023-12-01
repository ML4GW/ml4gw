import lal
import lalsimulation
import numpy as np
import pytest
import torch
from astropy import units as u

import ml4gw.waveforms as waveforms


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[20.0, 30.0, 40.0])
def mass_1(request):
    return request.param


@pytest.fixture(params=[15.0, 25.0, 35.0])
def mass_2(request):
    return request.param


@pytest.fixture(params=[100.0, 1000.0])
def distance(request):
    return request.param


@pytest.fixture(params=[100.0, 1000.0])
def inclination(request):
    return request.param


def test_taylor_f2(mass_1, mass_2, distance, inclination, sample_rate):
    # Fix spins and coal. phase, ref, freq.
    phic, f_ref = 0.0, 15
    params = dict(
        m1=mass_1 * lal.MSUN_SI,
        m2=mass_2 * lal.MSUN_SI,
        S1x=0,
        S1y=0,
        S1z=0,
        S2x=0,
        S2y=0,
        S2z=0,
        distance=(distance * u.Mpc).to("m").value,
        inclination=inclination,
        phiRef=phic,
        longAscNodes=0.0,
        eccentricity=0.0,
        meanPerAno=0.0,
        deltaF=1.0 / sample_rate,
        f_min=10.0,
        f_ref=f_ref,
        f_max=100,
        approximant=lalsimulation.TaylorF2,
        LALpars=lal.CreateDict(),
    )
    hp_lal, hc_lal = lalsimulation.SimInspiralChooseFDWaveform(**params)
    lal_freqs = np.array(
        [hp_lal.f0 + ii * hp_lal.deltaF for ii in range(len(hp_lal.data.data))]
    )

    torch_freqs = torch.arange(
        params["f_min"], params["f_max"], params["deltaF"]
    )
    _params = torch.tensor(
        [mass_1, mass_2, distance, phic, inclination]
    ).repeat(
        10, 1
    )  # repeat along batch dim for testing
    batched_mass1 = _params[:, 0]
    batched_mass2 = _params[:, 1]
    batched_distance = _params[:, 2]
    batched_phic = _params[:, 3]
    batched_inclination = _params[:, 4]
    hp_torch, hc_torch = waveforms.TaylorF2(
        torch_freqs,
        batched_mass1,
        batched_mass2,
        batched_distance,
        batched_phic,
        batched_inclination,
        f_ref,
    )

    assert hp_torch.shape[0] == 10  # entire batch is returned

    # select only first element of the batch for further testing since
    # all are repeated
    hp_torch = hp_torch[0]
    hc_torch = hc_torch[0]
    # restrict between fmin and fmax
    lal_mask = (lal_freqs > params["f_min"]) & (lal_freqs < params["f_max"])
    torch_mask = (torch_freqs > params["f_min"]) & (
        torch_freqs < params["f_max"]
    )

    hp_lal_data = hp_lal.data.data[lal_mask]
    hc_lal_data = hc_lal.data.data[lal_mask]
    hp_torch = hp_torch[torch_mask]
    hc_torch = hc_torch[torch_mask]

    assert np.allclose(hp_lal_data.real, hp_torch.real)
    assert np.allclose(hp_lal_data.imag, hp_torch.imag)
    assert np.allclose(hc_lal_data.real, hc_torch.real)
    assert np.allclose(hc_lal_data.imag, hc_torch.imag)
