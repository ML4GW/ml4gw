import lal
import lalsimulation
import numpy as np
import pytest
import torch
from astropy import units as u

import ml4gw.waveforms as waveforms


@pytest.fixture(params=[128, 256])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[20.0, 30.0, 40.0])
def mass_1(request):
    return request.param


@pytest.fixture(params=[15.0, 25.0, 35.0])
def mass_2(request):
    return request.param


@pytest.fixture(params=[15.0, 30.0])
def chirp_mass(request):
    return request.param


@pytest.fixture(params=[0.99, 0.5])
def mass_ratio(request):
    return request.param


@pytest.fixture(params=[0.0, 0.5])
def chi1z(request):
    return request.param


@pytest.fixture(params=[-0.1, 0.1])
def chi2z(request):
    return request.param


@pytest.fixture(params=[100.0, 1000.0])
def distance(request):
    return request.param


@pytest.fixture(params=[100.0, 1000.0])
def inclination(request):
    return request.param


def test_taylor_f2(
    chirp_mass, mass_ratio, chi1z, chi2z, distance, inclination, sample_rate
):
    mass_2 = chirp_mass * (1 + mass_ratio) ** 0.2 / mass_ratio**0.6
    mass_1 = mass_ratio * mass_2
    # Fix coal. phase, ref, freq.
    phic, f_ref = 0.0, 25
    params = dict(
        m1=mass_1 * lal.MSUN_SI,
        m2=mass_2 * lal.MSUN_SI,
        S1x=0,
        S1y=0,
        S1z=chi1z,
        S2x=0,
        S2y=0,
        S2z=chi2z,
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
        [chirp_mass, mass_ratio, chi1z, chi2z, distance, phic, inclination]
    ).repeat(
        10, 1
    )  # repeat along batch dim for testing
    batched_chirp_mass = _params[:, 0]
    batched_mass_ratio = _params[:, 1]
    batched_chi1 = _params[:, 2]
    batched_chi2 = _params[:, 3]
    batched_distance = _params[:, 4]
    batched_phic = _params[:, 5]
    batched_inclination = _params[:, 6]
    hc_torch, hp_torch = waveforms.TaylorF2()(
        torch_freqs,
        batched_chirp_mass,
        batched_mass_ratio,
        batched_chi1,
        batched_chi2,
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

    assert np.allclose(
        1e21 * hp_lal_data.real, 1e21 * hp_torch.real.numpy(), atol=1e-3
    )
    assert np.allclose(
        1e21 * hp_lal_data.imag, 1e21 * hp_torch.imag.numpy(), atol=1e-3
    )
    assert np.allclose(
        1e21 * hc_lal_data.real, 1e21 * hc_torch.real.numpy(), atol=1e-3
    )
    assert np.allclose(
        1e21 * hc_lal_data.imag, 1e21 * hc_torch.imag.numpy(), atol=1e-3
    )


def test_phenom_d(
    chirp_mass, mass_ratio, chi1z, chi2z, distance, inclination, sample_rate
):
    total_mass = chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio**0.6
    mass_1 = total_mass / (1 + mass_ratio)
    mass_2 = mass_1 * mass_ratio
    phic, f_ref = 0.0, 25

    params = dict(
        m1=mass_1 * lal.MSUN_SI,
        m2=mass_2 * lal.MSUN_SI,
        S1x=0,
        S1y=0,
        S1z=chi1z,
        S2x=0,
        S2y=0,
        S2z=chi2z,
        distance=(distance * u.Mpc).to("m").value,
        inclination=inclination,
        phiRef=phic,
        longAscNodes=0.0,
        eccentricity=0.0,
        meanPerAno=0.0,
        deltaF=1.0 / sample_rate,
        f_min=10.0,
        f_ref=f_ref,
        f_max=300,
        approximant=lalsimulation.IMRPhenomD,
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
        [chirp_mass, mass_ratio, chi1z, chi2z, distance, phic, inclination]
    ).repeat(
        10, 1
    )  # repeat along batch dim for testing
    batched_chirp_mass = _params[:, 0]
    batched_mass_ratio = _params[:, 1]
    batched_chi1 = _params[:, 2]
    batched_chi2 = _params[:, 3]
    batched_distance = _params[:, 4]
    batched_phic = _params[:, 5]
    batched_inclination = _params[:, 6]
    hc_torch, hp_torch = waveforms.IMRPhenomD()(
        torch_freqs,
        batched_chirp_mass,
        batched_mass_ratio,
        batched_chi1,
        batched_chi2,
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

    assert np.allclose(
        1e21 * hp_lal_data.real, 1e21 * hp_torch.real.numpy(), atol=3e-4
    )
    assert np.allclose(
        1e21 * hp_lal_data.imag, 1e21 * hp_torch.imag.numpy(), atol=3e-4
    )
    assert np.allclose(
        1e21 * hc_lal_data.real, 1e21 * hc_torch.real.numpy(), atol=3e-4
    )
    assert np.allclose(
        1e21 * hc_lal_data.imag, 1e21 * hc_torch.imag.numpy(), atol=3e-4
    )


def test_phenom_p(chirp_mass, mass_ratio, chi1z, chi2z, distance, sample_rate):
    mass_2 = chirp_mass * (1 + mass_ratio) ** 0.2 / mass_ratio**0.6
    mass_1 = mass_2 * mass_ratio
    if mass_2 > mass_1:
        mass_1, mass_2 = mass_2, mass_1
        mass_ratio = 1 / mass_ratio
    f_ref = 20.0
    phic = 0.0
    tc = 0.0
    inclination = 0.0
    params = dict(
        m1=mass_1 * lal.MSUN_SI,
        m2=mass_2 * lal.MSUN_SI,
        S1x=0,
        S1y=0,
        S1z=chi1z,
        S2x=0,
        S2y=0,
        S2z=chi2z,
        distance=(distance * u.Mpc).to("m").value,
        inclination=inclination,
        phiRef=phic,
        longAscNodes=0.0,
        eccentricity=0.0,
        meanPerAno=0.0,
        deltaF=1.0 / sample_rate,
        f_min=10.0,
        f_ref=f_ref,
        f_max=300,
        approximant=lalsimulation.IMRPhenomPv2,
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
        [
            chirp_mass,
            mass_ratio,
            0,
            0,
            chi1z,
            0,
            0,
            chi2z,
            distance,
            tc,
            phic,
            inclination,
        ]
    ).repeat(10, 1)
    # repeat along batch dim for testing
    batched_chirp_mass = _params[:, 0]
    batched_mass_ratio = _params[:, 1]
    batched_chi1x = _params[:, 2]
    batched_chi1y = _params[:, 3]
    batched_chi1z = _params[:, 4]
    batched_chi2x = _params[:, 5]
    batched_chi2y = _params[:, 6]
    batched_chi2z = _params[:, 7]
    batched_distance = _params[:, 8]
    batched_tc = _params[:, 9]
    batched_phic = _params[:, 10]
    batched_inclination = _params[:, 11]
    hc_torch, hp_torch = waveforms.IMRPhenomPv2()(
        torch_freqs,
        batched_chirp_mass,
        batched_mass_ratio,
        batched_chi1x,
        batched_chi1y,
        batched_chi1z,
        batched_chi2x,
        batched_chi2y,
        batched_chi2z,
        batched_distance,
        batched_tc,
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

    assert np.allclose(
        1e21 * hp_lal_data.real, 1e21 * hp_torch.real.numpy(), atol=3e-3
    )
    assert np.allclose(
        1e21 * hp_lal_data.imag, 1e21 * hp_torch.imag.numpy(), atol=3e-3
    )
    assert np.allclose(
        1e21 * hc_lal_data.real, 1e21 * hc_torch.real.numpy(), atol=3e-3
    )
    assert np.allclose(
        1e21 * hc_lal_data.imag, 1e21 * hc_torch.imag.numpy(), atol=3e-3
    )
