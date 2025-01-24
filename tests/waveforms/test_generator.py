import astropy.units as u
import lal
import lalsimulation
import pytest
import torch

from ml4gw.waveforms import IMRPhenomD, conversion
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator


@pytest.fixture(params=[10, 100, 1000])
def n_samples(request):
    return request.param


@pytest.fixture(params=[1, 2, 10])
def duration(request):
    return request.param


@pytest.fixture(params=[1024, 2048, 4096])
def sample_rate(request):
    return request.param


def test_cbc_waveform_generator(
    chirp_mass,
    mass_ratio,
    chi1,
    chi2,
    phase,
    distance,
    theta_jn,
    sample_rate,
):
    sample_rate = 4096
    duration = 1
    f_min = 20
    f_ref = 40
    right_pad = 0.1

    generator = TimeDomainCBCWaveformGenerator(
        approximant=IMRPhenomD(),
        sample_rate=sample_rate,
        duration=duration,
        f_min=f_min,
        f_ref=f_ref,
        right_pad=right_pad,
    )

    mass_1, mass_2 = conversion.chirp_mass_and_mass_ratio_to_components(
        chirp_mass, mass_ratio
    )
    s1x = torch.zeros_like(chi1)
    s1y = torch.zeros_like(chi1)
    s1z = chi1
    s2x = torch.zeros_like(chi2)
    s2y = torch.zeros_like(chi2)
    s2z = chi2
    parameters = {
        "chirp_mass": chirp_mass,
        "mass_ratio": mass_ratio,
        "mass_1": mass_1,
        "mass_2": mass_2,
        "chi1": chi1,
        "chi2": chi2,
        "s1z": s1z,
        "s2z": s2z,
        "s1x": s1x,
        "s1y": s1y,
        "s2x": s2x,
        "s2y": s2y,
        "phic": phase,
        "distance": distance,
        "inclination": theta_jn,
    }
    hc, hp = generator(**parameters)

    # now compare each waveform with lalsimulation SimInspiralTD
    for i in range(len(chirp_mass)):

        # test far (> 400 Mpc) waveforms (O(1e-3) agreement)

        # construct lalinference params
        params = dict(
            m1=mass_1[i].item() * lal.MSUN_SI,
            m2=mass_2[i].item() * lal.MSUN_SI,
            S1x=s1x[i].item(),
            S1y=s2y[i].item(),
            S1z=s1z[i].item(),
            S2x=s2x[i].item(),
            S2y=s2y[i].item(),
            S2z=s1z[i].item(),
            distance=(distance[i].item() * u.Mpc).to("m").value,
            inclination=theta_jn[i].item(),
            phiRef=phase[i].item(),
            longAscNodes=0.0,
            eccentricity=0.0,
            meanPerAno=0.0,
            deltaT=1 / sample_rate,
            f_min=f_min,
            f_ref=f_ref,
            approximant=lalsimulation.IMRPhenomD,
            LALparams=lal.CreateDict(),
        )
        return params
        # hp_lal, hc_lal = lalsimulation.SimInspiralTD(**params)
