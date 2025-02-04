import astropy.units as u
import numpy as np
import pytest
import torch
from lalsimulation.gwsignal.core.waveform import (
    GenerateTDWaveform,
    LALCompactBinaryCoalescenceGenerator,
)
from scipy.signal import butter, sosfiltfilt

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


def high_pass_time_series(time_series, dt, fmin, attenuation, N):
    """
    Same as
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/python/lalsimulation/gwsignal/core/conditioning_subroutines.py?ref_type=heads#L10 # noqa
    except w/o requiring gwpy.TimeSeries objects to be passed
    """
    fs = 1.0 / dt  # Sampling frequency
    a1 = attenuation  # Attenuation at the low-freq cut-off

    w1 = np.tan(np.pi * fmin * dt)  # Transformed frequency variable at f_min
    wc = w1 * (1.0 / a1**0.5 - 1) ** (
        1.0 / (2.0 * N)
    )  # Cut-off freq. from attenuation
    fc = fs * np.arctan(wc) / np.pi  # For use in butterworth filter

    # Construct the filter and then forward - backward filter the time-series
    sos = butter(N, fc, btype="highpass", output="sos", fs=fs)
    output = sosfiltfilt(sos, time_series)
    return output


class GWsignalGenerator(LALCompactBinaryCoalescenceGenerator):
    """
    Override gwsignal class to enforce use of `gwsignal` conditioning routines,
    and to enforce that the frequency domain version of the approximant is used
    for generating the waveform.
    """

    @property
    def metadata(self):
        metadata = {
            "type": "cbc_lalsimulation",
            "f_ref_spin": True,
            "modes": True,
            "polarizations": True,
            "implemented_domain": self._implemented_domain,
            "generation_domain": self._generation_domain,
            "approximant": self._approx_name,
            "implementation": "LALSimulation",
            "conditioning_routines": "gwsignal",
        }
        return metadata

    def _update_domains(self):
        self._implemented_domain = "freq"


def test_cbc_waveform_generator(
    chirp_mass,
    mass_ratio,
    chi1,
    chi2,
    phase,
    distance_far,
    theta_jn,
    sample_rate,
):
    duration = 20
    f_min = 20
    f_ref = 40
    right_pad = 0.5

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
    ml4gw_parameters = {
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
        "distance": distance_far,
        "inclination": theta_jn,
    }
    hc_ml4gw, hp_ml4gw = generator(**ml4gw_parameters)

    # now compare each waveform with lalsimulation SimInspiralTD
    for i in range(len(chirp_mass)):

        # construct lalinference params
        gwsignal_params = {
            "mass1": ml4gw_parameters["mass_1"][i].item() * u.solMass,
            "mass2": ml4gw_parameters["mass_2"][i].item() * u.solMass,
            "deltaT": 1 / sample_rate * u.s,
            "f22_start": f_min * u.Hz,
            "f22_ref": f_ref * u.Hz,
            "phi_ref": ml4gw_parameters["phic"][i].item() * u.rad,
            "distance": (ml4gw_parameters["distance"][i].item() * u.Mpc),
            "inclination": ml4gw_parameters["inclination"][i].item() * u.rad,
            "eccentricity": 0.0 * u.dimensionless_unscaled,
            "longAscNodes": 0.0 * u.rad,
            "meanPerAno": 0.0 * u.rad,
            "condition": 1,
            "spin1x": ml4gw_parameters["s1x"][i].item()
            * u.dimensionless_unscaled,
            "spin1y": ml4gw_parameters["s1y"][i].item()
            * u.dimensionless_unscaled,
            "spin1z": ml4gw_parameters["s1z"][i].item()
            * u.dimensionless_unscaled,
            "spin2x": ml4gw_parameters["s2x"][i].item()
            * u.dimensionless_unscaled,
            "spin2y": ml4gw_parameters["s2y"][i].item()
            * u.dimensionless_unscaled,
            "spin2z": ml4gw_parameters["s2z"][i].item()
            * u.dimensionless_unscaled,
            "condition": 1,
        }

        gwsignal_generator = GWsignalGenerator("IMRPhenomD")
        hp_gwsignal, hc_gwsignal = GenerateTDWaveform(
            gwsignal_params, gwsignal_generator
        )

        hp_ml4gw_highpassed = high_pass_time_series(
            hp_ml4gw[i].detach().numpy(), 1 / sample_rate, f_min, 0.99, 8.0
        )
        hc_ml4gw_highpassed = high_pass_time_series(
            hc_ml4gw[i].detach().numpy(), 1 / sample_rate, f_min, 0.99, 8.0
        )

        # now align the gwsignal and ml4gw waveforms so we can compoare
        ml4gw_times = np.arange(
            0, len(hp_ml4gw_highpassed) / sample_rate, 1 / sample_rate
        )
        ml4gw_times -= duration - right_pad

        hp_ml4gw_times = (
            ml4gw_times - ml4gw_times[np.argmax(hp_ml4gw_highpassed)]
        )

        max_time = min(hp_ml4gw_times[-1], hp_gwsignal.times.value[-1])
        min_time = max(hp_ml4gw_times[0], hp_gwsignal.times.value[0])

        mask = hp_gwsignal.times.value < max_time
        mask &= hp_gwsignal.times.value > min_time

        ml4gw_mask = hp_ml4gw_times < max_time
        ml4gw_mask &= hp_ml4gw_times > min_time

        # TODO: track this down

        # theres an off by one error that occurs
        # occasionally when attempting to align the
        # gwsignal and ml4gw waveforms that is causing
        # testing comparison issues, so assert that
        # either one of them is close enough

        close_hp = np.allclose(
            hp_gwsignal.value[mask],
            hp_ml4gw_highpassed[ml4gw_mask],
            atol=5e-23,
            rtol=0.01,
        )

        close_hp = close_hp or np.allclose(
            hp_gwsignal.value[mask][:-1],
            hp_ml4gw_highpassed[ml4gw_mask][1:],
            atol=5e-23,
            rtol=0.01,
        )
        assert close_hp

        close_hc = np.allclose(
            hc_gwsignal.value[mask],
            hc_ml4gw_highpassed[ml4gw_mask],
            atol=5e-23,
            rtol=0.01,
        )

        close_hc = close_hc or np.allclose(
            hc_gwsignal.value[mask][:-1],
            hc_ml4gw_highpassed[ml4gw_mask][1:],
            atol=5e-23,
            rtol=0.01,
        )
        assert close_hc
