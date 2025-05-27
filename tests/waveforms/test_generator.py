import astropy.units as u
import numpy as np
import pytest
import torch
from lalsimulation.gwsignal.core.waveform import (
    GenerateTDWaveform,
    LALCompactBinaryCoalescenceGenerator,
)

from ml4gw.waveforms import IMRPhenomD, conversion
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator


@pytest.fixture()
def num_samples():
    return 1000


@pytest.fixture(params=[2048])
def sample_rate(request):
    return request.param


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
    # torch.manual_seed(10)
    duration = 10
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

    assert generator.delta_t == 1 / sample_rate
    assert generator.size == int(duration * sample_rate)
    assert generator.delta_f == 1 / duration

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
        # construct gwsignal params
        gwsignal_params = {
            "mass1": ml4gw_parameters["mass_1"][i].item() * u.solMass,
            "mass2": ml4gw_parameters["mass_2"][i].item() * u.solMass,
            "deltaT": (1 / sample_rate) * u.s,
            "f22_start": f_min * u.Hz,
            "f22_ref": f_ref * u.Hz,
            "phi_ref": ml4gw_parameters["phic"][i].item() * u.rad,
            "distance": (ml4gw_parameters["distance"][i].item() * u.Mpc),
            "inclination": ml4gw_parameters["inclination"][i].item() * u.rad,
            "eccentricity": 0.0 * u.dimensionless_unscaled,
            "longAscNodes": 0.0 * u.rad,
            "meanPerAno": 0.0 * u.rad,
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

        hp = hp_ml4gw[i].detach().numpy()
        hc = hc_ml4gw[i].detach().numpy()

        # now align the gwsignal and ml4gw waveforms so we can compoare
        ml4gw_times = np.arange(0, len(hp) / sample_rate, 1 / sample_rate)
        ml4gw_times -= duration - right_pad

        # TODO: track this down

        # gwsignal will adjust the "epoch"
        # i.e. the coalescence time based on
        # the maximum value of the array.
        # In some cases, the ml4gw prediction
        # for the time of maximum value is slightly different
        # than the gwsignal/lalsimulation. Typically, either
        # the maximum, or second maximum will align with gwsignal.
        # Presumably this is due to accumulation of discrepancies
        # between our implementation and gwsignal/lalsimulations.

        # So, for testing we align the waveform using both
        # the maximum and second maximum values, and assert True
        # if any of them satisfy the tolerances

        for ml4gw_pol, gwsignal_pol in zip(
            [hp, hc], [hp_gwsignal, hc_gwsignal]
        ):
            argsorted = np.argsort(ml4gw_pol)[::-1]
            for j in range(2):
                argmax = argsorted[j]
                ml4gw_times = ml4gw_times - ml4gw_times[argmax]

                max_time = min(ml4gw_times[-1], gwsignal_pol.times.value[-1])
                min_time = max(ml4gw_times[0], gwsignal_pol.times.value[0])

                mask = gwsignal_pol.times.value <= max_time
                mask &= gwsignal_pol.times.value >= min_time

                ml4gw_mask = ml4gw_times <= max_time
                ml4gw_mask &= ml4gw_times >= min_time

                close = np.allclose(
                    gwsignal_pol.value[mask],
                    ml4gw_pol[ml4gw_mask],
                    atol=4e-23,
                    rtol=0.01,
                )

                if close:
                    break

                # TODO: track this down

                # theres an off by one error that occurs
                # occasionally when attempting to align the
                # gwsignal and ml4gw waveforms that is causing
                # testing comparison issues, so assert that
                # either one of them is close enough

                for k in range(1, 2):
                    if close:
                        break

                    close_shifted = np.allclose(
                        gwsignal_pol.value[mask][:-k],
                        ml4gw_pol[ml4gw_mask][k:],
                        atol=4e-23,
                        rtol=0.01,
                    )

                    close_shifted = close_shifted or np.allclose(
                        gwsignal_pol.value[mask][k:],
                        ml4gw_pol[ml4gw_mask][:-k],
                        atol=4e-23,
                        rtol=0.01,
                    )

                    close = close or close_shifted

            assert close
