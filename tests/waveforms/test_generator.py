import astropy.units as u
import numpy as np
import pytest
import torch
from lalsimulation.gwsignal.core.waveform import (
    GenerateTDWaveform,
    LALCompactBinaryCoalescenceGenerator,
)
from scipy.signal import hilbert

from ml4gw.waveforms import IMRPhenomD, conversion
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator

TARGET_OVERLAP = 1 - 1e-6


@pytest.fixture(params=[2048])
def sample_rate(request):
    return request.param


# name, chirp_mass, mass_ratio, chi1, chi2, distance, phase, theta_jn, duration
ALIGNED_TEST_CASES = [
    ("equal_mass_zero_spin", 26.12, 0.99, 0.0, 0.0, 100.0, 0.0, 0.0, 10),
    ("low_q_limit", 15.0, 1 / 20.0, 0.0, 0.0, 100.0, 0.0, 0.4, 10),
    ("max_spin_aligned", 30.0, 0.8, 0.99, 0.99, 150.0, 0.8, 0.5, 10),
    ("max_spin_anti_aligned", 30.0, 0.8, -0.99, -0.99, 150.0, 0.8, 0.5, 10),
    ("low_mass_bns", 1.21, 0.9, 0.05, 0.05, 40.0, 0.0, 0.0, 200),
    ("high_mass_bbh", 70.0, 0.9, 0.5, 0.5, 500.0, 1.5, 0.8, 10),
]


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


def compute_match(h1_td, h2_td, sample_rate, f_min):
    """
    Overlap between time-domain waveforms, maximized over time shifts.
    Note that the waveforms have been interpolated to the same time
    grid in the testing function below.
    """
    h1_fd = torch.fft.rfft(h1_td)
    h2_fd = torch.fft.rfft(h2_td)

    freqs = torch.fft.rfftfreq(len(h1_td), d=1.0 / sample_rate)
    freq_mask = freqs >= f_min

    h1_fd = h1_fd * freq_mask
    h2_fd = h2_fd * freq_mask

    cross_corr = torch.fft.irfft(h1_fd * torch.conj(h2_fd))
    norm1 = torch.sum(torch.fft.irfft(h1_fd) ** 2)
    norm2 = torch.sum(torch.fft.irfft(h2_fd) ** 2)
    return cross_corr.abs().max() / torch.sqrt(norm1 * norm2)


@pytest.mark.parametrize(
    "name,chirp_mass,mass_ratio,chi1,chi2,distance,phase,theta_jn,duration",
    ALIGNED_TEST_CASES,
)
def test_cbc_waveform_generator(
    name,
    chirp_mass,
    mass_ratio,
    chi1,
    chi2,
    distance,
    phase,
    theta_jn,
    duration,
    sample_rate,
):
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

    chirp_mass = torch.tensor([chirp_mass], dtype=torch.float64)
    mass_ratio = torch.tensor([mass_ratio], dtype=torch.float64)
    chi1 = torch.tensor([chi1], dtype=torch.float64)
    chi2 = torch.tensor([chi2], dtype=torch.float64)
    distance = torch.tensor([distance], dtype=torch.float64)
    phase = torch.tensor([phase], dtype=torch.float64)
    theta_jn = torch.tensor([theta_jn], dtype=torch.float64)

    mass_1, mass_2 = conversion.chirp_mass_and_mass_ratio_to_components(
        chirp_mass, mass_ratio
    )
    ml4gw_parameters = {
        "chirp_mass": chirp_mass,
        "mass_ratio": mass_ratio,
        "mass_1": mass_1,
        "mass_2": mass_2,
        "chi1": chi1,
        "chi2": chi2,
        "s1z": chi1,
        "s2z": chi2,
        "s1x": torch.zeros_like(chi1),
        "s1y": torch.zeros_like(chi1),
        "s2x": torch.zeros_like(chi2),
        "s2y": torch.zeros_like(chi2),
        "phic": phase,
        "distance": distance,
        "inclination": theta_jn,
    }
    hc_ml4gw, hp_ml4gw = generator(**ml4gw_parameters)

    gwsignal_params = {
        "mass1": mass_1.item() * u.solMass,
        "mass2": mass_2.item() * u.solMass,
        "deltaT": (1 / sample_rate) * u.s,
        "f22_start": f_min * u.Hz,
        "f22_ref": f_ref * u.Hz,
        "phi_ref": phase.item() * u.rad,
        "distance": distance.item() * u.Mpc,
        "inclination": theta_jn.item() * u.rad,
        "eccentricity": 0.0 * u.dimensionless_unscaled,
        "longAscNodes": 0.0 * u.rad,
        "meanPerAno": 0.0 * u.rad,
        "spin1x": 0.0 * u.dimensionless_unscaled,
        "spin1y": 0.0 * u.dimensionless_unscaled,
        "spin1z": chi1.item() * u.dimensionless_unscaled,
        "spin2x": 0.0 * u.dimensionless_unscaled,
        "spin2y": 0.0 * u.dimensionless_unscaled,
        "spin2z": chi2.item() * u.dimensionless_unscaled,
        "condition": 1,
    }

    gwsignal_generator = GWsignalGenerator("IMRPhenomD")
    hp_gwsignal, _ = GenerateTDWaveform(gwsignal_params, gwsignal_generator)

    # Interpolate gwsignal's hp onto ml4gw's time grid.
    ml4gw_times = np.arange(generator.size) / sample_rate - (
        duration - right_pad
    )
    hp_gws_td = np.interp(
        ml4gw_times,
        hp_gwsignal.times.value,
        hp_gwsignal.value,
        left=0.0,
        right=0.0,
    )

    # For reasons I can't track down, gwsignal's hc is not exactly the
    # scaled Hilbert transform of hp, so instead I just compute the
    # overlap of our hc with the Hilbert transform of gwsignal's hp.
    cfac = np.cos(theta_jn.item())
    pfac = 0.5 * (1.0 + cfac * cfac)
    hc_gws_td = (cfac / pfac) * np.imag(hilbert(hp_gws_td))

    assert (
        compute_match(hp_ml4gw[0], torch.tensor(hp_gws_td), sample_rate, f_min)
        > TARGET_OVERLAP
    )
    assert (
        compute_match(hc_ml4gw[0], torch.tensor(hc_gws_td), sample_rate, f_min)
        > TARGET_OVERLAP
    )
