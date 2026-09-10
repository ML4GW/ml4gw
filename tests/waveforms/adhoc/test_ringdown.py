import lal
import lalsimulation
import numpy as np
import pytest
import torch

from ml4gw.waveforms import Ringdown


def get_expected_amplitude(frequency, quality, epsilon, distance):
    spin = 1 - (2 / quality) ** (20 / 9)
    mass = (
        (1 / (2 * np.pi))
        * (lal.C_SI**3 / (lal.G_SI * frequency))
        * (1 - 0.63 * (2 / quality) ** (2 / 3))
    )
    f_quality = 1 + (7 / 24) / quality**2
    g_spin = 1 - 0.63 * (1 - spin) ** (3 / 10)
    amplitude = (
        np.sqrt(5 * epsilon / 2)
        * (lal.G_SI * mass / lal.C_SI**2)
        * quality ** (-0.5)
        * f_quality ** (-0.5)
        * g_spin ** (-0.5)
    )
    return amplitude / (distance * 1e6 * lal.PC_SI)


def get_lal_time_evolution(sample_rate, mass, spin, epsilon, phase, distance):
    # At zero inclination only the positive-m component remains. LAL uses an
    # azimuthal phase for its (2, 2) mode, while Ringdown accepts the phase of
    # the damped sinusoid directly, so this maps between the two conventions.
    lal_phase = (np.pi - phase) / 2
    hplus, hcross = lalsimulation.SimBlackHoleRingdown(
        lal.LIGOTimeGPS(0),
        lal_phase,
        1 / sample_rate,
        mass * lal.MSUN_SI,
        spin,
        epsilon,
        distance * 1e6 * lal.PC_SI,
        0,
        2,
        2,
    )
    waveform = np.asarray(hplus.data.data) + 1j * np.asarray(hcross.data.data)
    # Dividing by the initial complex value removes LALSuite's numerical
    # amplitude and spheroidal-harmonic phase while retaining its evolution.
    return waveform / waveform[0] * np.exp(1j * phase)


@pytest.mark.parametrize(
    ("sample_rate, duration, mass, epsilon, phase, distance"),
    [
        pytest.param(2048, 0.5, 50.0, 0.01, 0.0, 100.0, id="short-low-mass"),
        pytest.param(4096, 1.0, 100.0, 0.04, 0.3, 200.0, id="long-high-mass"),
    ],
)
def test_ringdown_matches_lal_evolution_with_closed_form_factors(
    sample_rate,
    duration,
    mass,
    epsilon,
    phase,
    distance,
):
    spins = np.array([0.0, 0.5, 0.9, 0.99])
    inclinations = np.linspace(0, np.pi, 7)
    spins, inclinations = np.meshgrid(spins, inclinations, indexing="ij")
    spins = spins.ravel()
    inclinations = inclinations.ravel()

    modes = [
        lalsimulation.SimBlackHoleRingdownMode(
            mass * lal.MTSUN_SI, spin, 2, 2, -2
        )
        for spin in spins
    ]
    frequency, quality = zip(*modes, strict=True)
    parameters = [
        frequency,
        quality,
        [epsilon] * len(spins),
        [phase] * len(spins),
        inclinations,
        [distance] * len(spins),
    ]
    parameters = [torch.tensor(x, dtype=torch.float64) for x in parameters]

    ringdown = Ringdown(sample_rate, duration)
    cross, plus = ringdown(*parameters)

    for i, (spin, inclination) in enumerate(
        zip(spins, inclinations, strict=True)
    ):
        evolution = get_lal_time_evolution(
            sample_rate, mass, spin, epsilon, phase, distance
        )
        amplitude = get_expected_amplitude(
            frequency[i], quality[i], epsilon, distance
        )
        cos_inclination = np.cos(inclination)
        expected_plus = amplitude * (1 + cos_inclination**2) * evolution.real
        expected_cross = amplitude * (2 * cos_inclination) * evolution.imag
        n_samples = len(evolution)

        # Normalize away the physical strain scale so that the absolute
        # tolerance only covers numerical differences in the dimensionless
        # time evolution and closed-form angular factors.
        for polarization, actual, expected in (
            ("plus", plus[i, :n_samples].numpy(), expected_plus),
            ("cross", cross[i, :n_samples].numpy(), expected_cross),
        ):
            np.testing.assert_allclose(
                actual / amplitude,
                expected / amplitude,
                rtol=0,
                atol=1e-6,
                err_msg=(
                    f"{polarization} polarization differs for spin={spin} "
                    f"and inclination={inclination}"
                ),
            )
