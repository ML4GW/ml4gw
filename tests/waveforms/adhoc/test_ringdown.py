import lal
import lalsimulation
import numpy as np
import pytest
import torch

from ml4gw.waveforms import Ringdown


@pytest.mark.parametrize(
    ("sample_rate, duration, mass, epsilon, phase, inclination, distance"),
    [
        pytest.param(
            2048, 0.5, 50.0, 0.01, 0.0, 0.7, 100.0, id="short-low-mass"
        ),
        pytest.param(
            4096, 1.0, 100.0, 0.04, 0.3, 0.9, 200.0, id="long-high-mass"
        ),
    ],
)
def test_ringdown_matches_lalsimulation(
    sample_rate,
    duration,
    mass,
    epsilon,
    phase,
    inclination,
    distance,
):
    spins = [0.0, 0.5, 0.9]
    frequency, quality = zip(
        *[
            lalsimulation.SimBlackHoleRingdownMode(
                mass * lal.MTSUN_SI, spin, 2, 2, -2
            )
            for spin in spins
        ],
        strict=True,
    )
    parameters = [
        frequency,
        quality,
        [epsilon] * len(spins),
        [phase] * len(spins),
        [inclination] * len(spins),
        [distance] * len(spins),
    ]
    parameters = [torch.tensor(x, dtype=torch.float64) for x in parameters]

    ringdown = Ringdown(sample_rate, duration)
    cross, plus = ringdown(*parameters)

    # SimBlackHoleRingdown uses an azimuthal phase for the m=2 mode,
    # whereas Ringdown accepts the phase of the damped sinusoid directly.
    lal_phase = (np.pi - phase) / 2
    distance *= 1e6 * lal.PC_SI

    for i, spin in enumerate(spins):
        hplus, hcross = lalsimulation.SimBlackHoleRingdown(
            lal.LIGOTimeGPS(0),
            lal_phase,
            1 / sample_rate,
            mass * lal.MSUN_SI,
            spin,
            epsilon,
            distance,
            inclination,
            2,
            2,
        )
        expected_plus = np.asarray(hplus.data.data)
        expected_cross = np.asarray(hcross.data.data)
        n_samples = len(expected_plus)

        # Ringdown uses closed-form fits for the mass, spin, and angular
        # dependence rather than LALSuite's numerical Kerr solution. The
        # approximation stays within 7% of LALSuite over this parameter range.
        for actual, expected in (
            (plus[i, :n_samples].numpy(), expected_plus),
            (cross[i, :n_samples].numpy(), expected_cross),
        ):
            difference = np.linalg.norm(actual - expected)
            relative_error = difference / np.linalg.norm(expected)
            assert relative_error < 0.07
