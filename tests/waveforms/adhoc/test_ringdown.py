import lal
import lalsimulation
import numpy as np
import pytest
import torch

from ml4gw.waveforms import Ringdown


def get_lal_waveform_without_angular_response(
    sample_rate, mass, spin, epsilon, phase, distance
):
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
    angular_component = np.conj(
        lalsimulation.SimBlackHoleRingdownSpheroidalWaveFunction(
            0, spin, 2, 2, -2
        )
    )
    # LAL defines hcross = -Im(h), so hplus + 1j * hcross contains the
    # conjugate angular response. Remove only that response; the physical
    # strain amplitude and time evolution remain unchanged.
    return waveform / angular_component


@pytest.mark.parametrize("spin", [0.0, 0.5, 0.9, 0.99])
def test_ringdown_matches_lal_scaling_after_angular_correction(spin):
    configurations = [
        (2048, 0.5, 50.0, 0.01, 0.0, 100.0),
        (4096, 1.0, 100.0, 0.04, 0.3, 200.0),
    ]
    inclinations = np.linspace(0, np.pi, 7)
    scale_factors = []

    for (
        sample_rate,
        duration,
        mass,
        epsilon,
        phase,
        distance,
    ) in configurations:
        frequency, quality = lalsimulation.SimBlackHoleRingdownMode(
            mass * lal.MTSUN_SI, spin, 2, 2, -2
        )
        parameters = [
            [frequency] * len(inclinations),
            [quality] * len(inclinations),
            [epsilon] * len(inclinations),
            [phase] * len(inclinations),
            inclinations,
            [distance] * len(inclinations),
        ]
        parameters = [torch.tensor(x, dtype=torch.float64) for x in parameters]

        ringdown = Ringdown(sample_rate, duration)
        cross, plus = ringdown(*parameters)
        intrinsic = get_lal_waveform_without_angular_response(
            sample_rate, mass, spin, epsilon, phase, distance
        )
        n_samples = min(len(intrinsic), plus.shape[1])
        intrinsic = intrinsic[:n_samples]
        intrinsic_norm = np.linalg.norm(intrinsic)

        for i, inclination in enumerate(inclinations):
            cos_inclination = np.cos(inclination)
            comparisons = (
                (
                    "plus",
                    plus[i, :n_samples].numpy(),
                    1 + cos_inclination**2,
                    intrinsic.real,
                ),
                (
                    "cross",
                    cross[i, :n_samples].numpy(),
                    2 * cos_inclination,
                    intrinsic.imag,
                ),
            )

            for polarization, actual, angular_factor, waveform in comparisons:
                if np.isclose(angular_factor, 0, atol=1e-15):
                    np.testing.assert_allclose(
                        actual / intrinsic_norm,
                        0,
                        rtol=0,
                        atol=1e-12,
                        err_msg=(
                            f"{polarization} polarization is nonzero for "
                            f"spin={spin} and inclination={inclination}"
                        ),
                    )
                    continue

                expected = angular_factor * waveform
                scale = (
                    np.vdot(expected, actual).real
                    / np.vdot(expected, expected).real
                )
                assert scale > 0
                residual = np.linalg.norm(actual - scale * expected)
                residual /= np.linalg.norm(expected)
                assert residual < 1e-6, (
                    f"{polarization} polarization shape differs for "
                    f"spin={spin} and inclination={inclination}"
                )
                scale_factors.append(scale)

    # The closed-form amplitude has a spin-dependent normalization relative to
    # LALSuite, but its scaling must be independent of all other parameters,
    # inclination, and polarization.
    np.testing.assert_allclose(
        scale_factors,
        scale_factors[0],
        rtol=1e-6,
        atol=0,
    )
