import lal
import lalsimulation
import numpy as np
import pytest
import torch

from ml4gw.waveforms import Ringdown

RINGDOWN_TEST_CASES = [
    pytest.param(2048, 0.5, 50.0, 0.01, 0.0, 100.0, id="short-low-mass"),
    pytest.param(4096, 1.0, 100.0, 0.04, 0.3, 200.0, id="long-high-mass"),
]


def get_lal_waveform(
    sample_rate, mass, spin, epsilon, phase, distance, inclination
):
    # LAL uses an azimuthal phase for its (2, 2) mode, while Ringdown accepts
    # the phase of the damped sinusoid directly, so this maps conventions.
    lal_phase = (np.pi - phase) / 2
    hplus, hcross = lalsimulation.SimBlackHoleRingdown(
        lal.LIGOTimeGPS(0),
        lal_phase,
        1 / sample_rate,
        mass * lal.MSUN_SI,
        spin,
        epsilon,
        distance * 1e6 * lal.PC_SI,
        inclination,
        2,
        2,
    )
    return np.asarray(hplus.data.data), np.asarray(hcross.data.data)


def get_lal_waveform_without_angular_response(
    sample_rate, mass, spin, epsilon, phase, distance
):
    # At zero inclination only the positive-m component remains.
    hplus, hcross = get_lal_waveform(
        sample_rate, mass, spin, epsilon, phase, distance, 0
    )
    waveform = hplus + 1j * hcross
    angular_component = np.conj(
        lalsimulation.SimBlackHoleRingdownSpheroidalWaveFunction(
            0, spin, 2, 2, -2
        )
    )
    # LAL defines hcross = -Im(h), so hplus + 1j * hcross contains the
    # conjugate angular response. Remove only that response; the physical
    # strain amplitude and time evolution remain unchanged.
    return waveform / angular_component


def get_scale_and_residual(actual, expected):
    scale = np.vdot(expected, actual).real / np.vdot(expected, expected).real
    actual_norm = np.linalg.norm(actual)
    if actual_norm == 0:
        return scale, np.inf
    residual = np.linalg.norm(actual - scale * expected)
    residual /= actual_norm
    return scale, residual


@pytest.mark.parametrize(
    "sample_rate,duration,mass,epsilon,phase,distance",
    RINGDOWN_TEST_CASES,
)
@pytest.mark.parametrize("spin", [0.0, 0.5, 0.9, 0.99])
def test_ringdown_matches_lal_scaling_after_angular_correction(
    spin, sample_rate, duration, mass, epsilon, phase, distance
):
    inclinations = np.linspace(0, np.pi, 7)
    scale_factors = []

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
            scale, residual = get_scale_and_residual(actual, expected)
            assert scale > 0
            assert residual < 1e-6, (
                f"{polarization} polarization shape differs for spin={spin} "
                f"and inclination={inclination}"
            )
            scale_factors.append(scale)

    # The closed-form amplitude has a spin-dependent normalization relative to
    # LALSuite, but its scaling must be independent of inclination and
    # polarization for each parameter set.
    np.testing.assert_allclose(
        scale_factors,
        scale_factors[0],
        rtol=1e-6,
        atol=0,
    )


@pytest.mark.parametrize(
    "sample_rate,duration,mass,epsilon,phase,distance",
    RINGDOWN_TEST_CASES,
)
def test_ringdown_matches_unmodified_lal_at_zero_spin(
    sample_rate, duration, mass, epsilon, phase, distance
):
    spin = 0
    inclination = 0
    frequency, quality = lalsimulation.SimBlackHoleRingdownMode(
        mass * lal.MTSUN_SI, spin, 2, 2, -2
    )
    parameters = [frequency, quality, epsilon, phase, inclination, distance]
    parameters = [torch.tensor([x], dtype=torch.float64) for x in parameters]
    cross, plus = Ringdown(sample_rate, duration)(*parameters)
    lal_plus, lal_cross = get_lal_waveform(
        sample_rate,
        mass,
        spin,
        epsilon,
        phase,
        distance,
        inclination,
    )
    n_samples = min(len(lal_plus), plus.shape[1])
    scale_factors = []
    for actual, expected in (
        (plus[0, :n_samples].numpy(), lal_plus[:n_samples]),
        (cross[0, :n_samples].numpy(), lal_cross[:n_samples]),
    ):
        scale, residual = get_scale_and_residual(actual, expected)
        assert residual < 1e-6
        scale_factors.append(scale)

    # At zero spin the spherical and spheroidal angular functions coincide.
    # LAL's numerical mode gives a stable scale of 1.0173 for these cases, so a
    # 2% bound covers that difference while rejecting the old amplitude law.
    np.testing.assert_allclose(scale_factors, 1, rtol=0.02, atol=0)
