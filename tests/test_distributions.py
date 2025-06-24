import math

import numpy as np
import pytest
import torch
from astropy.cosmology import Planck18
from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame
from bilby.core.utils.random import seed as bilby_seed
from scipy import optimize, stats

from ml4gw import distributions

# TODO: for all tests, how to validate that
# distribution has the expected shape?


def test_log_uniform(seed_everything):
    sampler = distributions.LogUniform(math.e, math.e**2)
    samples = sampler.sample((10,))
    assert len(samples) == 10
    assert ((torch.e <= samples) & (torch.e**2 <= 100)).all()

    # check that the mean is roughly correct
    # (within three standard deviations)
    samples = sampler.sample((100000,))
    log_samples = np.log(samples)

    mean = log_samples.mean().item()
    variance = 4 / 12
    sample_variance = variance / 10000
    sample_std = sample_variance**0.5
    assert abs(mean - 1.5) < (3 * sample_std)


def test_cosine(seed_everything):
    sampler = distributions.Cosine()
    samples = sampler.sample((10,))
    assert len(samples) == 10
    assert ((-math.pi / 2 <= samples) & (samples <= math.pi / 2)).all()

    sampler = distributions.Cosine(-3, 5)
    samples = sampler.sample((100,))
    assert len(samples) == 100
    assert ((-3 <= samples) & (samples <= 5)).all()

    assert torch.all(sampler.log_prob(torch.tensor([-4, 6])) == float("-inf"))


def test_power_law(seed_everything):
    """Test PowerLaw distribution"""
    ref_snr = 8
    sampler = distributions.PowerLaw(ref_snr, float("inf"), index=-4)
    samples = sampler.sample((100000,)).numpy()
    # check x^-4 behavior
    counts, ebins = np.histogram(samples, bins=1000)
    bins = ebins[1:] + ebins[:-1]
    bins *= 0.5

    def foo(x, a, b):
        return a * x**b

    popt, _ = optimize.curve_fit(foo, bins, counts)
    # popt[1] is the index
    assert popt[1] == pytest.approx(-4, rel=1e-1)

    min_dist = 10
    max_dist = 1000
    uniform_in_volume = distributions.PowerLaw(min_dist, max_dist, index=2)
    samples = uniform_in_volume.sample((100000,)).numpy()
    # check d^2 behavior
    counts, ebins = np.histogram(samples, bins=1000)
    bins = ebins[1:] + ebins[:-1]
    bins *= 0.5

    popt, _ = optimize.curve_fit(foo, bins, counts)
    # popt[1] is the index
    assert popt[1] == pytest.approx(2, rel=1e-1)

    # test 1/x distribution
    inverse_in_distance = distributions.PowerLaw(min_dist, max_dist, index=-1)
    samples = inverse_in_distance.sample((100000,)).numpy()
    counts, ebins = np.histogram(samples, bins=1000)
    bins = ebins[1:] + ebins[:-1]
    bins *= 0.5
    popt, _ = optimize.curve_fit(foo, bins, counts)

    assert popt[1] == pytest.approx(-1, rel=1e-1)


def test_delta_function(seed_everything):
    sampler = distributions.DeltaFunction(peak=20)
    samples = sampler.sample((10,))
    assert (samples == 20).all()


class TestCosmologyDistributions:
    # bilby randomness currently comes into play in
    # only this test, so set the seed separately from
    # the `seed_everything` fixture so that we don't
    # get too constrained in finding a seed that passes
    # every random test.
    bilby_seed(123456789)

    @pytest.fixture
    def num_samples(self):
        return 1000

    @pytest.fixture
    def num_trials(self):
        return 1000

    @pytest.fixture
    def alpha(self):
        return 0.05

    @pytest.fixture(params=[False])
    def source_frame_time(self, request):
        return request.param

    @pytest.fixture
    def boundaries(self):
        return {
            "redshift": (0, 5),
            "comoving_distance": (1000, 2000),
            "luminosity_distance": (10000, 45000),
        }

    def test_uniform_comoving_volume(
        self, seed_everything, boundaries, num_samples, num_trials, alpha
    ):
        alpha = 0.05
        for distance_type, (minimum, maximum) in boundaries.items():
            ml4gw_dist = distributions.UniformComovingVolume(
                minimum=minimum,
                maximum=maximum,
                distance_type=distance_type,
            )
            bilby_dist = UniformComovingVolume(
                minimum=minimum, maximum=maximum, name=distance_type
            )

            ml4gw_samples = ml4gw_dist.sample((num_samples,))
            assert ml4gw_samples.shape == (num_samples,)

            # Not sure that this is the ideal way to test this
            count = 0
            for _ in range(num_trials):
                ml4gw_dist.sample((num_samples,))
                bilby_samples = bilby_dist.sample(num_samples)
                _, p_value = stats.ks_2samp(
                    ml4gw_samples.numpy(), bilby_samples
                )
                if p_value < alpha:
                    count += 1

            mean = num_trials * alpha
            sigma = num_trials * alpha * (1 - alpha)
            assert abs(count - mean) < 3 * sigma

            # Compare log probability between ml4gw and bilby
            # using the same samples
            ml4gw_log_prob = ml4gw_dist.log_prob(ml4gw_samples)
            bilby_log_prob = bilby_dist.ln_prob(ml4gw_samples)
            assert np.allclose(
                ml4gw_log_prob.numpy(), bilby_log_prob, rtol=1e-2
            )

            # Check that the luminosity distance calculation
            # matches astropy's
            z_grid = ml4gw_dist.z_grid.numpy()
            # The d_L calculation differs by ~5% at z=0.015, and that
            # difference improves with increasing z.
            mask = z_grid > 0.015
            ml4gw_dl = ml4gw_dist.luminosity_dist_grid.numpy()[mask]
            astropy_dl = Planck18.luminosity_distance(z_grid[mask]).value
            assert np.allclose(ml4gw_dl, astropy_dl, rtol=5e-2)

    def test_rate_evolution(
        self, seed_everything, boundaries, num_samples, num_trials, alpha
    ):
        alpha = 0.05

        def rate_function(z):
            return 1 / (1 + z)

        for distance_type, (minimum, maximum) in boundaries.items():
            ml4gw_dist = distributions.RateEvolution(
                rate_function=rate_function,
                minimum=minimum,
                maximum=maximum,
                distance_type=distance_type,
            )
            bilby_dist = UniformSourceFrame(
                minimum=minimum, maximum=maximum, name=distance_type
            )

            ml4gw_samples = ml4gw_dist.sample((num_samples,))
            assert ml4gw_samples.shape == (num_samples,)

            # Not sure that this is the ideal way to test this
            count = 0
            for _ in range(num_trials):
                ml4gw_dist.sample((num_samples,))
                bilby_samples = bilby_dist.sample(num_samples)
                _, p_value = stats.ks_2samp(
                    ml4gw_samples.numpy(), bilby_samples
                )
                if p_value < alpha:
                    count += 1

            mean = num_trials * alpha
            sigma = num_trials * alpha * (1 - alpha)
            assert abs(count - mean) < 3 * sigma

            # Compare log probability between ml4gw and bilby
            # using the same samples
            ml4gw_log_prob = ml4gw_dist.log_prob(ml4gw_samples)
            bilby_log_prob = bilby_dist.ln_prob(ml4gw_samples)
            assert np.allclose(
                ml4gw_log_prob.numpy(), bilby_log_prob, rtol=1e-2
            )

    def test_raises_errors(self):
        with pytest.raises(ValueError, match=r"Distance type must be*"):
            distributions.UniformComovingVolume(
                minimum=0, maximum=5, distance_type="dummy"
            )

        with pytest.raises(ValueError, match=r"Maximum redshift*"):
            distributions.UniformComovingVolume(
                minimum=0, maximum=6, distance_type="redshift"
            )
