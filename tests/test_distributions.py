import math

import numpy as np
import pytest
import torch
from astropy.cosmology import Planck18
from bilby.gw.prior import UniformComovingVolume
from scipy import optimize, stats

from ml4gw import distributions

# TODO: for all tests, how to validate that
# distribution has the expected shape?


def test_log_uniform():
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


def test_cosine():
    sampler = distributions.Cosine()
    samples = sampler.sample((10,))
    assert len(samples) == 10
    assert ((-math.pi / 2 <= samples) & (samples <= math.pi / 2)).all()

    sampler = distributions.Cosine(-3, 5)
    samples = sampler.sample((100,))
    assert len(samples) == 100
    assert ((-3 <= samples) & (samples <= 5)).all()

    assert torch.all(sampler.log_prob(torch.tensor([-4, 6])) == float("-inf"))


def test_power_law():
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


def test_delta_function():
    sampler = distributions.DeltaFunction(peak=20)
    samples = sampler.sample((10,))
    assert (samples == 20).all()


def test_uniform_comoving_volume():
    # Check that the ml4gw UCV distribution for
    # redshift matches bilby's
    minimum = 0
    maximum = 5
    bilby_dist = UniformComovingVolume(
        minimum=minimum, maximum=maximum, name="redshift"
    )
    ml4gw_dist = distributions.UniformComovingVolume(
        minimum=minimum, maximum=maximum, distance_type="redshift"
    )
    bilby_samples = bilby_dist.sample(100000)
    ml4gw_samples = ml4gw_dist.sample((100000,))
    _, p_value = stats.ks_2samp(ml4gw_samples.numpy(), bilby_samples)
    assert p_value > 0.05

    # Compare log probability between ml4gw and bilby
    # using the same samples
    ml4gw_log_prob = ml4gw_dist.log_prob(ml4gw_samples)
    bilby_log_prob = bilby_dist.ln_prob(ml4gw_samples)
    assert np.allclose(ml4gw_log_prob.numpy(), bilby_log_prob, rtol=1e-2)

    # Check that the luminosity distance calculation
    # matches astropy's
    z_grid = ml4gw_dist.z_grid.numpy()
    # The d_L calculation differs by ~5% at z=0.015, and that
    # difference improves with increasing z.
    mask = z_grid > 0.015
    ml4gw_dl = ml4gw_dist.luminosity_dist_grid.numpy()[mask]
    astropy_dl = Planck18.luminosity_distance(z_grid[mask]).value
    assert np.allclose(ml4gw_dl, astropy_dl, rtol=5e-2)

    # Repeat for comoving distance

    minimum = 1000
    maximum = 2000
    bilby_dist = UniformComovingVolume(
        minimum=minimum, maximum=maximum, name="comoving_distance"
    )
    ml4gw_dist = distributions.UniformComovingVolume(
        minimum=minimum, maximum=maximum, distance_type="comoving_distance"
    )
    bilby_samples = bilby_dist.sample(100000)
    ml4gw_samples = ml4gw_dist.sample((100000,))
    _, p_value = stats.ks_2samp(ml4gw_samples.numpy(), bilby_samples)
    assert p_value > 0.05

    # Compare log probability between ml4gw and bilby
    # using the same samples
    ml4gw_log_prob = ml4gw_dist.log_prob(ml4gw_samples)
    bilby_log_prob = bilby_dist.ln_prob(ml4gw_samples)
    assert np.allclose(ml4gw_log_prob.numpy(), bilby_log_prob, rtol=1e-2)

    # Repeat for luminosity distance

    minimum = 10000
    maximum = 45000
    bilby_dist = UniformComovingVolume(
        minimum=minimum, maximum=maximum, name="luminosity_distance"
    )
    ml4gw_dist = distributions.UniformComovingVolume(
        minimum=minimum, maximum=maximum, distance_type="luminosity_distance"
    )
    bilby_samples = bilby_dist.sample(100000)
    ml4gw_samples = ml4gw_dist.sample((100000,))
    _, p_value = stats.ks_2samp(ml4gw_samples.numpy(), bilby_samples)
    assert p_value > 0.05

    # Compare log probability between ml4gw and bilby
    # using the same samples
    ml4gw_log_prob = ml4gw_dist.log_prob(ml4gw_samples)
    bilby_log_prob = bilby_dist.ln_prob(ml4gw_samples)
    assert np.allclose(ml4gw_log_prob.numpy(), bilby_log_prob, rtol=1e-2)

    with pytest.raises(ValueError, match=r"Distance type must be*"):
        distributions.UniformComovingVolume(
            minimum=minimum, maximum=maximum, distance_type="dummy"
        )

    with pytest.raises(ValueError, match=r"Maximum redshift*"):
        distributions.UniformComovingVolume(
            minimum=minimum, maximum=6, distance_type="redshift"
        )
