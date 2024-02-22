import numpy as np
import pytest
import torch
from scipy import optimize
from torch import pi

from ml4gw import distributions

# TODO: for all tests, how to validate that
# distribution has the expected shape?


def test_log_uniform():
    sampler = distributions.LogUniform(torch.e, torch.e**2)
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
    assert ((-pi / 2 <= samples) & (samples <= pi / 2)).all()

    sampler = distributions.Cosine(torch.as_tensor(-3), torch.as_tensor(5))
    samples = sampler.sample((100,))
    assert len(samples) == 100
    assert ((-3 <= samples) & (samples <= 5)).all()


def test_power_law():
    """Test PowerLaw distribution"""
    ref_snr = 8
    sampler = distributions.PowerLaw(ref_snr, float("inf"), index=-4)
    samples = sampler.sample((10000,)).numpy()
    # check x^-4 behavior
    counts, ebins = np.histogram(samples, bins=100)
    bins = ebins[1:] + ebins[:-1]
    bins *= 0.5

    def foo(x, a, b):
        return a * x**b

    popt, _ = optimize.curve_fit(foo, bins, counts, (20, 3))
    # popt[1] is the index
    assert popt[1] == pytest.approx(-4, rel=1e-1)

    min_dist = 10
    max_dist = 1000
    uniform_in_volume = distributions.PowerLaw(min_dist, max_dist, index=2)
    samples = uniform_in_volume.sample((10000,)).numpy()
    # check d^2 behavior
    counts, ebins = np.histogram(samples, bins=100)
    bins = ebins[1:] + ebins[:-1]
    bins *= 0.5

    popt, _ = optimize.curve_fit(foo, bins, counts, (20, 3))
    # popt[1] is the index
    assert popt[1] == pytest.approx(2, rel=1e-1)

    # test 1/x distribution
    inverse_in_distance = distributions.PowerLaw(min_dist, max_dist, index=-1)
    samples = inverse_in_distance.sample((10000,)).numpy()
    counts, ebins = np.histogram(samples, bins=100)
    bins = ebins[1:] + ebins[:-1]
    bins *= 0.5
    popt, _ = optimize.curve_fit(foo, bins, counts, (20, 3))
    # popt[1] is the index
    assert popt[1] == pytest.approx(-1, rel=1e-1)


def test_delta_function():
    sampler = distributions.DeltaFunction(peak=torch.as_tensor(20))
    samples = sampler.sample((10,))
    assert (samples == 20).all()
