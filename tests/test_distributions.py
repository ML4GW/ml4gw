from math import pi

import numpy as np
import pytest
from bbhnet.data import distributions
from scipy import optimize

# TODO: for all tests, how to validate that
# distribution has the expected shape?


def test_uniform():
    sampler = distributions.Uniform()

    samples = sampler(10)
    assert len(samples) == 10
    assert ((0 <= samples) & (samples <= 1)).all()

    sampler = distributions.Uniform(-3, 5)
    samples = sampler(100)
    assert len(samples) == 100
    assert ((-3 <= samples) & (samples <= 5)).all()

    # check that the mean is roughly correct
    # (within two standard deviations)
    samples = sampler(10000)
    mean = samples.mean().item()
    variance = 64 / 12
    sample_variance = variance / 10000
    sample_std = sample_variance**0.5
    assert abs(mean - 1) < (2 * sample_std)


def test_cosine():
    sampler = distributions.Cosine()
    samples = sampler(10)
    assert len(samples) == 10
    assert ((-pi / 2 <= samples) & (samples <= pi / 2)).all()

    sampler = distributions.Cosine(-3, 5)
    samples = sampler(100)
    assert len(samples) == 100
    assert ((-3 <= samples) & (samples <= 5)).all()


def test_log_normal():
    sampler = distributions.LogNormal(6, 4)
    samples = sampler(10)
    assert len(samples) == 10
    assert (0 < samples).all()

    sampler = distributions.LogNormal(6, 4, 3)
    samples = sampler(100)
    assert len(samples) == 100
    assert (3 <= samples).all()

    # check that mean is roughly correct
    # (within 2 standard deviations)
    sampler = distributions.LogNormal(10, 2)
    samples = sampler(10000)
    mean = samples.mean().item()
    assert (abs(mean - 10) / 10) < (2 * 2 / 10000**0.5)


def test_volume_distributed_snr():
    """Test PowerLaw distribution against expected distribution of SNRs"""
    ref_snr = 8
    sampler = distributions.PowerLaw(
        x_min=ref_snr, x_max=float("inf"), alpha=4
    )
    samples = sampler(10000).numpy()
    # check 1/rho^4 behavior
    counts, ebins = np.histogram(samples, bins=100)
    bins = ebins[1:] + ebins[:-1]
    bins *= 0.5

    def foo(x, a, b):
        return a * x ** (-b)

    popt, pcov = optimize.curve_fit(foo, bins, counts, (20, 3))
    # popt[1] is the index
    assert popt[1] == pytest.approx(4, rel=1e-1)
