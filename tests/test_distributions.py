import math
from math import pi

import numpy as np
import pytest
from scipy import optimize

from ml4gw import distributions

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
    # (within three standard deviations)
    samples = sampler(100000)
    mean = samples.mean().item()
    variance = 64 / 12
    sample_variance = variance / 10000
    sample_std = sample_variance**0.5
    assert abs(mean - 1) < (3 * sample_std)


def test_log_uniform():
    sampler = distributions.LogUniform(math.e, math.e**2)
    samples = sampler(10)
    assert len(samples) == 10
    assert ((math.e <= samples) & (math.e**2 <= 100)).all()

    # check that the mean is roughly correct
    # (within three standard deviations)
    samples = sampler(100000)
    log_samples = np.log10(samples)

    mean = log_samples.mean().item()
    variance = 4 / 12
    sample_variance = variance / 10000
    sample_std = sample_variance**0.5
    assert abs(mean - 1.5) < (3 * sample_std)


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
    assert (abs(mean - 10) / 10) < (3 * 2 / 10000**0.5)


def test_power_law():
    """Test PowerLaw distribution against expected distribution of SNRs"""
    ref_snr = 8
    sampler = distributions.PowerLaw(
        x_min=ref_snr, x_max=float("inf"), alpha=4
    )
    samples = sampler(10000).numpy()
    # check x^-4 behavior
    counts, ebins = np.histogram(samples, bins=100)
    bins = ebins[1:] + ebins[:-1]
    bins *= 0.5

    def foo(x, a, b):
        return a * x ** (-b)

    popt, _ = optimize.curve_fit(foo, bins, counts, (20, 3))
    # popt[1] is the index
    assert popt[1] == pytest.approx(4, rel=1e-1)
