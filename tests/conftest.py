import numpy as np
import pytest
import torch
from scipy.special import erfinv
from torch.distributions import Uniform


@pytest.fixture
def compare_against_numpy():
    """
    idea here is that if relative error is
    distributed as a zero mean gaussian with
    variance sigma, pick a tolerance such that
    all values will fall into spec prob fraction
    of the time
    """

    def compare(value, expected):
        sigma = 0.01
        prob = 0.9999
        N = np.prod(expected.shape)
        tol = sigma * erfinv(prob ** (1 / N)) * 2**0.5
        np.testing.assert_allclose(value, expected, rtol=tol)

    return compare


@pytest.fixture
def validate_whitened():
    def validate(whitened, highpass, lowpass, sample_rate, df):
        # make sure we have 0 mean unit variance
        means = whitened.mean(axis=-1)
        target = torch.zeros_like(means)
        torch.testing.assert_close(means, target, rtol=0, atol=0.02)

        stds = whitened.std(axis=-1)
        target = torch.ones_like(stds)

        # if we're highpassingi or lowpassing, then we
        # shouldn't expect the standard deviation to be
        # one because we're subtracting some power, so
        # remove roughly the expected power contributed
        # by the highpassed frequencies from the target.
        if highpass is not None:
            nyquist = sample_rate / 2
            target *= (1 - highpass / nyquist) ** 0.5
        if lowpass is not None:
            nyquist = sample_rate / 2
            target *= (lowpass / nyquist) ** 0.5

        # TODO: most statistically accurate test would be
        # to ensure that variances of the whitened data
        # along the time dimension are distributed like
        # a chi-squared with degrees of freedom equal to
        # the number of samples along time dimension, but
        # there's extra variance to account for in the
        # PSD as well that throws this off. There should be
        # a way to account for all of these sources of noise
        # in the tolerance, but for now we'll just adopt the
        # tolerance that gwpy uses in its tests
        torch.testing.assert_close(stds, target, rtol=0.04, atol=0.0)

        # check that frequencies up to close to the highpass/lowpass
        # frequencies have near 0 power.
        # TODO: the tolerance will need to increase
        # the closer to the cutoff frequency we get,
        # so what's a better way to check this
        if highpass is not None:
            fft = torch.fft.rfft(whitened, norm="ortho").abs()
            idx = int(0.8 * highpass / df)
            passed = fft[:, :, :idx]
            target = torch.zeros_like(passed)
            torch.testing.assert_close(passed, target, rtol=0, atol=0.07)
        if lowpass is not None:
            fft = torch.fft.rfft(whitened, norm="ortho").abs()
            idx = int(1.2 * lowpass / df)
            passed = fft[:, :, idx:]
            target = torch.zeros_like(passed)
            torch.testing.assert_close(passed, target, rtol=0, atol=0.07)

    return validate


# number of samples to draw from
# the distributions for testing
N_SAMPLES = 500


@pytest.fixture(params=[256, 1024, 2048])
def sample_rate(request):
    return request.param


@pytest.fixture()
def chirp_mass(request):
    dist = Uniform(5, 100)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def mass_ratio():
    dist = Uniform(0.125, 0.99)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def a_1(request):
    dist = Uniform(0, 0.90)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def a_2(request):
    dist = Uniform(0, 0.90)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def tilt_1(request):
    dist = Uniform(0, torch.pi)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def tilt_2(request):
    dist = Uniform(0, torch.pi)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def phi_12(request):
    dist = Uniform(0, 2 * torch.pi)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def phi_jl(request):
    dist = Uniform(0, 2 * torch.pi)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def distance(request):
    dist = Uniform(100, 3000)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def distance_far(request):
    dist = Uniform(500, 3000)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def distance_close(request):
    dist = Uniform(100, 500)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def theta_jn(request):
    dist = Uniform(0, torch.pi)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def phase(request):
    dist = Uniform(0, 2 * torch.pi)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def chi1(request):
    dist = Uniform(-0.999, 0.999)
    return dist.sample((N_SAMPLES,))


@pytest.fixture()
def chi2(request):
    dist = Uniform(-0.999, 0.999)
    return dist.sample((N_SAMPLES,))
