from functools import partial

import numpy as np
import pytest
import scipy
import torch
from packaging import version
from scipy import signal
from scipy.special import erfinv

from ml4gw.spectral import fast_spectral_density, spectral_density, whiten

# idea here is that if relative error is
# distributed a zero mean gaussian with
# variance sigma, pick a tolerance such that
# all values will fall into spec prob fraction
# of the time
sigma = 1e-3
prob = 0.999


def get_tolerance(shape):
    N = np.product(shape)
    return sigma * erfinv(prob ** (1 / N)) * 2**0.5


@pytest.fixture(params=[1, 4, 8])
def length(request):
    return request.param


@pytest.fixture(params=[1024, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[0.5, 2, 4])
def fftlength(request):
    return request.param


@pytest.fixture(params=[None, 0.1, 0.5, 1])
def overlap(request):
    return request.param


@pytest.fixture(params=[None, 55, [25, 55]])
def freq_low(request):
    return request.param


@pytest.fixture(params=[None, 65, [35, 65]])
def freq_high(request):
    return request.param


@pytest.fixture(params=["mean", "median"])
def average(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def ndim(request):
    return request.param


def test_fast_spectral_density(
    length, sample_rate, fftlength, overlap, average, ndim
):
    batch_size = 8
    num_channels = 5
    if overlap is not None and overlap >= fftlength:
        return

    shape = [int(length * sample_rate)]
    if ndim > 1:
        shape.insert(0, num_channels)
    if ndim > 2:
        shape.insert(0, batch_size)
    x = np.random.randn(*shape)

    nperseg = int(fftlength * sample_rate)
    if overlap is None:
        nstride = int(fftlength * sample_rate // 2)
    else:
        nstride = int((fftlength - overlap) * sample_rate)

    window = torch.hann_window(nperseg)
    fsd = partial(
        fast_spectral_density,
        nperseg=nperseg,
        nstride=nstride,
        window=window,
        scale=1 / (sample_rate * (window**2).sum()),
        average=average,
    )
    # make sure initial shape check works
    if fftlength > length:
        with pytest.raises(ValueError) as exc_info:
            fsd(torch.Tensor(x))
        assert str(exc_info.value).startswith("Number of samples")
        return

    # perform the transform and confirm the shape is correct
    torch_result = fsd(torch.Tensor(x)).numpy()
    num_freq_bins = int(fftlength * sample_rate) // 2 + 1
    shape[-1] = num_freq_bins
    assert torch_result.shape == tuple(shape)

    # now verify against the result from scipy
    _, scipy_result = signal.welch(
        x,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=nperseg - nstride,
        window=signal.windows.hann(nperseg, False),
        average=average,
    )

    # if we're using the fast implementation, only guarantee
    # that components higher than the first two are correct
    torch_result = torch_result[..., 2:]
    scipy_result = scipy_result[..., 2:]
    tol = get_tolerance(scipy_result.shape)
    np.testing.assert_allclose(torch_result, scipy_result, rtol=tol)

    # make sure we catch any calls with too many dimensions
    if ndim == 3:
        with pytest.raises(ValueError) as exc_info:
            fsd(torch.Tensor(x[None]))
        assert str(exc_info.value).startswith("Can't compute spectral")


@pytest.fixture(params=[0, 1])
def y_ndim(request):
    return request.param


def _shape_checks(ndim, y_ndim, x, y, f):
    # verify that time dimensions must match
    with pytest.raises(ValueError) as exc_info:
        f(x, y=y[..., :-1])
    assert str(exc_info.value).startswith("Time dimensions")

    # verify that y can't have more dims than x
    if y_ndim == 0:
        with pytest.raises(ValueError) as exc_info:
            f(x, y=y[None])
        assert str(exc_info.value).startswith("Can't compute")

        if ndim == 1:
            assert "1D" in str(exc_info.value)

    # verify that if x is greater than 1D and y has
    # the same dimensionality, their shapes must
    # fully match
    if ndim > 1 and y_ndim == 0:
        with pytest.raises(ValueError) as exc_info:
            f(x, y=y[:-1])
        assert str(exc_info.value).startswith("If x and y tensors")

    # verify for 3D x's that 2D y's must have the same batch
    # dimension, and that y cannot be 1D
    if ndim == 3 and y_ndim == 1:
        with pytest.raises(ValueError) as exc_info:
            f(x, y=y[:-1])
        assert str(exc_info.value).startswith("If x is a 3D tensor")

        with pytest.raises(ValueError) as exc_info:
            f(x, y=y[0])
        assert str(exc_info.value).startswith("Can't compute cross")


def test_fast_spectral_density_with_y(
    y_ndim, length, sample_rate, fftlength, overlap, average, ndim
):
    batch_size = 8
    num_channels = 5
    if overlap is not None and overlap >= fftlength:
        return

    if y_ndim == 1 and ndim == 1:
        return

    shape = [int(length * sample_rate)]
    if ndim > 1:
        shape.insert(0, num_channels)
    if ndim > 2:
        shape.insert(0, batch_size)
    x = np.random.randn(*shape)

    if ndim == 1 or (y_ndim == 1 and ndim == 2):
        y = np.random.randn(shape[-1])
    elif ndim == 3 and y_ndim == 1:
        y = np.random.randn(shape[0], shape[-1])
    else:
        y = np.random.randn(*shape)

    x = torch.Tensor(x)
    y = torch.Tensor(y)

    nperseg = int(fftlength * sample_rate)
    if overlap is None:
        nstride = int(fftlength * sample_rate // 2)
    else:
        nstride = int((fftlength - overlap) * sample_rate)
    window = torch.hann_window(nperseg)
    fsd = partial(
        fast_spectral_density,
        nperseg=nperseg,
        nstride=nstride,
        window=window,
        scale=1 / (sample_rate * (window**2).sum()),
        average=average,
    )

    # make sure we catch if the fftlength is too long for the data
    if fftlength > length:
        with pytest.raises(ValueError) as exc_info:
            fsd(x, y=y)
        assert str(exc_info.value).startswith("Number of samples")
        return

    # perform the transform and confirm the shape is correct
    torch_result = fsd(x, y=y).numpy()
    num_freq_bins = int(fftlength * sample_rate) // 2 + 1
    shape[-1] = num_freq_bins
    assert torch_result.shape == tuple(shape)

    if ndim == 3:
        scipy_result = []
        if y_ndim == 1:
            x = x.transpose(1, 0)
            y = [y] * len(x)

        for i, j in zip(x, y):
            _, result = signal.csd(
                i,
                j,
                fs=sample_rate,
                nperseg=nperseg,
                noverlap=nperseg - nstride,
                window=signal.windows.hann(nperseg, False),
                average=average,
            )
            scipy_result.append(result)
        scipy_result = np.stack(scipy_result)

        if y_ndim == 1:
            x = x.transpose(1, 0)
            y = y[0]
            scipy_result = scipy_result.transpose(1, 0, 2)
    else:
        _, scipy_result = signal.csd(
            x,
            y,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=nperseg - nstride,
            window=signal.windows.hann(nperseg, False),
            average=average,
        )
    assert scipy_result.shape == torch_result.shape

    scipy_version = version.parse(scipy.__version__)
    num_windows = (x.shape[-1] - nperseg) // nstride + 1
    if (
        average == "median"
        and scipy_version < version.parse("1.9")
        and num_windows > 1
    ):
        # scipy actually had a bug in the median calc for
        # csd, see this issue:
        # https://github.com/scipy/scipy/issues/15601
        from scipy.signal.spectral import _median_bias

        scipy_result *= _median_bias(num_freq_bins)
        scipy_result /= _median_bias(num_windows)

    torch_result = torch_result[..., 2:]
    scipy_result = scipy_result[..., 2:]
    tol = get_tolerance(scipy_result.shape)
    np.testing.assert_allclose(torch_result, scipy_result, rtol=tol)

    _shape_checks(ndim, y_ndim, x, y, fsd)


def test_spectral_density(
    length, sample_rate, fftlength, overlap, average, ndim
):
    batch_size = 8
    num_channels = 5
    if overlap is not None and overlap >= fftlength:
        return

    shape = [int(length * sample_rate)]
    if ndim > 1:
        shape.insert(0, num_channels)
    if ndim > 2:
        shape.insert(0, batch_size)
    x = np.random.randn(*shape)

    nperseg = int(fftlength * sample_rate)
    if overlap is None:
        nstride = int(fftlength * sample_rate // 2)
    else:
        nstride = int((fftlength - overlap) * sample_rate)

    window = torch.hann_window(nperseg)
    sd = partial(
        spectral_density,
        nperseg=nperseg,
        nstride=nstride,
        window=window,
        scale=1 / (sample_rate * (window**2).sum()),
        average=average,
    )
    # make sure initial shape check works
    if fftlength > length:
        with pytest.raises(ValueError) as exc_info:
            sd(torch.Tensor(x))
        assert str(exc_info.value).startswith("Number of samples")
        return

    # perform the transform and confirm the shape is correct
    torch_result = sd(torch.Tensor(x)).numpy()
    num_freq_bins = int(fftlength * sample_rate) // 2 + 1
    shape[-1] = num_freq_bins
    assert torch_result.shape == tuple(shape)

    # now verify against the result from scipy
    _, scipy_result = signal.welch(
        x,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=nperseg - nstride,
        window=signal.windows.hann(nperseg, False),
        average=average,
    )
    tol = get_tolerance(scipy_result.shape)
    np.testing.assert_allclose(torch_result, scipy_result, rtol=tol)

    # make sure we catch any calls with too many dimensions
    if ndim == 3:
        with pytest.raises(ValueError) as exc_info:
            sd(torch.Tensor(x[None]))
        assert str(exc_info.value).startswith("Can't compute spectral")


@pytest.fixture(params=[1, 2])
def fduration(request):
    return request.param


@pytest.fixture(params=[None, 32])
def highpass(request):
    return request.param


@pytest.fixture(params=[32, 64, 128])
def whiten_length(request):
    return request.param


@pytest.fixture(params=[64, 128])
def background_length(request):
    return request.param


def test_whiten(
    fftlength,
    fduration,
    sample_rate,
    highpass,
    ndim,
    whiten_length,
    background_length,
    validate_whitened,
):
    batch_size = 8
    num_channels = 5
    background_size = int(background_length * sample_rate)
    background_shape = (background_size,)

    mean = 2
    std = 5
    if ndim > 1:
        background_shape = (num_channels,) + background_shape

        arr = torch.arange(num_channels).view(-1, 1)
        mean = arr + mean
        std = 0.1 * arr + std
    if ndim > 2:
        arr = torch.arange(num_channels)
        mean = torch.stack(
            [i * num_channels + mean for i in range(batch_size)]
        )
        std = torch.stack(
            [i * 0.1 * num_channels + std for i in range(batch_size)]
        )
        background_shape = (batch_size,) + background_shape

    background = mean + std * torch.randn(*background_shape)
    nperseg = int(fftlength * sample_rate)
    window = torch.hann_window(nperseg)
    psd = spectral_density(
        background,
        nperseg=nperseg,
        nstride=int(fftlength * sample_rate / 2),
        window=window,
        scale=1 / (sample_rate * (window**2).sum()),
    )

    size = int(whiten_length * sample_rate)
    X = mean + std * torch.randn(batch_size, num_channels, size)
    whitened = whiten(X, psd, fduration, sample_rate, highpass)
    expected_size = int((whiten_length - fduration) * sample_rate)
    assert whitened.shape == (batch_size, num_channels, expected_size)

    validate_whitened(whitened, highpass, sample_rate, 1 / whiten_length)

    # inject a gaussian pulse into the timeseries and
    # ensure that its max value comes out to the same place
    # adapted from gwpy's tests
    # https://github.com/gwpy/gwpy/blob/e9f687e8d34720d9d386a6bd7f95e3b759264739/gwpy/timeseries/tests/test_timeseries.py#L1285  # noqa
    t = np.arange(size) / sample_rate - whiten_length / 2
    glitch = torch.Tensor(signal.gausspulse(t, bw=100))
    glitch = 10 * std * glitch
    inj = X + glitch
    whitened = whiten(inj, psd, fduration, sample_rate, highpass)
    maxs = whitened.argmax(-1) / sample_rate + fduration / 2
    target = torch.ones_like(maxs) * whiten_length / 2
    torch.testing.assert_close(maxs, target, rtol=0, atol=0.01)
