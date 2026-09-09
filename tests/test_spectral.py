from functools import partial

import numpy as np
import pytest
import torch
from scipy import signal

from ml4gw.spectral import (
    fast_spectral_density,
    minimum_phase_whitening_filter,
    spectral_density,
    whiten,
)


class TestMinimumPhaseWhiteningFilter:
    def test_flat_psd_is_impulse(self):
        psd = torch.ones(33, dtype=torch.float64)
        kernel = minimum_phase_whitening_filter(psd)

        expected = torch.zeros(64, dtype=torch.float64)
        expected[0] = 1
        torch.testing.assert_close(kernel, expected, rtol=0, atol=1e-12)

    def test_response_has_inverse_asd_magnitude(self):
        psd = torch.linspace(0.5, 4, 33, dtype=torch.float64).repeat(2, 1)
        kernel = minimum_phase_whitening_filter(psd)
        response = torch.fft.rfft(kernel, dim=-1)

        torch.testing.assert_close(
            response.abs(), psd.rsqrt(), rtol=1e-12, atol=1e-12
        )

    def test_odd_length_response(self):
        psd = torch.linspace(0.5, 4, 33, dtype=torch.float64)
        kernel = minimum_phase_whitening_filter(psd, n_fft=65)

        assert kernel.size(-1) == 65
        torch.testing.assert_close(
            torch.fft.rfft(kernel).abs(),
            psd.rsqrt(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_first_order_filter_matches_analytic_response(self):
        n_fft = 256
        coefficient = 0.8
        omega = (
            2
            * torch.pi
            * torch.arange(n_fft // 2 + 1, dtype=torch.float64)
            / n_fft
        )
        expected = 1 - coefficient * torch.exp(-1j * omega)
        psd = expected.abs().square().reciprocal()

        kernel = minimum_phase_whitening_filter(psd)

        expected_kernel = torch.zeros(n_fft, dtype=torch.float64)
        expected_kernel[:2] = torch.tensor(
            [1, -coefficient], dtype=torch.float64
        )
        torch.testing.assert_close(kernel, expected_kernel, rtol=0, atol=3e-15)

    @pytest.mark.parametrize(
        "psd",
        [
            torch.tensor(1.0),
            torch.tensor([1.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, torch.inf]),
            torch.tensor([1, 2]),
        ],
    )
    def test_invalid_psd(self, psd):
        with pytest.raises(ValueError):
            minimum_phase_whitening_filter(psd)

    def test_inconsistent_fft_length(self):
        with pytest.raises(ValueError, match="PSD length"):
            minimum_phase_whitening_filter(torch.ones(3), n_fft=8)


@pytest.fixture(params=[4, 8])
def length(request):
    return request.param


@pytest.fixture(params=[1024, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[0.5, 2])
def fftlength(request):
    return request.param


@pytest.fixture(params=[None, 0.5])
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
    length,
    sample_rate,
    fftlength,
    overlap,
    average,
    ndim,
    compare_against_numpy,
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
    compare_against_numpy(torch_result, scipy_result, num_bad=1)

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
    y_ndim,
    length,
    sample_rate,
    fftlength,
    overlap,
    average,
    ndim,
    compare_against_numpy,
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

        for i, j in zip(x, y, strict=True):
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

    torch_result = torch_result[..., 2:]
    scipy_result = scipy_result[..., 2:]
    compare_against_numpy(torch_result, scipy_result, num_bad=1)
    _shape_checks(ndim, y_ndim, x, y, fsd)


def test_spectral_density(
    length,
    sample_rate,
    fftlength,
    overlap,
    average,
    ndim,
    compare_against_numpy,
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
    compare_against_numpy(torch_result, scipy_result, num_bad=1)

    # make sure we catch any calls with too many dimensions
    if ndim == 3:
        with pytest.raises(ValueError) as exc_info:
            sd(torch.Tensor(x[None]))
        assert str(exc_info.value).startswith("Can't compute spectral")


@pytest.fixture(params=[2])
def fduration(request):
    return request.param


@pytest.fixture(params=[None, 32])
def highpass(request):
    return request.param


@pytest.fixture(params=[None, 512])
def lowpass(request):
    return request.param


@pytest.fixture(params=[64])
def whiten_length(request):
    return request.param


def test_whiten(
    fduration,
    highpass,
    lowpass,
    ndim,
    whiten_length,
    validate_whitened,
):
    # hard-coding fftlength since longer values
    # produce higher variance estimates of the PSD.
    # which lead to higher variance in whitened output.
    # Adopting this value and the associated tolerance
    # in conftest.py to stay consistent with gwpy's tests.
    # TODO: what's the exact nature of this relationship
    # and how we can we build tolerances against it?
    fftlength = 2
    batch_size = 8
    num_channels = 5
    sample_rate = 16384
    background_length = 64
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

    with pytest.raises(ValueError, match=r"Not enough timeseries samples*"):
        failure_size = int(fduration * sample_rate)
        X = torch.randn(batch_size, num_channels, failure_size)
        whiten(X, psd, fduration, sample_rate, highpass, lowpass)
    size = int(whiten_length * sample_rate)
    filter_size = int(fduration * sample_rate)
    X = mean + std * torch.randn(batch_size, num_channels, size)

    whitened = whiten(X, psd, fduration, sample_rate, highpass, lowpass)
    expected_size = size - filter_size
    assert whitened.shape == (batch_size, num_channels, expected_size)
    validate_whitened(
        whitened, highpass, lowpass, sample_rate, 1 / whiten_length
    )

    # Check that not cropping produces the expected values and shape
    whitened_uncropped = whiten(
        X, psd, fduration, sample_rate, highpass, lowpass, crop=False
    )
    pad = filter_size // 2
    assert torch.all(whitened_uncropped[..., pad:-pad] == whitened)

    # inject a gaussian pulse into the timeseries and
    # ensure that its max value comes out to the same place
    # adapted from gwpy's tests
    # https://github.com/gwpy/gwpy/blob/e9f687e8d34720d9d386a6bd7f95e3b759264739/gwpy/timeseries/tests/test_timeseries.py#L1285  # noqa
    t = np.arange(size) / sample_rate - whiten_length / 2
    glitch = torch.Tensor(signal.gausspulse(t, fc=128, bw=2))
    glitch = 10 * std * glitch
    inj = X + glitch
    whitened = whiten(inj, psd, fduration, sample_rate, highpass, lowpass)
    maxs = whitened.argmax(-1) / sample_rate + fduration / 2
    target = torch.ones_like(maxs) * whiten_length / 2
    torch.testing.assert_close(maxs, target, rtol=0, atol=0.01)


def test_spectral_density_short_input():
    nperseg = 512
    window = torch.hann_window(nperseg)
    with pytest.raises(ValueError, match="Number of samples"):
        fast_spectral_density(
            torch.randn(100),
            nperseg=nperseg,
            nstride=256,
            window=window,
            scale=1.0,
        )


def test_spectral_density_odd_nperseg():
    nperseg = 511
    x = torch.randn(nperseg * 4)
    window = torch.hann_window(nperseg)
    result = spectral_density(
        x,
        nperseg=nperseg,
        nstride=256,
        window=window,
        scale=1.0 / (window**2).sum(),
    )
    assert result.shape[-1] == nperseg // 2 + 1
