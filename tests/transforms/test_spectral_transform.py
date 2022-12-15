from io import BytesIO

import numpy as np
import pytest
import scipy
import torch
from packaging import version
from scipy import signal

from ml4gw.transforms.spectral import SpectralDensity

TOL = 1e-7


@pytest.fixture(params=[1, 4, 8])
def length(request):
    return request.param


@pytest.fixture(params=[1024, 4096])
def sample_rate(request):
    return request.param


@pytest.mark.parametrize(
    "overlap,fftlength",
    [[None, 2], [0.5, 2], [2, 2], [None, 4], [2, 4], [4, 4]],
)
def test_init(overlap, fftlength, sample_rate):
    if overlap is not None and overlap >= fftlength:
        with pytest.raises(ValueError):
            transform = SpectralDensity(sample_rate, fftlength, overlap)
        return
    else:
        transform = SpectralDensity(sample_rate, fftlength, overlap)

    if overlap is None:
        expected_stride = int(fftlength * sample_rate // 2)
    else:
        expected_stride = int(fftlength * sample_rate) - int(
            overlap * sample_rate
        )
    assert transform.nstride == expected_stride
    assert len(list(transform.buffers())) == 2

    # test read/write
    weights_io = BytesIO()
    torch.save(transform.state_dict(), weights_io)

    weights_io.seek(0)
    transform.load_state_dict(torch.load(weights_io))


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


@pytest.fixture(params=[True, False])
def fast(request):
    return request.param


@pytest.fixture(params=["mean", "median"])
def average(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def ndim(request):
    return request.param


def test_spectral_density(
    length, sample_rate, fftlength, overlap, fast, average, ndim
):
    batch_size = 8
    num_channels = 5
    if overlap is not None and overlap >= fftlength:
        return

    transform = SpectralDensity(
        sample_rate, fftlength, overlap, average=average, fast=fast
    )

    shape = [int(length * sample_rate)]
    if ndim > 1:
        shape.insert(0, num_channels)
    if ndim > 2:
        shape.insert(0, batch_size)
    x = np.random.randn(*shape)

    # make sure we catch if the fftlength is too long for the data
    if fftlength > length:
        with pytest.raises(ValueError) as exc_info:
            transform(torch.Tensor(x))
        assert str(exc_info.value).startswith("Number of samples")
        return

    # perform the transform and confirm the shape is correct
    torch_result = transform(torch.Tensor(x)).numpy()
    num_freq_bins = int(fftlength * sample_rate) // 2 + 1
    shape[-1] = num_freq_bins
    assert torch_result.shape == tuple(shape)

    # now verify against the result from scipy
    _, scipy_result = signal.welch(
        x,
        fs=sample_rate,
        nperseg=transform.nperseg,
        noverlap=transform.nperseg - transform.nstride,
        window=signal.windows.hann(transform.nperseg, False),
        average=average,
    )

    # if we're using the fast implementation, only guarantee
    # that components higher than the first two are correct
    if fast:
        torch_result = torch_result[..., 2:]
        scipy_result = scipy_result[..., 2:]
    assert np.isclose(torch_result, scipy_result, rtol=TOL).all()


@pytest.fixture(params=[0, 1])
def y_ndim(request):
    return request.param


def _shape_checks(ndim, y_ndim, x, y, transform):
    # verify that time dimensions must match
    with pytest.raises(ValueError) as exc_info:
        transform(x, y[..., :-1])
    assert str(exc_info.value).startswith("Time dimensions")

    # verify that y can't have more dims than x
    if y_ndim == 0:
        with pytest.raises(ValueError) as exc_info:
            transform(x, y[None])
        assert str(exc_info.value).startswith("Can't compute")

        if ndim == 1:
            assert "1D" in str(exc_info.value)

    # verify that if x is greater than 1D and y has
    # the same dimensionality, their shapes must
    # fully match
    if ndim > 1 and y_ndim == 0:
        with pytest.raises(ValueError) as exc_info:
            transform(x, y[:-1])
        assert str(exc_info.value).startswith("If x and y tensors")

    # verify for 3D x's that 2D y's must have the same batch
    # dimension, and that y cannot be 1D
    if ndim == 3 and y_ndim == 1:
        with pytest.raises(ValueError) as exc_info:
            transform(x, y[:-1])
        assert str(exc_info.value).startswith("If x is a 3D tensor")

        with pytest.raises(ValueError) as exc_info:
            transform(x, y[0])
        assert str(exc_info.value).startswith("Can't compute cross")


def test_transform_with_csd(
    y_ndim, length, sample_rate, fftlength, overlap, average, ndim, fast
):
    batch_size = 8
    num_channels = 5
    if overlap is not None and overlap >= fftlength:
        return

    if y_ndim == 1 and ndim == 1:
        return

    transform = SpectralDensity(
        sample_rate, fftlength, overlap, average=average, fast=fast
    )

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

    # make sure we catch if the fftlength is too long for the data
    if not fast:
        with pytest.raises(NotImplementedError):
            transform(x, y)
        return
    elif fftlength > length:
        with pytest.raises(ValueError) as exc_info:
            transform(x, y)
        assert str(exc_info.value).startswith("Number of samples")
        return

    # perform the transform and confirm the shape is correct
    torch_result = transform(x, y).numpy()
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
                nperseg=transform.nperseg,
                noverlap=transform.nperseg - transform.nstride,
                window=signal.windows.hann(transform.nperseg, False),
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
            nperseg=transform.nperseg,
            noverlap=transform.nperseg - transform.nstride,
            window=signal.windows.hann(transform.nperseg, False),
            average=average,
        )
    assert scipy_result.shape == torch_result.shape

    scipy_version = version.parse(scipy.__version__)
    num_windows = (x.shape[-1] - transform.nperseg) // transform.nstride + 1
    if (
        average == "median"
        and scipy_version < version.parse("1.9")
        and num_windows > 1
    ):
        # scipy actually had a bug in the median calc for
        # csd, see this issue:
        # https://github.com/scipy/scipy/issues/15601
        try:
            from scipy.signal.spectral import _median_bias
        except ImportError:
            from scipy.signal._spectral_py import _median_bias

        scipy_result *= _median_bias(num_freq_bins)
        scipy_result /= _median_bias(num_windows)

    if fast:
        torch_result = torch_result[..., 2:]
        scipy_result = scipy_result[..., 2:]

    ratio = torch_result / scipy_result
    assert np.isclose(torch_result, scipy_result, rtol=TOL).all(), ratio

    _shape_checks(ndim, y_ndim, x, y, transform)
