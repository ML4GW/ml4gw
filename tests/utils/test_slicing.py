from unittest.mock import patch

import numpy as np
import pytest
import torch

from ml4gw.utils import slicing


@pytest.fixture(params=[1, 8, 9])
def kernel_size(request):
    return request.param


@pytest.fixture(params=[1, 5])
def num_channels(request):
    return request.param


def test_slice_kernels(kernel_size, num_channels):
    x = torch.arange(100)
    idx = 7 * torch.arange(2, 10)
    result = slicing.slice_kernels(x, idx, kernel_size)
    assert result.shape == (8, kernel_size)

    for i, y in enumerate(result.cpu().numpy()):
        start = (i + 2) * 7
        stop = start + kernel_size
        assert (y == np.arange(start, stop)).all()

    X = torch.stack([x + i * 100 for i in range(num_channels)])
    result = slicing.slice_kernels(X, idx, kernel_size)
    assert result.shape == (8, num_channels, kernel_size)

    for i, Y in enumerate(result.cpu().numpy()):
        for j, y in enumerate(Y):
            start = j * 100 + (i + 2) * 7
            stop = start + kernel_size
            assert (y == np.arange(start, stop)).all()

    idx = torch.stack([idx + i * 3 for i in range(num_channels)]).t()
    result = slicing.slice_kernels(X, idx, kernel_size)
    assert result.shape == (8, num_channels, kernel_size)

    for i, Y in enumerate(result.cpu().numpy()):
        for j, y in enumerate(Y):
            start = j * 103 + (i + 2) * 7
            stop = start + kernel_size
            assert (y == np.arange(start, stop)).all()


def test_sample_kernels_1D(kernel_size, num_channels):
    N = 8
    x = torch.arange(100)
    with pytest.raises(ValueError):
        slicing.sample_kernels(x[: kernel_size - 1], N, kernel_size)

    # first test 1D behavior
    return_value = 7 * torch.arange(N)
    with patch("torch.randint", return_value=return_value) as mock:
        result = slicing.sample_kernels(x, N, kernel_size)
    mock.assert_called_once_with(100 - kernel_size, size=(N,))

    assert result.shape == (N, kernel_size)
    for i, y in enumerate(result.cpu().numpy()):
        start = i * 7
        stop = start + kernel_size
        assert (y == np.arange(start, stop)).all()

    # now 1D behavior without patching
    result = slicing.sample_kernels(x, N, kernel_size)
    assert result.shape == (N, kernel_size)
    for i, y in enumerate(result.cpu().numpy()):
        assert y[0] <= (100 - kernel_size)
        assert (y == np.arange(y[0], y[0] + kernel_size)).all()


@pytest.fixture(params=[None, 0, 4, -4])
def max_center_offset(request):
    return request.param


@pytest.fixture(params=[True, False])
def coincident(request):
    return request.param


def test_sample_kernels_2D(
    kernel_size, num_channels, max_center_offset, coincident
):
    N = 8
    x = torch.arange(100)
    X = torch.stack([x + i * 100 for i in range(num_channels)])
    with pytest.raises(ValueError):
        slicing.sample_kernels(X[:, : kernel_size - 1], N, kernel_size)

    shape = (N,)
    return_value = 7 * torch.arange(N)
    if not coincident:
        return_value = torch.stack(
            [return_value + i for i in range(num_channels)]
        ).t()
        shape = (N, num_channels)

    if (
        max_center_offset is not None
        and max_center_offset < 0
        and kernel_size <= 2 * abs(max_center_offset)
    ):
        with pytest.raises(ValueError):
            slicing.sample_kernels(
                X, N, kernel_size, max_center_offset, coincident
            )
        return

    with patch("torch.randint", return_value=return_value) as mock:
        result = slicing.sample_kernels(
            X, N, kernel_size, max_center_offset, coincident
        )
    assert result.shape == (N, num_channels, kernel_size)
    result = result.cpu().numpy()

    if max_center_offset is None:
        min_val, max_val = 0, 100
    elif max_center_offset == 0:
        min_val, max_val = 50 - kernel_size, 50
    elif max_center_offset == -4:
        min_val, max_val = 54 - kernel_size, 46
    else:
        min_val, max_val = 46 - kernel_size, 50
    mock.assert_called_with(min_val, max_val, size=shape)

    for i, sample in enumerate(result):
        for j, channel in enumerate(sample):
            if coincident:
                start = i * 7 + j * 100
                stop = start + kernel_size
            else:
                start = i * 7 + j * 101
                stop = start + kernel_size

            assert (channel == np.arange(start, stop)).all()
