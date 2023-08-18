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


def test_unfold_windows():
    # 1D
    x = torch.tensor([1, 2, 3, 4, 5, 6], dtype=float)
    result = slicing.unfold_windows(x, window_size=3, stride=2)
    assert result.tolist() == [[1, 2, 3], [3, 4, 5]]

    result, rem = slicing.unfold_windows(
        x, window_size=3, stride=2, drop_last=False
    )
    assert rem.tolist() == [[6]]

    # 2D
    x = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], dtype=float)
    result = slicing.unfold_windows(x, window_size=3, stride=2)
    assert result.tolist() == [[[1, 2, 3], [2, 3, 4]], [[3, 4, 5], [4, 5, 6]]]

    result, rem = slicing.unfold_windows(
        x, window_size=3, stride=2, drop_last=False
    )
    assert rem.tolist() == [[[6], [7]]]

    # 3D
    x = torch.tensor(
        [
            [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]],
            [[3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9]],
        ],
        dtype=float,
    )
    result = slicing.unfold_windows(x, window_size=3, stride=2)
    assert result.tolist() == [
        [[[1, 2, 3], [2, 3, 4]], [[3, 4, 5], [4, 5, 6]]],
        [[[3, 4, 5], [4, 5, 6]], [[5, 6, 7], [6, 7, 8]]],
    ]

    result, rem = slicing.unfold_windows(
        x, window_size=3, stride=2, drop_last=False
    )
    assert rem.tolist() == [[[[6], [7]], [[8], [9]]]]


def test_slice_kernels(kernel_size, num_channels):
    x = torch.arange(100)
    idx = 7 * torch.arange(2, 10)

    # test 1D slice
    result = slicing.slice_kernels(x, idx, kernel_size)
    assert result.shape == (8, kernel_size)
    for i, y in enumerate(result.cpu().numpy()):
        start = (i + 2) * 7
        stop = start + kernel_size
        assert (y == np.arange(start, stop)).all()

    # ensure 1D slice with >1D idx raises error
    with pytest.raises(ValueError) as exc:
        slicing.slice_kernels(x, idx[None], kernel_size)
    assert str(exc.value).startswith("idx tensor has 2 dimensions")

    # test 2D slice with 1D indices
    X = torch.stack([x + i * 100 for i in range(num_channels)])
    result = slicing.slice_kernels(X, idx, kernel_size)
    assert result.shape == (8, num_channels, kernel_size)

    for i, Y in enumerate(result.cpu().numpy()):
        for j, y in enumerate(Y):
            start = j * 100 + (i + 2) * 7
            stop = start + kernel_size
            assert (y == np.arange(start, stop)).all()

    # test 2D slice with 2D indices
    idx = torch.stack([idx + i * 3 for i in range(num_channels)]).t()
    result = slicing.slice_kernels(X, idx, kernel_size)
    assert result.shape == (8, num_channels, kernel_size)

    for i, Y in enumerate(result.cpu().numpy()):
        for j, y in enumerate(Y):
            start = j * 103 + (i + 2) * 7
            stop = start + kernel_size
            assert (y == np.arange(start, stop)).all()

    # ensure that 2D slice with > 2D indices raises error
    idx = idx[:, :, None]
    with pytest.raises(ValueError) as exc:
        slicing.slice_kernels(X, idx, kernel_size)
    assert str(exc.value).startswith("Can't slice 2D array")

    # test 3D slice with 1D indices
    idx = idx[:, 0, 0]
    X = torch.stack([X + i * 1000 for i in range(len(idx))])
    result = slicing.slice_kernels(X, idx, kernel_size)
    for i, Y in enumerate(result.cpu().numpy()):
        for j, y in enumerate(Y):
            start = i * 1007 + j * 100 + 14
            stop = start + kernel_size
            assert (y == np.arange(start, stop)).all(), (i, j)

    # ensure 3D slice with wrong number of indices raises error
    with pytest.raises(ValueError) as exc:
        slicing.slice_kernels(X, idx[:-2], kernel_size)
    assert str(exc.value).startswith("Can't slice kernels from batch")

    # ensure 3D slice with >1D indices raises error
    idx = idx[:, None]
    with pytest.raises(ValueError) as exc:
        slicing.slice_kernels(X, idx, kernel_size)
    assert str(exc.value).startswith("Can't slice 3D array")

    # ensure >3D slice raises error
    idx = idx[:, 0]
    X = X[None]
    with pytest.raises(ValueError) as exc:
        slicing.slice_kernels(X, idx, kernel_size)
    assert str(exc.value).startswith("Can't slice kernels from tensor")


def test_sample_kernels_1D(kernel_size, num_channels):
    N = 8
    x = torch.arange(100)

    # make sure that we enforce that we have enough data to sample
    with pytest.raises(ValueError) as exc:
        slicing.sample_kernels(x[: kernel_size - 1], kernel_size, N)
    assert str(exc.value).startswith("Can't sample kernels of size")

    # make sure that we enforce that N is not None
    with pytest.raises(ValueError) as exc:
        slicing.sample_kernels(x, kernel_size)
    assert str(exc.value).startswith("Must specify number of kernels")

    # first test 1D behavior
    return_value = 7 * torch.arange(N)
    with patch("torch.randint", return_value=return_value) as mock:
        result = slicing.sample_kernels(x, kernel_size, N)
    mock.assert_called_once_with(100 - kernel_size, size=(N,))

    assert result.shape == (N, kernel_size)
    for i, y in enumerate(result.cpu().numpy()):
        start = i * 7
        stop = start + kernel_size
        assert (y == np.arange(start, stop)).all()

    # now 1D behavior without patching
    result = slicing.sample_kernels(x, kernel_size, N)
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

    # make sure that we enforce that we have enough data to sample
    with pytest.raises(ValueError) as exc:
        slicing.sample_kernels(X[:, : kernel_size - 1], kernel_size, N)
    assert str(exc.value).startswith("Can't sample kernels of size")

    # make sure that we enforce that N is not None
    with pytest.raises(ValueError) as exc:
        slicing.sample_kernels(X, kernel_size)
    assert str(exc.value).startswith("Must specify number of kernels")

    # make sure we enforce that min_val is not negative
    if max_center_offset is not None:
        with pytest.raises(ValueError) as exc:
            bad_kernel_size = int(X.shape[-1] // 2) - max_center_offset + 1
            slicing.sample_kernels(X, bad_kernel_size, N, max_center_offset)
        assert str(exc.value).startswith("Kernel size")

    # for non-coincident sampling, we'll request a 2D
    # matrix of sample indices from `torch.randint`
    # rather than 1, and we'll make sure to create a
    # return value that has different values for each
    # channel to ensure the correct response
    shape = (N,)
    return_value = 7 * torch.arange(N)
    if not coincident:
        return_value = torch.stack(
            [return_value + i for i in range(num_channels)]
        ).t()
        shape = (N, num_channels)

    # if we requested some padding between the
    # edge of the kernel and the center of the
    # timeseries, but our kernel isn't big enough
    # to accommodate it, this should raise an error
    if (
        max_center_offset is not None
        and max_center_offset < 0
        and kernel_size <= 2 * abs(max_center_offset)
    ):
        with pytest.raises(ValueError):
            slicing.sample_kernels(
                X, kernel_size, N, max_center_offset, coincident
            )
        return

    with patch("torch.randint", return_value=return_value) as mock:
        result = slicing.sample_kernels(
            X, kernel_size, N, max_center_offset, coincident
        )
    assert result.shape == (N, num_channels, kernel_size)
    result = result.cpu().numpy()

    if max_center_offset is None:
        min_val, max_val = 0, 100 - kernel_size
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


@pytest.fixture(params=[None, 1])
def N(request):
    return request.param


def test_sample_kernels_3D(kernel_size, num_channels, max_center_offset, N):
    batch_size = 8
    if N is not None:
        N = batch_size

    x = torch.arange(100)
    X = torch.stack([x + i * 100 for i in range(num_channels)])
    X = torch.stack([X + i * 1000 for i in range(batch_size)])

    # make sure that we enforce that we have enough data to sample
    with pytest.raises(ValueError) as exc:
        slicing.sample_kernels(X[:, :, : kernel_size - 1], kernel_size, N)
    assert str(exc.value).startswith("Can't sample kernels of size")

    # make sure we raise an error if we specify N
    # but it doesn't match the batch size of X
    with pytest.raises(ValueError) as exc:
        slicing.sample_kernels(X, kernel_size, batch_size - 1)
    assert str(exc.value).startswith(f"Can't sample {batch_size - 1} kernels")

    # make sure we enforce that min_val is not negative
    if max_center_offset is not None:
        with pytest.raises(ValueError) as exc:
            bad_kernel_size = int(X.shape[-1] // 2) - max_center_offset + 1
            slicing.sample_kernels(X, bad_kernel_size, N, max_center_offset)
        assert str(exc.value).startswith("Kernel size")

    # if we requested some padding between the
    # edge of the kernel and the center of the
    # timeseries, but our kernel isn't big enough
    # to accommodate it, this should raise an error
    if (
        max_center_offset is not None
        and max_center_offset < 0
        and kernel_size <= 2 * abs(max_center_offset)
    ):
        with pytest.raises(ValueError):
            slicing.sample_kernels(
                X, kernel_size, N, max_center_offset, coincident
            )
        return

    return_value = 7 * torch.arange(batch_size)
    with patch("torch.randint", return_value=return_value) as mock:
        result = slicing.sample_kernels(
            X, kernel_size, N, max_center_offset, coincident
        )
    assert result.shape == (batch_size, num_channels, kernel_size)
    result = result.cpu().numpy()

    if max_center_offset is None:
        min_val, max_val = 0, 100 - kernel_size
    elif max_center_offset == 0:
        min_val, max_val = 50 - kernel_size, 50
    elif max_center_offset == -4:
        min_val, max_val = 54 - kernel_size, 46
    else:
        min_val, max_val = 46 - kernel_size, 50
    mock.assert_called_with(min_val, max_val, size=(batch_size,))

    for i, sample in enumerate(result):
        for j, channel in enumerate(sample):
            if coincident:
                start = i * 1007 + j * 100
                stop = start + kernel_size
            else:
                start = i * 1007 + j * 101
                stop = start + kernel_size

            assert (channel == np.arange(start, stop)).all()
