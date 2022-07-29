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
