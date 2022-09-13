import numpy as np
import pytest

from ml4gw.dataloading.in_memory_dataset import InMemoryDataset


@pytest.fixture(params=[1, 10])
def num_kernels(request):
    return request.param


@pytest.fixture
def sample_rate():
    return 128


@pytest.fixture(params=[8])
def kernel_size(request):
    return request.param


@pytest.fixture(params=[1, 2])
def stride(request):
    return request.param


@pytest.fixture(params=[0, 1, -1])
def extra(request):
    return request.param


@pytest.fixture(params=[1, 4])
def batch_size(request):
    return request.param


def test_in_memory_dataset_coincident_deterministic(
    num_kernels, sample_rate, kernel_size, stride, extra, batch_size
):
    num_samples = (num_kernels - 1) * stride + kernel_size
    if extra == -1:
        num_samples += stride - 1
    elif extra < stride:
        num_samples += extra

    x = np.arange(num_samples)
    X = np.stack([i * num_samples + x for i in range(3)])

    dataset = InMemoryDataset(
        X,
        kernel_size,
        batch_size=batch_size,
        stride=stride,
        coincident=True,
        shuffle=False,
    )
    assert dataset.num_kernels == num_kernels
    assert dataset.y is None

    if batch_size == 1 or batch_size > num_kernels:
        assert len(dataset) == num_kernels
    else:
        assert len(dataset) == 3

    for i, x in enumerate(dataset):
        assert x.shape[1:] == (3, kernel_size)
        if batch_size == 1 or (i + 1) < len(dataset):
            assert x.shape[0] == batch_size
        elif batch_size > num_kernels:
            assert x.shape[0] == num_kernels
        else:
            assert x.shape[0] == 2

        for j, sample in enumerate(x.cpu().numpy()):
            for k, channel in enumerate(sample):
                start = i * batch_size * stride + j * stride + k * num_samples
                stop = start + kernel_size
                expected = np.arange(start, stop).astype("float32")
                assert (channel == expected).all()

    assert (i + 1) == len(dataset)
