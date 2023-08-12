from unittest.mock import patch

import numpy as np
import pytest
import torch

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


@pytest.fixture
def X(num_kernels, stride, kernel_size, extra):
    num_samples = (num_kernels - 1) * stride + kernel_size
    if extra == -1:
        num_samples += stride - 1
    elif extra < stride:
        num_samples += extra

    x = torch.arange(num_samples)
    return torch.stack([i * num_samples + x for i in range(3)])


@pytest.fixture(params=[True, False])
def use_y(request):
    return request.param


@pytest.fixture
def Xy(X, use_y):
    y = X[-1] + 1 if use_y else None
    return X, y


@pytest.fixture
def validate_shape(kernel_size, batch_size, num_kernels):
    def f(i, x, length, y=None):
        if batch_size == 1 or (i + 1) < length:
            expected_batch = batch_size
        elif batch_size > num_kernels:
            expected_batch = num_kernels
        else:
            expected_batch = 2

        assert x.shape == (expected_batch, 3, kernel_size)
        if y is not None:
            assert y.shape == (expected_batch, kernel_size)

    return f


def test_in_memory_dataset_coincident_deterministic(
    Xy, num_kernels, kernel_size, stride, batch_size, validate_shape
):
    X, y = Xy
    if y is not None:
        # ensure that y has to be same length as x in time
        with pytest.raises(ValueError) as exc:
            dataset = InMemoryDataset(
                X,
                kernel_size,
                y=y[:-1],
                batch_size=batch_size,
                stride=stride,
                coincident=True,
                shuffle=False,
            )
        assert str(exc.value).startswith("Target timeseries")

    dataset = InMemoryDataset(
        X,
        kernel_size,
        y=y,
        batch_size=batch_size,
        stride=stride,
        coincident=True,
        shuffle=False,
    )
    # validate some attributes on the dataset
    assert dataset.num_kernels == num_kernels
    if y is None:
        assert dataset.y is None

    if batch_size == 1 or batch_size > num_kernels:
        assert len(dataset) == num_kernels
    else:
        assert len(dataset) == 3

    # now iterate through and make sure all the
    # produced arrays match our expectations
    num_samples = X.shape[-1]
    for i, x in enumerate(dataset):
        if y is not None:
            x, y_ = x
            y_ = y_.cpu().numpy()
        else:
            y_ = None

        validate_shape(i, x, len(dataset), y_)

        for j, sample in enumerate(x.cpu().numpy()):
            for k, channel in enumerate(sample):
                start = i * batch_size * stride + j * stride + k * num_samples
                stop = start + kernel_size
                expected = np.arange(start, stop).astype("float32")
                assert (channel == expected).all()

            # y should be equal to the last channel plus 1
            if y is not None:
                assert (y_[j] == (expected + 1)).all()

    assert (i + 1) == len(dataset)


@pytest.fixture(params=[1, 4])
def batches_per_epoch(request):
    return request.param


def test_in_memory_dataset_non_coincident_deterministic(
    X,
    num_kernels,
    kernel_size,
    stride,
    batch_size,
    validate_shape,
    batches_per_epoch,
):
    # ensure that we can't pass a target to a non-coincident dataset
    with pytest.raises(ValueError) as exc:
        dataset = InMemoryDataset(
            X,
            kernel_size,
            y=X[-1],
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            stride=stride,
            coincident=False,
            shuffle=False,
        )
    assert str(exc.value).startswith("Can't sample target array")

    # ensure that we have to pass batches_per_epoch
    with pytest.raises(ValueError) as exc:
        dataset = InMemoryDataset(
            X,
            kernel_size,
            batch_size=batch_size,
            stride=stride,
            coincident=False,
            shuffle=False,
        )
    assert str(exc.value).startswith("Must specify number of batches")

    if num_kernels < (batch_size * batches_per_epoch):
        with pytest.raises(ValueError):
            dataset = InMemoryDataset(
                X,
                kernel_size,
                batch_size=batch_size,
                stride=stride,
                coincident=False,
                shuffle=False,
                batches_per_epoch=batches_per_epoch,
            )
        return

    # create, validate attributes, validate returned values
    dataset = InMemoryDataset(
        X,
        kernel_size,
        batch_size=batch_size,
        stride=stride,
        coincident=False,
        shuffle=False,
        batches_per_epoch=batches_per_epoch,
    )
    assert dataset.num_kernels == num_kernels
    assert dataset.y is None
    assert len(dataset) == batches_per_epoch

    num_samples = X.shape[-1]
    for i, x in enumerate(dataset):
        assert x.shape == (batch_size, 3, kernel_size)

        for j, sample in enumerate(x.cpu().numpy()):
            sample_idx = i * batch_size + j
            for k, channel in enumerate(sample):
                sample_step = sample_idx // (num_kernels ** (2 - k))
                start = k * num_samples + sample_step * stride
                stop = start + kernel_size
                expected = np.arange(start, stop).astype("float32")
                assert (channel == expected).all()

    assert (i + 1) == len(dataset)


def test_in_memory_dataset_coincident_stochastic(
    Xy, num_kernels, kernel_size, stride, batch_size, validate_shape
):
    # don't do some of the checks on errors or attributes
    # since the shuffle flag doesn't affect any of these
    X, y = Xy
    dataset = InMemoryDataset(
        X,
        kernel_size,
        y=y,
        batch_size=batch_size,
        stride=stride,
        coincident=True,
        shuffle=True,
    )

    # patch randperm and call __iter__ to make sure
    # we know exactly what the iteration order will be
    idx = torch.randperm(num_kernels)
    with patch("torch.randperm", return_value=idx):
        data_it = iter(dataset)
    idx = idx.cpu().numpy()

    # now iterate through the initialized iterator and
    # use the sampled indices to find where in the
    # arange array we ought to be
    num_samples = X.shape[-1]
    for i in range(len(dataset)):
        x = next(data_it)
        if y is not None:
            x, y_ = x
            y_ = y_.cpu().numpy()
        else:
            y_ = None

        validate_shape(i, x, len(dataset), y_)
        for j, sample in enumerate(x.cpu().numpy()):
            for k, channel in enumerate(sample):
                sample_idx = i * batch_size + j
                start_idx = idx[sample_idx]
                start = start_idx * stride + k * num_samples
                stop = start + kernel_size
                expected = np.arange(start, stop).astype("float32")
                assert (channel == expected).all(), (i, j, k)

            if y is not None:
                assert (y_[j] == (expected + 1)).all()

    with pytest.raises(StopIteration):
        next(data_it)


def test_in_memory_dataset_non_coincident_stochastic(
    X,
    num_kernels,
    kernel_size,
    stride,
    batch_size,
    validate_shape,
    batches_per_epoch,
):
    # skip checks besides this one since we checked
    # errors and attributes earlier
    if num_kernels < (batch_size * batches_per_epoch):
        with pytest.raises(ValueError):
            dataset = InMemoryDataset(
                X,
                kernel_size,
                batch_size=batch_size,
                stride=stride,
                coincident=False,
                shuffle=False,
                batches_per_epoch=batches_per_epoch,
            )
        return

    dataset = InMemoryDataset(
        X,
        kernel_size,
        batch_size=batch_size,
        stride=stride,
        coincident=False,
        shuffle=True,
        batches_per_epoch=batches_per_epoch,
    )

    # same plan as the coincident test: patch the
    # random generator we're going to use in __iter__
    # so we can predict the iteration order
    samples = batches_per_epoch * batch_size
    idx = torch.randint(num_kernels, size=(samples, 3))
    with patch("torch.randint", return_value=idx) as mock:
        data_it = iter(dataset)
    mock.assert_called_once_with(
        num_kernels, size=(samples, 3), device=dataset.X.device
    )
    idx = idx.cpu().numpy()

    # use the sampled indices to make sure the iteration
    # order matches our expectations
    num_samples = X.shape[-1]
    for i in range(len(dataset)):
        x = next(data_it)
        assert x.shape == (batch_size, 3, kernel_size)

        for j, sample in enumerate(x.cpu().numpy()):
            for k, channel in enumerate(sample):
                sample_idx = i * batch_size + j
                start_idx = idx[sample_idx, k]
                start = k * num_samples + start_idx * stride
                stop = start + kernel_size
                expected = np.arange(start, stop).astype("float32")
                assert (channel == expected).all()

    with pytest.raises(StopIteration):
        next(data_it)
