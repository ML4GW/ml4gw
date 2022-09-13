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


@pytest.fixture
def X(num_kernels, stride, kernel_size, extra):
    num_samples = (num_kernels - 1) * stride + kernel_size
    if extra == -1:
        num_samples += stride - 1
    elif extra < stride:
        num_samples += extra

    x = np.arange(num_samples)
    return np.stack([i * num_samples + x for i in range(3)])


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
    assert dataset.num_kernels == num_kernels
    if y is None:
        assert dataset.y is None

    if batch_size == 1 or batch_size > num_kernels:
        assert len(dataset) == num_kernels
    else:
        assert len(dataset) == 3

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
        with pytest.raises(ValueError) as exc:
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
