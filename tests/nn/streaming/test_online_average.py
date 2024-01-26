from math import isclose

import numpy as np
import pytest
import torch

from ml4gw.nn.streaming import OnlineAverager
from ml4gw.utils.slicing import unfold_windows


@pytest.fixture(params=[1, 2, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def update_size(request):
    return request.param


@pytest.fixture(params=[2, 10])
def num_updates(request):
    return request.param


@pytest.fixture(params=[1, 2])
def num_channels(request):
    return request.param


@pytest.fixture
def validate_output(num_updates, update_size, num_channels):
    def f(start, stop, offset, output):
        expected = np.arange(start, stop)
        for channel in output:
            for n, (i, j) in enumerate(zip(expected, channel)):
                step = (offset * update_size + n) // update_size
                factor = min((step + 1) / num_updates, 1)
                assert isclose(i * factor, j, rel_tol=1e-6)

    return f


def test_online_averager(
    batch_size, update_size, num_updates, num_channels, validate_output
):
    averager = OnlineAverager(
        update_size=update_size,
        batch_size=batch_size,
        num_updates=num_updates,
        num_channels=num_channels,
    )

    # make a batch of overlapping aranged data such that
    # the online average is just the values themselves
    kernel_size = 100
    total_updates = 2 * batch_size - 1
    size = total_updates * update_size + kernel_size
    x = torch.arange(size)[None].type(torch.float32)
    x = unfold_windows(x, kernel_size, update_size)
    x = torch.repeat_interleave(x, num_channels, axis=1)
    assert x.size(0) == 2 * batch_size

    # initialize a blank initial snapshot
    state = averager.get_initial_state()

    # perform the first aggregation step
    stream, new_state = averager(x[:batch_size], state)

    # make sure the shapes are right
    expected_shape = (num_channels, update_size * batch_size)
    assert stream.shape == expected_shape
    assert new_state.shape == state.shape

    # now validate that the streamed value is correct
    start = kernel_size - update_size * num_updates
    stop = start + update_size * batch_size
    validate_output(start, stop, 0, stream.cpu().numpy())

    # now take the next step and confirm everything again
    stream, newer_state = averager(x[batch_size:], new_state)

    assert stream.shape == expected_shape
    assert newer_state.shape == state.shape

    start += update_size * batch_size
    stop = start + update_size * batch_size
    validate_output(start, stop, batch_size, stream.cpu().numpy())
