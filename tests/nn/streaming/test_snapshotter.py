import numpy as np
import pytest
import torch

from ml4gw.nn.streaming import Snapshotter


@pytest.fixture(params=[1, 4, 100])
def snapshot_size(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def stride_size(request):
    return request.param


@pytest.fixture(params=[[1], 1, [4], 4, [1, 4], [1, 2, 4]])
def channels(request):
    return request.param


@pytest.fixture
def snapshotter(snapshot_size, stride_size, batch_size, channels):
    return Snapshotter(snapshot_size, stride_size, batch_size, channels)


def test_snapshotter(snapshot_size, stride_size, batch_size, channels):
    if isinstance(channels, int):
        num_channels = channels
        channels = [num_channels]
        channels_per_snapshot = None
    else:
        num_channels = sum(channels)
        channels_per_snapshot = channels

    # test that we don't allow for snapshotting
    # when the stride is longer than the kernel,
    # since there's no real stateful behavior
    # in this case
    if stride_size >= snapshot_size:
        with pytest.raises(ValueError):
            snapshotter = Snapshotter(
                num_channels,
                snapshot_size,
                stride_size,
                batch_size,
                channels_per_snapshot,
            )
        return

    # now make sure we test for agreement between
    # num_channels and any way of breaking up the channels
    if channels_per_snapshot is not None:
        with pytest.raises(ValueError):
            snapshotter = Snapshotter(
                num_channels + 1,
                snapshot_size,
                stride_size,
                batch_size,
                channels_per_snapshot,
            )

    snapshotter = Snapshotter(
        num_channels,
        snapshot_size,
        stride_size,
        batch_size,
        channels_per_snapshot,
    )

    # now run an input through as a new sequence and
    # make sure we get the appropriate number of outputs
    offset = torch.arange(num_channels)[:, None] * snapshot_size

    update_size = stride_size * batch_size
    snapshot = snapshotter.get_initial_state() + 1
    snapshot = torch.cumsum(snapshot, axis=-1) - 1
    snapshot += offset

    update = torch.arange(update_size) + 1 + snapshot[0, -1]
    update = update.view(1, -1).repeat(num_channels, 1)
    update += offset

    outputs = snapshotter(update, snapshot)
    outputs = [i.cpu().numpy() for i in outputs]
    new_snapshot = outputs.pop(-1)

    # make sure we have as many outputs as we
    # specified number of channel groups
    assert len(outputs) == len(channels)

    # now go through and make sure that each
    # of the windows in each batch contains
    # the expected data
    offset = 0
    for k, (output, channel_dim) in enumerate(zip(outputs, channels)):
        expected = (batch_size, channel_dim, snapshot_size)
        assert output.shape == tuple(expected)

        for i, row in enumerate(output):
            for j, channel in enumerate(row):
                start = j * snapshot_size + i * stride_size
                stop = start + snapshot_size
                expected = np.arange(start, stop) + offset
                np.testing.assert_equal(channel, expected)
        offset += channel_dim * snapshot_size

    expected_size = snapshot_size - stride_size
    assert new_snapshot.shape == (num_channels, expected_size)
    for i, channel in enumerate(new_snapshot):
        start = update_size + i * snapshot_size
        stop = start + expected_size
        expected = np.arange(start, stop)
        np.testing.assert_equal(channel, expected)
