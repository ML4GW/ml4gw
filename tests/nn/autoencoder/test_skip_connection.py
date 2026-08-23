import pytest
import torch

from ml4gw.nn.autoencoder.skip_connection import (
    AddSkipConnect,
    ConcatSkipConnect,
    SkipConnection,
)


@pytest.mark.parametrize("in_channels", [1, 2, 4, 8])
def test_base_skip_connection(in_channels):
    sc = SkipConnection()
    assert sc.get_out_channels(in_channels) == in_channels

    batch, length_x, length_state = 4, 32, 28
    x = torch.randn(batch, in_channels, length_x)
    state = torch.randn(batch, in_channels, length_state)

    out = sc(x, state)
    assert out.shape == (batch, in_channels, length_state)


@pytest.mark.parametrize("in_channels", [1, 2, 4])
@pytest.mark.parametrize("length_x,length_state", [(32, 32), (32, 28), (28, 32)])
def test_add_skip_connect(in_channels, length_x, length_state):
    sc = AddSkipConnect()
    assert sc.get_out_channels(in_channels) == in_channels

    batch = 4
    x = torch.randn(batch, in_channels, length_x)
    state = torch.randn(batch, in_channels, length_state)

    out = sc(x, state)
    assert out.shape == (batch, in_channels, length_state)

    if length_x == length_state:
        assert torch.allclose(out, x + state)


@pytest.mark.parametrize("in_channels", [1, 2, 4])
@pytest.mark.parametrize("length_x,length_state", [(32, 32), (32, 28), (28, 32)])
def test_concat_skip_connect_single_group(in_channels, length_x, length_state):
    sc = ConcatSkipConnect(groups=1)
    assert sc.get_out_channels(in_channels) == 2 * in_channels

    batch = 4
    x = torch.randn(batch, in_channels, length_x)
    state = torch.randn(batch, in_channels, length_state)

    out = sc(x, state)
    assert out.shape == (batch, 2 * in_channels, length_state)


@pytest.mark.parametrize("groups", [2, 4])
def test_concat_skip_connect_multiple_groups(groups):
    channels_per_group = 3
    in_channels = groups * channels_per_group
    sc = ConcatSkipConnect(groups=groups)
    assert sc.get_out_channels(in_channels) == 2 * in_channels

    batch, length = 4, 32
    x = torch.randn(batch, in_channels, length)
    state = torch.randn(batch, in_channels, length)

    out = sc(x, state)
    assert out.shape == (batch, 2 * in_channels, length)

    # Verify channel interleaving
    x_splits = torch.split(x, groups, dim=1)
    state_splits = torch.split(state, groups, dim=1)
    expected_frags = [i for j in zip(x_splits, state_splits, strict=True) for i in j]
    expected_out = torch.cat(expected_frags, dim=1)
    assert torch.equal(out, expected_out)


def test_concat_skip_connect_invalid_channels_error():
    sc = ConcatSkipConnect(groups=3)
    x = torch.randn(2, 4, 16)  # 4 channels cannot be divided by 3 groups
    state = torch.randn(2, 4, 16)
    with pytest.raises(ValueError, match="cannot be divided evenly into 3 groups"):
        sc(x, state)
