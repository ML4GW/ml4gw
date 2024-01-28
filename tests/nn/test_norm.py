import numpy as np
import pytest
import torch

from ml4gw.nn.norm import GroupNorm1D


@pytest.fixture(params=[1, 2, 3, 4])
def factor(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 4])
def num_groups(request):
    return request.param


def test_group_norm(num_groups, factor):

    with pytest.raises(ValueError):
        GroupNorm1D(1, 2)

    with pytest.raises(ValueError):
        GroupNorm1D(5, 3)

    num_channels = num_groups * factor
    x = torch.randn(128, num_channels, 1024)

    norm_layer = GroupNorm1D(num_channels, num_groups)
    normed = norm_layer(x).detach().numpy()
    assert normed.shape == x.shape

    torch_norm = torch.nn.GroupNorm(num_groups, num_channels)
    torch_normed = torch_norm(x).detach().numpy()
    assert np.allclose(torch_normed, normed, atol=1e-7)
