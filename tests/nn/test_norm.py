import pytest
import torch

from ml4gw.nn.norm import GroupNorm1D


@pytest.fixture(params=[1, 2, 3, 4])
def factor(request):
    return request.param


@pytest.fixture(params=[None, 1, 2, 3, 4])
def num_groups(request):
    return request.param


def update_weights(norm):
    optim = torch.optim.SGD(norm.parameters(), lr=1e-1)
    for i in range(10):
        optim.zero_grad()
        x, y = [torch.randn(8, norm.num_channels, 128) for _ in range(2)]
        y = 0.2 + 0.5 * y
        y_hat = norm(x)
        loss = ((y_hat - y) ** 2).mean()
        loss.backward()
        optim.step()
    return norm


def copy_weights(target, source):
    target.weight.requires_grad = False
    target.bias.requires_grad = False
    target.weight.copy_(source.weight.data[:, 0])
    target.bias.copy_(source.bias.data[:, 0])
    return target


def test_group_norm(num_groups, factor):
    with pytest.raises(ValueError):
        GroupNorm1D(1, 2)

    with pytest.raises(ValueError):
        GroupNorm1D(5, 3)

    if num_groups is None:
        num_channels = factor
    else:
        num_channels = num_groups * factor

    norm = GroupNorm1D(num_channels, num_groups)

    # update the norm layers weights so that
    # we have something interesting to compare
    norm = update_weights(norm)

    # copy learned parameters into normal groupnorm
    # and verify that outputs are similar
    ref = torch.nn.GroupNorm(norm.num_groups, norm.num_channels)
    ref = copy_weights(ref, norm)

    x = torch.randn(128, num_channels, 1024)
    x_ref = ref(x)
    x = norm(x)

    close = torch.isclose(x, x_ref, rtol=1e-6)
    num_wrong = (~close).sum()
    assert (num_wrong / x.numel()) < 0.01
