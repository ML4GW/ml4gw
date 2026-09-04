import pytest
import torch

from ml4gw.nn.autoencoder.utils import match_size


@pytest.mark.parametrize("target_size", [8, 16, 23, 32])
@pytest.mark.parametrize("input_size", [8, 15, 16, 27, 32])
def test_match_size(target_size, input_size):
    x = torch.arange(1, input_size + 1)
    y = match_size(x, target_size)

    assert y.shape == (target_size,)

    if target_size == input_size:
        assert torch.equal(x, y)
    elif target_size > input_size:
        matches = y.unfold(-1, input_size, 1).eq(x).all(dim=-1)
        locations = matches.nonzero(as_tuple=False)
        assert len(locations) == 1
        left = locations.item()
        right = target_size - input_size - left
        assert abs(left - right) <= 1
        assert torch.count_nonzero(y[:left]) == 0
        assert torch.count_nonzero(y[target_size - right :]) == 0
    else:
        matches = x.unfold(-1, target_size, 1).eq(y).all(dim=-1)
        locations = matches.nonzero(as_tuple=False)
        assert len(locations) == 1
        left = locations.item()
        right = input_size - target_size - left
        assert abs(left - right) <= 1
