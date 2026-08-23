import pytest
import torch

from ml4gw.nn.autoencoder.utils import match_size


@pytest.mark.parametrize("target_size", [8, 16, 23, 32])
@pytest.mark.parametrize("input_size", [8, 15, 16, 27, 32])
def test_match_size(target_size, input_size):
    batch, channels = 4, 2
    x = torch.randn(batch, channels, input_size)
    y = match_size(x, target_size)

    assert y.shape == (batch, channels, target_size)

    if target_size == input_size:
        assert torch.equal(x, y)
    elif target_size > input_size:
        diff = target_size - input_size
        left = int(diff // 2)
        assert torch.equal(y[..., left : left + input_size], x)
        if left > 0:
            assert torch.all(y[..., :left] == 0)
        right = diff - left
        if right > 0:
            assert torch.all(y[..., -right:] == 0)
    else:
        diff = input_size - target_size
        left = int(diff // 2)
        right = diff - left
        end = -right if right > 0 else None
        assert torch.equal(y, x[..., left:end])
