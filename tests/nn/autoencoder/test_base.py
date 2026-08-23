import pytest
import torch
from torch import Tensor, nn

from ml4gw.nn.autoencoder.base import Autoencoder
from ml4gw.nn.autoencoder.skip_connection import AddSkipConnect


class DummyBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.decoder = nn.Conv1d(out_channels, in_channels, kernel_size=1)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)


def test_empty_autoencoder():
    ae = Autoencoder()
    assert len(ae.blocks) == 0

    x = torch.randn(4, 2, 32)
    encoded = ae.encode(x)
    assert torch.equal(encoded, x)

    decoded = ae.decode(x)
    assert torch.equal(decoded, x)

    out = ae(x)
    assert torch.equal(out, x)


def test_single_block_autoencoder():
    ae = Autoencoder()
    ae.blocks.append(DummyBlock(2, 4))

    x = torch.randn(4, 2, 32)

    # Isolated encode
    encoded = ae.encode(x)
    assert encoded.shape == (4, 4, 32)

    # Isolated decode
    decoded = ae.decode(encoded)
    assert decoded.shape == (4, 2, 32)

    # Forward round-trip
    out = ae(x)
    assert out.shape == x.shape

    # Encode with return_states
    enc, states = ae.encode(x, return_states=True)
    assert torch.equal(enc, encoded)
    assert len(states) == 0  # for single block, states[:-1] is empty


def test_multi_block_autoencoder_without_skip():
    ae = Autoencoder()
    ae.blocks.append(DummyBlock(2, 4))
    ae.blocks.append(DummyBlock(4, 8))
    ae.blocks.append(DummyBlock(8, 16))

    x = torch.randn(4, 2, 32)

    # Encode and return states
    enc, states = ae.encode(x, return_states=True)
    assert enc.shape == (4, 16, 32)
    assert len(states) == 2
    assert states[0].shape == (4, 4, 32)
    assert states[1].shape == (4, 8, 32)

    # Decode in isolation
    decoded = ae.decode(enc)
    assert decoded.shape == (4, 2, 32)

    # Forward
    out = ae(x)
    assert out.shape == x.shape


def test_autoencoder_with_skip_connection():
    skip = AddSkipConnect()
    ae = Autoencoder(skip_connection=skip)
    # Using blocks with equal in/out channels so AddSkipConnect can add x + state
    ae.blocks.append(DummyBlock(4, 4))
    ae.blocks.append(DummyBlock(4, 4))
    ae.blocks.append(DummyBlock(4, 4))

    x = torch.randn(4, 4, 32)

    # Decode without states when skip_connection is set should raise
    enc, states = ae.encode(x, return_states=True)
    with pytest.raises(ValueError, match="Must pass intermediate states"):
        ae.decode(enc, states=None)

    # Decode with wrong number of states should raise
    with pytest.raises(ValueError, match="Passed 1 intermediate states, expected 2"):
        ae.decode(enc, states=[states[0]])

    # Decode with correct states
    decoded = ae.decode(enc, states=states)
    assert decoded.shape == (4, 4, 32)

    # Forward works automatically
    out = ae(x)
    assert out.shape == x.shape
