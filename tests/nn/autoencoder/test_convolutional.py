import pytest
import torch
from torch import nn

from ml4gw.nn.autoencoder.convolutional import (
    ConvBlock,
    ConvolutionalAutoencoder,
)
from ml4gw.nn.autoencoder.skip_connection import (
    AddSkipConnect,
    ConcatSkipConnect,
)


@pytest.mark.parametrize("in_channels", [1, 2])
@pytest.mark.parametrize("encode_channels", [4, 8])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("stride", [1, 2])
def test_conv_block(in_channels, encode_channels, kernel_size, stride):
    block = ConvBlock(
        in_channels=in_channels,
        encode_channels=encode_channels,
        kernel_size=kernel_size,
        stride=stride,
    )
    batch, length = 4, 32
    x = torch.randn(batch, in_channels, length)

    encoded = block.encode(x)
    assert encoded.shape == (batch, encode_channels, length // stride)

    decoded = block.decode(encoded)
    assert decoded.shape[1] == in_channels


@pytest.mark.parametrize("in_channels", [1, 2])
@pytest.mark.parametrize(
    "encode_channels",
    [[8, 16], [16, 32, 64]],
)
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("decode_channels", [None, 3])
@pytest.mark.parametrize(
    "skip_connection",
    [None, AddSkipConnect(), ConcatSkipConnect()],
)
def test_convolutional_autoencoder_shapes(
    in_channels,
    encode_channels,
    kernel_size,
    stride,
    decode_channels,
    skip_connection,
):
    ae = ConvolutionalAutoencoder(
        in_channels=in_channels,
        encode_channels=encode_channels,
        kernel_size=kernel_size,
        stride=stride,
        decode_channels=decode_channels,
        skip_connection=skip_connection,
    )

    batch = 4
    input_size = 64
    x = torch.randn(batch, in_channels, input_size)

    # Encode in isolation
    encoded = ae.encode(x)
    assert encoded.shape[0] == batch
    assert encoded.shape[1] == encode_channels[-1]

    # Forward round-trip (preserves exact input length)
    out = ae(x)
    expected_out_channels = decode_channels or in_channels
    assert out.shape == (batch, expected_out_channels, input_size)


@pytest.mark.parametrize("groups", [1, 2])
def test_convolutional_autoencoder_groups(groups):
    in_channels = 2 * groups
    encode_channels = [4, 8]
    ae = ConvolutionalAutoencoder(
        in_channels=in_channels,
        encode_channels=encode_channels,
        kernel_size=3,
        stride=1,
        groups=groups,
    )

    x = torch.randn(4, in_channels, 32)
    out = ae(x)
    assert out.shape == (4, in_channels, 32)


def test_convolutional_autoencoder_custom_activations():
    ae = ConvolutionalAutoencoder(
        in_channels=2,
        encode_channels=[8, 16],
        kernel_size=3,
        activation=nn.ELU,
        output_activation=nn.Tanh,
    )
    x = torch.randn(4, 2, 32)
    out = ae(x)
    assert out.shape == (4, 2, 32)
    # Output activation is Tanh, values must be in [-1, 1]
    assert torch.all(out >= -1.0)
    assert torch.all(out <= 1.0)


@pytest.mark.parametrize(
    "skip_connection",
    [None, ConcatSkipConnect()],
)
def test_convolutional_autoencoder_gradient_backward(skip_connection):
    ae = ConvolutionalAutoencoder(
        in_channels=2,
        encode_channels=[8, 16],
        kernel_size=3,
        stride=2,
        skip_connection=skip_connection,
    )

    x = torch.randn(4, 2, 64)
    target = torch.randn(4, 2, 64)

    out = ae(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    # Verify gradients exist, are finite, and not NaN
    for name, param in ae.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            msg = f"Gradient has NaN/Inf for {name}"
            assert torch.isfinite(param.grad).all(), msg
