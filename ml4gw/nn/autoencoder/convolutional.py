from collections.abc import Callable, Sequence
from typing import Optional

import torch

from ml4gw.nn.autoencoder.base import Autoencoder
from ml4gw.nn.autoencoder.skip_connection import SkipConnection
from ml4gw.nn.autoencoder.utils import match_size

Module = Callable[[...], torch.nn.Module]


class ConvBlock(Autoencoder):
    def __init__(
        self,
        in_channels: int,
        encode_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU(),
        norm: Module = torch.nn.BatchNorm1d,
        decode_channels: Optional[int] = None,
        output_activation: Optional[torch.nn.Module] = None,
        skip_connection: Optional[SkipConnection] = None,
    ) -> None:
        super().__init__(skip_connection=None)

        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) // 2)
        self.stride = stride

        out_channels = encode_channels * groups
        self.encode_layer = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=False,
            groups=groups,
        )

        decode_channels = decode_channels or in_channels
        in_channels = encode_channels * groups
        if skip_connection is not None:
            in_channels = skip_connection.get_out_channels(in_channels)
        self.decode_layer = torch.nn.ConvTranspose1d(
            in_channels,
            decode_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=False,
            groups=groups,
        )

        self.activation = activation
        if output_activation is not None:
            self.output_activation = output_activation
        else:
            self.output_activation = activation

        self.encode_norm = norm(out_channels)
        self.decode_norm = norm(decode_channels)

    def encode(self, X):
        X = self.encode_layer(X)
        X = self.encode_norm(X)
        return self.activation(X)

    def decode(self, X):
        X = self.decode_layer(X)
        X = self.decode_norm(X)
        return self.output_activation(X)


class ConvolutionalAutoencoder(Autoencoder):
    """
    Build a stack of convolutional autoencoder layer
    blocks. The output of each decoder layer will
    match the shape of the input to its corresponding
    encoder layer, except for the last decoder which
    can have an arbitrary number of channels specified
    by `decode_channels`.

    All layers also share the same `activation` except
    for the last decoder layer, which can have an
    arbitrary `output_activation`.
    """

    def __init__(
        self,
        in_channels: int,
        encode_channels: Sequence[int],
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU(),
        output_activation: Optional[torch.nn.Module] = None,
        norm: Module = torch.nn.BatchNorm1d,
        decode_channels: Optional[int] = None,
        skip_connection: Optional[SkipConnection] = None,
    ) -> None:
        # TODO: how to do this dynamically? Maybe the base
        # architecture looks for overlapping arguments between
        # this and the skip connection class and then provides them?
        # if skip_connection is not None:
        #     skip_connection = skip_connection(groups)
        super().__init__(skip_connection=skip_connection)

        output_activation = output_activation or activation
        for i, channels in enumerate(encode_channels):
            # All intermediate layers should decode to
            # the same number of channels. The last decoder
            # should decode to whatever number of channels
            # was specified, even if it's `None` (in which
            # case it will just be in_channels anyway)
            decode = in_channels if i else decode_channels

            # don't have the middle layer skip to itself
            # TODO: wait I don't think this makes sense.
            # j = len(encode_channels) - 1 - i
            # connect = skip_connection if j else None
            connect = skip_connection

            # all intermediate layers should use the same
            # activation. Only the last decoder should have
            # a potentially different activation
            out_act = None if i else output_activation

            block = ConvBlock(
                in_channels,
                channels,
                kernel_size,
                stride,
                groups,
                activation=activation,
                norm=norm,
                decode_channels=decode,
                skip_connection=connect,
                output_activation=out_act,
            )
            self.blocks.append(block)
            in_channels = channels * groups

    def decode(self, *X, states=None, input_size: Optional[int] = None):
        X = super().decode(*X, states=states)
        if input_size is not None:
            return match_size(X, input_size)
        return X

    def forward(self, X):
        input_size = X.size(-1)
        X = super().forward(X)
        return match_size(X, input_size)
