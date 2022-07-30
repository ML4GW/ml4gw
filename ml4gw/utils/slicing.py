from typing import Optional, Union

import torch
from torchtyping import TensorType

# need to define these for flake8 compatibility
batch = time = channel = None  # noqa

TimeSeriesTensor = Union[TensorType["time"], TensorType["channel", "time"]]

BatchTimeSeriesTensor = Union[
    TensorType["batch", "time"], TensorType["batch", "channel", "time"]
]


def slice_kernels_via_concat(x, idx, kernel_size):
    if x.ndim == 1:
        output = torch.zeros((len(idx), kernel_size))
        for i, j in enumerate(idx):
            output[i] = x[j : j + kernel_size]
        return output
    elif x.ndim == 2 and idx.ndim == 1:
        output = torch.zeros((len(idx), len(x), kernel_size))
        for i, j in enumerate(idx):
            output[i] = x[:, j : j + kernel_size]
        return output
    elif x.ndim == 2 and idx.ndim == 1:
        output = torch.zeros((len(idx), len(x), kernel_size))
        for i, row in enumerate(idx):
            for j, val in enumerate(row):
                output[i, j] = x[val : val + kernel_size]
        return output


def slice_kernels(
    x: TimeSeriesTensor, idx: TensorType[..., torch.int64], kernel_size: int
) -> BatchTimeSeriesTensor:
    if x.ndim == 1:
        if idx.ndim != 1:
            raise ValueError(
                f"idx tensor has {idx.ndim} dimensions, expected 1"
            )

        # this is a one dimensional array that we want
        # to select kernels from beginning at each of
        # the specified indices
        kernels = torch.arange(kernel_size).view(kernel_size, 1)
        kernels = kernels.repeat(1, len(idx))
        kernels = (kernels + idx).t()
        return torch.take(x, kernels)
    elif x.ndim == 2 and idx.ndim == 1:
        # this is a multi-channel timeseries and we want
        # to select a set of kernels from the channels
        # coincidentally

        # 1 x kernel_size x 1
        kernels = torch.arange(kernel_size).view(1, kernel_size, 1)

        # channels x kernel_size x batch_size
        kernels = kernels.repeat(len(x), 1, len(idx))
        kernels += idx

        # channels x batch_size x kernel_size
        kernels = kernels.transpose(1, 2)

        # channels x (batch_size * kernel_size)
        kernels = kernels.reshape(len(x), -1)

        # channels x (batch_size * kernel_size)
        # (only this time it's data from x)
        x = torch.take_along_dim(x, kernels, axis=1)

        # channels x batch_size x kernel_size
        x = x.reshape(len(x), len(idx), kernel_size)

        # batch_size x channels x kernel_size
        return x.transpose(0, 1).contiguous()
    elif x.ndim == 2 and idx.ndim == 2:
        # this is a multi-channel timeseries and we want
        # to select _different_ kernels from each channel
        if len(x) != idx.shape[1]:
            raise ValueError(
                "Can't slice array with shape {} with indices "
                "with shape {}".format(x.shape, idx.shape)
            )
        kernels = torch.arange(kernel_size).view(kernel_size, 1, 1)
        kernels = kernels.repeat(1, len(idx), len(x))
        kernels = (kernels + idx).transpose(0, 2)
        kernels = kernels.reshape(len(x), -1)
        x = torch.take_along_dim(x, kernels, axis=1)
        x = x.reshape(len(x), len(idx), kernel_size)
        return x.transpose(0, 1).contiguous()
    elif x.ndim == 2:
        raise ValueError(
            f"Can't slice 2D array with indices with {idx.ndim} dimensions"
        )
    elif x.ndim == 3:
        if idx.ndim != 1:
            raise ValueError(
                f"idx tensor has {idx.ndim} dimensions for slicing "
                "tensor with 3 dimensions, expected 1"
            )


def sample_kernels(
    X: TimeSeriesTensor,
    N: int,
    kernel_size: int,
    max_center_offset: Optional[int] = None,
    coincident: bool = True,
) -> BatchTimeSeriesTensor:
    if X.shape[-1] < kernel_size:
        raise ValueError(
            "Can't sample kernels of size {} from "
            "tensor with shape {}".format(kernel_size, X.shape)
        )

    if X.ndim == 1:
        idx = torch.randint(len(X) - kernel_size, size=(N,))
        return slice_kernels(X, idx, kernel_size)

    center = int(X.shape[1] // 2)
    if max_center_offset is None:
        min_val, max_val = 0, X.shape[1]
    elif max_center_offset >= 0:
        min_val = center - max_center_offset - kernel_size
        max_val = center
    elif kernel_size <= 2 * abs(max_center_offset):
        raise ValueError(
            "Negative center offset value {} is too large "
            "for request kernel size {}".format(max_center_offset, kernel_size)
        )
    else:
        min_val = center - max_center_offset - kernel_size
        max_val = center + max_center_offset

    if coincident:
        shape = (N,)
    else:
        shape = (N, len(X))

    idx = torch.randint(min_val, max_val, size=shape)
    return slice_kernels(X, idx, kernel_size)
