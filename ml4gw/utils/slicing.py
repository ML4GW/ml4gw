from typing import Optional, Union

import torch
from torchtyping import TensorType

# need this for flake8 compatibility
batch = time = channel = None  # noqa

TimeSeriesTensor = Union[TensorType["time"], TensorType["channel", "time"]]

BatchTimeSeriesTensor = Union[
    TensorType["batch", "time"], TensorType["batch", "channel", "time"]
]


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
        kernels = torch.arange(kernel_size).view(1, kernel_size, 1)
        kernels = kernels.repeat(len(x), 1, len(idx))
        kernels = (kernels + idx).transpose(1, 2)
        kernels = kernels.reshape(len(x), -1)
        x = torch.take_along_dim(x, kernels, axis=1)
        x = x.reshape(len(x), len(idx), kernel_size)
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
    else:
        raise ValueError(f"Can't slice array with {x.ndim} dimensions")


def sample_kernels(
    X: TimeSeriesTensor,
    N: int,
    kernel_size: int,
    max_center_offset: Optional[int] = None,
    coincident: bool = True,
) -> BatchTimeSeriesTensor:
    if X.ndim == 1:
        idx = torch.randint(len(X) - kernel_size, size=(N,))
        return slice_kernels(X, idx, kernel_size)

    center = int(X.shape[1] // 2)
    if max_center_offset is None:
        min_val, max_val = 0, X.shape[1]
    elif max_center_offset >= 0:
        min_val = center - max_center_offset - kernel_size
        max_val = center
    else:
        min_val = center + max_center_offset - kernel_size
        max_val = center - max_center_offset

    if coincident:
        shape = (N,)
    else:
        shape = (N, len(X))

    idx = torch.randint(min_val, max_val, size=shape)
    return slice_kernels(X, idx, kernel_size)
