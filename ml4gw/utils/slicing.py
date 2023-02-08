from typing import Optional, Union

import torch
from torch.nn.functional import unfold
from torchtyping import TensorType

# need to define these for flake8 compatibility
batch = time = channel = None  # noqa

TimeSeriesTensor = Union[TensorType["time"], TensorType["channel", "time"]]

BatchTimeSeriesTensor = Union[
    TensorType["batch", "time"], TensorType["batch", "channel", "time"]
]


def unfold_windows(
    x: torch.Tensor,
    window_size: int,
    stride: int,
    drop_last: bool = True,
):
    """Unfold a timeseries into windows

    Args:
        x:
            The timeseries to unfold. Can have shape
            `(batch_size, num_channels, length * sample_rate)`,
            `(num_channels, length * sample_rate)`, or
            `(length * sample_rate)`
        window_size:
            The size of the windows to unfold from x
        stride:
            The stride between windows
        drop_last:
            If true, does not return the remainder that exists
            when the timeseries cannot be evenly broken up into
            windows

    Returns:
       A tensor of shape
       `(num_windows, batch_size, num_channels, kernel_size)`,
       `(num_windows, num_channels, kernel_size)`, or
       `(num_windows, kernel_size)` depending on whether the
       input tensor is 3D, 2D, or 1D

       If `drop_last` is false, returns the remainder of the
       timeseries, shaped to be compatible with the returned
       unfolded tensor
    """

    num_windows = (x.shape[-1] - window_size) // stride + 1
    remainder = x.shape[-1] - window_size - (num_windows - 1) * stride
    if remainder == 0:
        if not drop_last:
            # create an empty tensor for consistency
            shape = list(x.shape)
            shape[-1] = 0
            remainder = torch.zeros(shape, dtype=x.dtype, device=x.device)
    else:
        # separate x from its remainder _regardless_ of whether
        # we end up returning it or not
        x, remainder = torch.split(
            x, [x.shape[-1] - remainder, remainder], dim=-1
        )

    reshape = list(x.shape[:-1])
    if x.ndim == 1:
        x = x[None, None, None, :]
    elif x.ndim == 2:
        x = x[None, :, None, :]
    elif x.ndim == 3:
        x = x[:, :, None, :]

    x = unfold(x, (1, num_windows), dilation=(1, stride))
    reshape += [num_windows, -1]
    x = x.reshape(*reshape)
    x = x.transpose(1, -2).transpose(0, 1)

    if not drop_last:
        return x, remainder[None]
    return x


def slice_kernels(
    x: Union[TimeSeriesTensor, TensorType["batch", "channel", "time"]],
    idx: TensorType[..., torch.int64],
    kernel_size: int,
) -> BatchTimeSeriesTensor:
    """Slice kernels from single or multichannel timeseries

    Given a 1D timeseries or a 2D tensor representing a
    multichannel timeseries, slice kernels of a given size
    from the timeseries starting at the indicated indices.
    Returns a batch of 1D or 2D kernels, and so will have
    one more dimension than `x`.

    Args:
        x: The timeseries tensor to slice kernels from
        idx:
            The indices in `x` of the first sample of each
            kernel. If `x` is 1D, `idx` must be 1D as well.
            If `x` is 2D and `idx` is 1D, `idx` is assumed
            to represent the first index of the kernels sliced
            from _all_ channels (i.e. the channels are sliced
            coincidentally). If `x` is 2D and `idx` is also 2D,
            `idx` should have shape `(batch_size, num_channels)`,
            and its values are assumed to represent the first index
            of the kernel sliced from each channel _independently_.
            If `x` is 3D, `idx` _must_ be 1D, and have the same length
            as `x`. In this case, it is assumed that the elements of
            `idx` represent the starting index in the last dimension
            of `x` from which to sample a batch of kernels
            coincidentally among the channels.
        kernel_size:
            The length of the kernels to slice from the timeseries
    Returns:
        A tensor of shape `(batch_size, kernel_size)` if `x` is
        1D and `(batch_size, num_channels, kernel_size)` if `x`
        is 2D, where `batch_size = idx.shape[0]` and
        `num_channels = x.shape[0]` if `x` is 2D.
    """

    # create the indices all the slices will be built around,
    # and ensure they live on the appropriate device
    kernels = torch.arange(kernel_size, device=x.device)
    idx = idx.to(kernels.device)

    # TODO: add try-catches aroud the actual slicing operations
    # to catch out-of-range index errors and raise with a
    # standardized error that's very explicit
    if x.ndim == 1:
        if idx.ndim != 1:
            raise ValueError(
                f"idx tensor has {idx.ndim} dimensions, expected 1"
            )

        # this is a one dimensional array that we want
        # to select kernels from beginning at each of
        # the specified indices
        kernels = kernels.view(kernel_size, 1)
        kernels = kernels.repeat(1, len(idx))
        kernels = (kernels + idx).t()
        return torch.take(x, kernels)
    elif x.ndim == 2 and idx.ndim == 1:
        # this is a multi-channel timeseries and we want
        # to select a set of kernels from the channels
        # coincidentally

        # channels x batch_size x kernel_size
        kernels = kernels.view(1, 1, kernel_size)
        kernels = kernels.repeat(len(x), len(idx), 1)
        kernels += idx.view(1, -1, 1)

        # channels x (batch_size * kernel_size)
        kernels = kernels.reshape(len(x), -1)

        # channels x (batch_size * kernel_size)
        # (only this time it's data from x)
        x = torch.take_along_dim(x, kernels, axis=1)

        # channels x batch_size x kernel_size
        x = x.reshape(len(x), len(idx), kernel_size)

        # batch_size x channels x kernel_size
        return x.transpose(0, 1)
    elif x.ndim == 2 and idx.ndim == 2:
        # this is a multi-channel timeseries and we want
        # to select _different_ kernels from each channel
        if len(x) != idx.shape[1]:
            raise ValueError(
                "Can't slice array with shape {} with indices "
                "with shape {}".format(x.shape, idx.shape)
            )

        # batch_size x num_channels x kernel_size
        kernels = kernels.view(1, 1, kernel_size)
        kernels = kernels.repeat(len(idx), len(x), 1)
        kernels += idx[:, :, None]

        # num_channels x (batch_size * kernel_size)
        kernels = kernels.transpose(0, 1)
        kernels = kernels.reshape(len(x), -1)

        # num_channels x (batch_size * kernel_size)
        # sample a batch's worth of kernels for each channel
        x = torch.take_along_dim(x, kernels, axis=1)

        # num_channels x batch_size x kernel_size
        x = x.reshape(len(x), len(idx), kernel_size)

        # batch_size x num_channels x kernel_size
        return x.transpose(0, 1)
    elif x.ndim == 2:
        raise ValueError(
            f"Can't slice 2D array with indices with {idx.ndim} dimensions"
        )
    elif x.ndim == 3 and idx.ndim == 1:
        # slice a single kernel coincidentally from a batch
        # of multichannel timeseries
        if len(idx) != len(x):
            raise ValueError(
                "Can't slice kernels from batch of length {} "
                "using indices of length {}".format(len(x), len(idx))
            )

        # batch_size x kernel_size
        kernels = kernels.view(1, -1)
        kernels = kernels.repeat(len(idx), 1)
        kernels += idx.view(-1, 1)

        # batch_size x num_channels x kernel_size
        kernels = kernels.view(-1, 1, kernel_size)
        kernels = kernels.repeat(1, x.shape[1], 1)

        return torch.take_along_dim(x, kernels, axis=-1)
    elif x.ndim == 3:
        raise ValueError(
            f"Can't slice 3D array with indices with {idx.ndim} dimensions"
        )
    else:
        raise ValueError(
            f"Can't slice kernels from tensor with shape {x.shape}"
        )


def sample_kernels(
    X: TimeSeriesTensor,
    kernel_size: int,
    N: Optional[int] = None,
    max_center_offset: Optional[int] = None,
    coincident: bool = True,
) -> BatchTimeSeriesTensor:
    """Randomly sample kernels from a single or multichannel timeseries

    For a tensor representing one or multiple channels of
    timeseries data, randomly slice kernels of a fixed
    length from the timeseries. If `X` is 1D, kernels will
    be sampled uniformly from `X`. If `X` is 2D, kernels
    will be sampled from the first dimension of `X` (assumed
    to be the time dimension) in a manner that depends on the
    values of the `max_center_offset` and `coincident` kwargs.
    If `X` is 3D, one kernel will be sampled coincidentally from
    each element along the 0th axis of `X`. In this case, `N` must
    either be `None` or be equal to `len(X)`.

    Args:
        X: The timeseries tensor from which to sample kernels
        kernel_size: The size of the kernels to sample
        N:
            The number of kernels to sample. Can be left as
            `None` if `X` is 3D, otherwise must be specified
        max_center_offeset:
            If `X` is 2D, this indicates the maximum distance
            from the center of the timeseries the edge of
            sampled kernels may fall. If left as `None`, kernels
            will be sampled uniformly across all of `X`'s time
            dimension. If greater than 0, defines the maximum
            distance that the rightmost edge of the kernel may
            fall from the center of the timeseries (the leftmost
            edge will always be sampled such that the center of
            the timeseries falls within or afer the kernel). If
            equal to 0, every kernel sampled will contain the center
            of the timeseries, which may fall anywhere within the
            kernel with uniform probability. If less than 0, defines
            the minimum distance that the center of the timeseries
            must fall from either edge of the kernel. If `X` is
            1D, this argument is ignored.
        coincident:
            If `X` is 2D, determines whether the individual channels
            of `X` sample the same kernels or different kernels
            independently, i.e. whether the channels of each batch
            element in the output will contain coincident data. If
            `X` is 1D, this argument is ignored.
    Returns:
        A batch of sampled kernels. If `X` is 1D, this will have
        shape `(N, kernel_size)`. If `X` is 2D, this will have
        shape `(N, num_channels, kernel_size)`, where
        `num_channels = X.shape[0]`.
    """

    if X.shape[-1] < kernel_size:
        raise ValueError(
            "Can't sample kernels of size {} from "
            "tensor with shape {}".format(kernel_size, X.shape)
        )
    elif X.ndim > 3:
        raise ValueError(
            f"Can't sample kernels from tensor with {X.ndim} dimensions"
        )
    elif X.ndim < 3 and N is None:
        raise ValueError(
            "Must specify number of kernels N if X "
            "has fewer than 3 dimensions"
        )
    elif X.ndim == 3 and N is not None and N != len(X):
        raise ValueError(
            "Can't sample {} kernels from 3D tensor "
            "with batch dimension {}".format(N, len(X))
        )

    if X.ndim == 1:
        idx = torch.randint(len(X) - kernel_size, size=(N,))
        return slice_kernels(X, idx, kernel_size)

    center = int(X.shape[-1] // 2)

    if max_center_offset is None:

        # sample uniformly from all of X's time dimension
        min_val, max_val = 0, X.shape[-1] - kernel_size
    elif max_center_offset >= 0:
        # a positive max_center_offset means we're allowed
        # to put some space between the center of the timeseries
        # and the right edge of the kernel, but not the center
        # and the left edge of the kernel
        min_val = center - max_center_offset - kernel_size
        max_val = center
    else:
        # a negative max_center_offset means that we want
        # to enforce that the center of the timeseries is
        # some distance away from the edge of the kernel
        min_val = center - max_center_offset - kernel_size
        max_val = center + max_center_offset

        if max_val <= min_val:
            # if our required offset value is more than half
            # the kernel length, we won't be able to sample
            # any kernels at all
            raise ValueError(
                "Negative center offset value {} is too large "
                "for requested kernel size {}".format(
                    max_center_offset, kernel_size
                )
            )

    if min_val < 0:
        # if kernel_size > center - max_center_offset,
        # we may end up with negative indices
        raise ValueError(
            "Kernel size {} is too large for requested center "
            "offset value {}".format(kernel_size, max_center_offset)
        )

    if X.ndim == 3 or coincident:
        # sampling coincidentally, so just need a single
        # index for each element in the output batch
        N = N or len(X)
        shape = (N,)
    else:
        # otherwise, each channel in each batch sample
        # will require its own sampling index
        shape = (N, len(X))

    idx = torch.randint(min_val, max_val, size=shape).to(X.device)
    return slice_kernels(X, idx, kernel_size)
