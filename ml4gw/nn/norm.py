from typing import Callable, Optional

import torch

NormLayer = Callable[[int], torch.nn.Module]


class GroupNorm1D(torch.nn.Module):
    """
    Custom implementation of GroupNorm which is faster than the
    out-of-the-box PyTorch version at inference time.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: Optional[int] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        num_groups = num_groups or num_channels
        if num_channels % num_groups:
            raise ValueError("num_groups must be a factor of num_channels")

        self.num_channels = num_channels
        self.num_groups = num_groups
        self.channels_per_group = self.num_channels // self.num_groups
        self.eps = eps

        shape = (self.num_channels, 1)
        self.weight = torch.nn.Parameter(torch.ones(shape))
        self.bias = torch.nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        keepdims = self.num_groups == self.num_channels

        # compute group variance via the E[x**2] - E**2[x] trick
        mean = x.mean(-1, keepdims=keepdims)
        sq_mean = (x**2).mean(-1, keepdims=keepdims)

        # if we have groups, do some reshape magic
        # to calculate group level stats then
        # reshape back to full channel dimension
        if self.num_groups != self.num_channels:
            mean = torch.stack([mean, sq_mean], dim=1)
            mean = mean.reshape(
                -1, 2, self.num_groups, self.channels_per_group
            )
            mean = mean.mean(-1, keepdims=True)
            mean = mean.expand(-1, -1, -1, self.channels_per_group)
            mean = mean.reshape(-1, 2, self.num_channels, 1)
            mean, sq_mean = mean[:, 0], mean[:, 1]

        # roll the mean and variance into the
        # weight and bias so that we have to do
        # fewer computations along the full time axis
        std = (sq_mean - mean**2 + self.eps) ** 0.5
        scale = self.weight / std
        shift = self.bias - scale * mean
        return shift + x * scale


class GroupNorm1DGetter:
    """
    Utility for making a NormLayer Callable that maps from
    an integer number of channels to a torch Module. Useful
    for command-line parameterization with jsonargparse.
    """

    def __init__(self, groups: Optional[int] = None) -> None:
        self.groups = groups

    def __call__(self, num_channels: int) -> torch.nn.Module:
        if self.groups is None:
            num_groups = None
        else:
            num_groups = min(num_channels, self.groups)
        return GroupNorm1D(num_channels, num_groups)


# TODO generalize faster 1dDGroupNorm to 2D
class GroupNorm2DGetter:
    """
    Utility for making a NormLayer Callable that maps from
    an integer number of channels to a torch Module. Useful
    for command-line parameterization with jsonargparse.
    """

    def __init__(self, groups: Optional[int] = None) -> None:
        self.groups = groups

    def __call__(self, num_channels: int) -> torch.nn.Module:
        if self.groups is None:
            num_groups = num_channels
        else:
            num_groups = min(num_channels, self.groups)
        return torch.nn.GroupNorm(num_groups, num_channels)
