from typing import Optional

import numpy as np
import torch


class ChannelWiseScaler(torch.nn.Module):
    def __init__(self, num_channels: Optional[int] = None) -> None:
        super().__init__()

        shape = (num_channels or 1,)
        if num_channels is not None:
            shape += (1,)

        mean = torch.zeros(shape)
        std = torch.zeros(shape)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def fit(self, X: np.ndarray) -> None:
        if X.ndim == 1:
            assert self.mean.ndim == self.std.ndim == 1
            mean = X.mean(keepdims=True)
            std = X.std(keepdims=True)
        elif X.ndim == 2:
            assert self.mean.ndim == self.std.ndim == 2
            assert len(X) == self.mean.size(0) == self.std.size(0)
            mean = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True)
        else:
            raise ValueError(
                "Can't fit channel wise mean and standard deviation "
                "from tensor of shape {}".format(X.shape)
            )

        with torch.no_grad():
            self.mean.copy_(torch.Tensor(mean))
            self.std.copy_(torch.Tensor(std))

    def forward(self, X: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if not reverse:
            return (X - self.mean) / self.std
        else:
            return self.std * X + self.mean
