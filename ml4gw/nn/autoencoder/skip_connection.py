import torch

from ml4gw.nn.autoencoder.utils import match_size


class SkipConnection(torch.nn.Module):
    def forward(self, X: torch.Tensor, state: torch.Tensor):
        return match_size(X, state.size(-1))

    def get_out_channels(self, in_channels):
        return in_channels


class AddSkipConnect(SkipConnection):
    def forward(self, X, state):
        X = super().forward(X, state)
        return X + state


class ConcatSkipConnect(SkipConnection):
    def __init__(self, groups: int = 1):
        super().__init__()
        self.groups = groups

    def get_out_channels(self, in_channels):
        return 2 * in_channels

    def forward(self, X, state):
        X = super().forward(X, state)
        if self.groups == 1:
            return torch.cat([X, state], dim=1)

        num_channels = X.size(1)
        rem = num_channels % self.groups
        if rem:
            raise ValueError(
                "Number of channels in input tensor {} cannot "
                "be divided evenly into {} groups".format(
                    num_channels, self.groups
                )
            )

        X = torch.split(X, self.groups, dim=1)
        state = torch.split(state, self.groups, dim=1)
        frags = [i for j in zip(X, state) for i in j]
        return torch.cat(frags, dim=1)
