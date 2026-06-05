# Adapted with modifications from https://github.com/state-spaces/s4
# (src/models/nn/dropout.py), licensed under Apache-2.0.

import torch
import torch.nn as nn

class DropoutNd(nn.Module):
    """N-dimensional dropout that ties the mask across sequence positions."""

    def __init__(
        self, p: float = 0.5, tie: bool = True, transposed: bool = True
    ):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")

        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (batch, dim, lengths...) if transposed
        else (batch, lengths..., dim).
        """
        if self.training:
            if not self.transposed:
                X = X.movedim(-1, 1)
            mask_shape = (
                X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            )
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = X.movedim(1, -1)
        return X