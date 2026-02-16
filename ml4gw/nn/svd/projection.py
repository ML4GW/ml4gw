"""
Frequency-domain SVD projection layer, adapted from DINGO's
LinearProjectionRB.

Projects multi-channel time-domain input onto a reduced SVD
basis in the frequency domain. Useful as a learned first layer
that filters out noise components orthogonal to the signal
manifold.

Reference:
    Dax et al., "Real-Time Gravitational Wave Science with Neural
    Posterior Estimation" (DINGO), https://github.com/dingo-gw/dingo
"""

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class FreqDomainSVDProjection(nn.Module):
    """Frequency-domain SVD projection for gravitational wave data.

    For each input channel (interferometer):

    1. FFT the time-domain input to get a complex frequency series.
    2. Stack real and imaginary parts into a real-valued vector.
    3. Project onto a reduced basis via a linear layer (no bias),
       optionally initialized with the right singular vectors V
       from a precomputed SVD.

    The output is flattened across channels:
    ``(batch, num_channels * n_svd)``.

    Args:
        num_channels:
            Number of input channels (e.g. interferometers).
        n_freq:
            Number of positive frequency bins in the FFT output
            (``n_samples // 2 + 1``).
        n_svd:
            Number of SVD basis components to project onto.
        V:
            Optional initial projection weights of shape
            ``(2 * n_freq, n_svd)`` — the right singular vectors
            from an SVD of stacked ``[real, imag]`` frequency data.
            If ``None``, the projection is randomly initialized.
        per_channel:
            If ``True``, use separate projection weights per channel
            (all initialized from the same V, but free to diverge
            during training). If ``False``, share one projection
            across all channels.
    """

    def __init__(
        self,
        num_channels: int,
        n_freq: int,
        n_svd: int,
        V: np.ndarray | Tensor | None = None,
        per_channel: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.n_freq = n_freq
        self.n_svd = n_svd
        self.per_channel = per_channel

        if V is not None:
            V_tensor = self._to_tensor(V)
            if V_tensor.shape != (2 * n_freq, n_svd):
                raise ValueError(
                    f"V must have shape ({2 * n_freq}, {n_svd}), "
                    f"got {tuple(V_tensor.shape)}"
                )
        else:
            V_tensor = None

        if per_channel:
            self.projections = nn.ModuleList()
            for _ in range(num_channels):
                proj = nn.Linear(2 * n_freq, n_svd, bias=False)
                if V_tensor is not None:
                    proj.weight.data = V_tensor.T.contiguous()
                self.projections.append(proj)
        else:
            self.projection = nn.Linear(2 * n_freq, n_svd, bias=False)
            if V_tensor is not None:
                self.projection.weight.data = V_tensor.T.contiguous()

    @staticmethod
    def _to_tensor(V: np.ndarray | Tensor) -> Tensor:
        if isinstance(V, np.ndarray):
            return torch.from_numpy(V).float()
        return V.float()

    @property
    def output_dim(self) -> int:
        """Total output dimension: ``num_channels * n_svd``."""
        return self.n_svd * self.num_channels

    def forward(
        self, x: Float[Tensor, "batch channels time"]
    ) -> Float[Tensor, "batch features"]:
        """Project multi-channel time-domain input onto SVD basis.

        Args:
            x:
                Time-domain input of shape
                ``(batch, num_channels, n_samples)``.

        Returns:
            Projected features of shape
            ``(batch, num_channels * n_svd)``.
        """
        batch_size = x.shape[0]

        # FFT per channel
        x_freq = torch.fft.rfft(x, dim=-1)

        # Stack real and imaginary: (batch, channels, 2 * n_freq)
        x_ri = torch.cat([x_freq.real, x_freq.imag], dim=-1)

        if self.per_channel:
            proj_list = []
            for ch in range(self.num_channels):
                proj_list.append(
                    self.projections[ch](x_ri[:, ch, :])
                )
            x_proj = torch.stack(proj_list, dim=1)
        else:
            x_proj = self.projection(x_ri)

        return x_proj.reshape(batch_size, -1)

    def freeze(self):
        """Freeze projection weights (Phase 1 training)."""
        params = (
            self.projections.parameters()
            if self.per_channel
            else self.projection.parameters()
        )
        for p in params:
            p.requires_grad = False

    def unfreeze(self):
        """Unfreeze projection weights (Phase 2 fine-tuning)."""
        params = (
            self.projections.parameters()
            if self.per_channel
            else self.projection.parameters()
        )
        for p in params:
            p.requires_grad = True
