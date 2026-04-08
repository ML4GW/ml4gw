"""
Dense residual block for post-SVD processing.

Uses LayerNorm rather than BatchNorm to avoid train/eval
discrepancies in gravitational wave applications where
batch statistics during training (mixed signal/noise batches)
don't match evaluation conditions.
"""

import torch
import torch.nn as nn


class DenseResidualBlock(nn.Module):
    """Fully-connected residual block with layer normalization.

    Computes ``LayerNorm(x + MLP(x))`` where the MLP is
    ``Linear → GELU → Dropout → Linear``.

    This block is designed for processing SVD coefficients or
    other 1D feature vectors in gravitational wave detection
    networks.

    .. note::
        This is intentionally separate from
        :class:`~ml4gw.nn.resnet.resnet_1d.BasicBlock`, which
        uses 1D convolutions for time-series data. Key differences:

        - **Layers**: ``nn.Linear`` (dense) vs ``nn.Conv1d``
        - **Normalization**: ``LayerNorm`` (post-residual) vs
          ``GroupNorm``/``BatchNorm`` (pre-residual per conv)
        - **Activation**: ``GELU`` vs ``ReLU``
        - **Dimensionality**: Fixed ``(batch, dim)`` vs variable
          channel/spatial dims with optional downsampling

        Unifying these would require a complex abstraction with
        little practical benefit, since the two blocks target
        fundamentally different data representations (1D feature
        vectors vs multi-channel time series).

    .. note::
        BatchNorm is intentionally not supported. In GW detection
        training, batches contain a mix of signal and noise samples
        at varying ratios. BatchNorm learns running statistics from
        this training distribution, but at evaluation time the
        input distribution differs, causing output collapse
        (near-constant predictions). LayerNorm normalizes
        per-sample, avoiding this failure mode.

    Args:
        dim:
            Feature dimension (input and output size).
        dropout:
            Dropout probability applied after the activation.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection with normalization.

        Args:
            x: Input tensor of shape ``(batch, dim)``.

        Returns:
            Output tensor of shape ``(batch, dim)``.
        """
        return self.norm(x + self.net(x))
