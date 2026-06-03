import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..types import TimeSeries1to3d


class ShiftedPearsonCorrelation(torch.nn.Module):
    """
    Compute the `Pearson correlation <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
    for two equal-length timeseries over a pre-defined number of time
    shifts in each direction. Useful for when you want a
    correlation, but not over every possible shift (i.e.
    a convolution).

    The number of dimensions of the second timeseries ``y``
    passed at call time should always be less than or equal
    to the number of dimensions of the first timeseries ``x``,
    and each dimension should match the corresponding one of
    ``x`` in  reverse order (i.e. if ``x`` has shape ``(B, C, T)``
    then ``y`` should either have shape ``(T,)``, ``(C, T)``, or
    ``(B, C, T)``).

    Note that no windowing to either timeseries is applied
    at call time. Users should do any requisite windowing
    beforehand.

    TODOs:
    - Should we perform windowing?
    - Should we support stride > 1?

    Args:
        max_shift:
            The maximum number of 1-step time shifts in
            each direction over which to compute the
            Pearson coefficient. Output shape will then
            be ``(2 * max_shifts + 1, B, C)``.
    """

    def __init__(self, max_shift: int) -> None:
        super().__init__()
        self.max_shift = max_shift

    def _shape_checks(self, x: TimeSeries1to3d, y: TimeSeries1to3d):
        if x.ndim > 3:
            raise ValueError(
                "Tensor x can only have up to 3 dimensions "
                f"to compute ShiftedPearsonCorrelation. Found {x.ndim}."
            )
        elif y.ndim > x.ndim:
            raise ValueError(
                "y may not have more dimensions that x for "
                "ShiftedPearsonCorrelation, but found shapes "
                f"{y.shape} and {x.shape}"
            )
        for dim in range(y.ndim):
            if y.size(-dim - 1) != x.size(-dim - 1):
                raise ValueError(
                    "x and y expected to have same size along "
                    f"last dimensions, but found shapes {x.shape} and "
                    f"{y.shape}"
                )

    def forward(
        self, x: TimeSeries1to3d, y: TimeSeries1to3d
    ) -> Float[Tensor, "windows ..."]:
        self._shape_checks(x, y)
        dim = x.size(-1)

        # pad x along time dimension so that it has shape
        # batch x channels x (time + 2 * max_shift)
        pad = (self.max_shift, self.max_shift)
        x = F.pad(x, pad)
        y = y - y.mean(-1, keepdims=True)

        num_shifts = 2 * self.max_shift + 1
        n_fft = 2 * (dim + self.max_shift)

        # Compute correlation via FFT
        x_fft = torch.fft.rfft(x, n=n_fft, dim=-1)
        y_fft = torch.fft.rfft(y, n=n_fft, dim=-1)

        corr = torch.fft.irfft(x_fft * y_fft.conj(), n=n_fft, dim=-1)
        corr = corr[..., :num_shifts]
        corr = corr.movedim(-1, 0)

        # Compute the variance of x at each shift via cumsum
        cumsum_x = F.pad(torch.cumsum(x, dim=-1), (1, 0))
        cumsum_x2 = F.pad(torch.cumsum(x**2, dim=-1), (1, 0))
        window_sum_x = (
            cumsum_x[..., dim : dim + num_shifts] - cumsum_x[..., :num_shifts]
        )
        window_sum_x2 = (
            cumsum_x2[..., dim : dim + num_shifts]
            - cumsum_x2[..., :num_shifts]
        )

        var_x = window_sum_x2 - (window_sum_x**2) / dim
        var_y = (y**2).sum(-1)

        norm = (var_x * var_y.unsqueeze(-1)) ** 0.5
        norm = norm.movedim(-1, 0)

        return corr / norm
