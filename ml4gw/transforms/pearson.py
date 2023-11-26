import torch

from ml4gw.utils.slicing import unfold_windows


class ShiftedPearsonCorrelation(torch.nn.Module):
    """
    Compute the [Pearson correlation]
    (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
    for two equal-length timeseries over a pre-defined number of time
    shifts in each direction. Useful for when you want a
    correlation, but not over every possible shift (i.e.
    a convolution).

    The number of dimensions of the second timeseries `y`
    passed at call time should always be less than or equal
    to the number of dimensions of the first timeseries `x`,
    and each dimension should match the corresponding one of
    `x` in  reverse order (i.e. if `x` has shape `(B, C, T)`
    then `y` should either have shape `(T,)`, `(C, T)`, or
    `(B, C, T)`).

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
            be `(2 * max_shifts + 1, B, C)`.
    """

    def __init__(self, max_shift: int) -> None:
        super().__init__()
        self.max_shift = max_shift

    def _shape_checks(self, x: torch.Tensor, y: torch.Tensor):
        if x.ndim > 3:
            raise ValueError(
                "Tensor x can only have up to 3 dimensions "
                f"to compute ShiftedPearsonCorrelation. Found {x.ndim}."
            )
        elif y.ndim > x.ndim:
            raise ValueError(
                "y may not have more dimensions that x for "
                "ShiftedPearsonCorrelation, but found shapes "
                "{} and {}".format(y.shape, x.shape)
            )
        for dim in range(y.ndim):
            if y.size(-dim - 1) != x.size(-dim - 1):
                raise ValueError(
                    "x and y expected to have same size along "
                    "last dimensions, but found shapes {} and {}".format(
                        x.shape, y.shape
                    )
                )

    # TODO: torchtyping annotate
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._shape_checks(x, y)
        dim = x.size(-1)

        # pad x along time dimension so that it has shape
        # batch x channels x (time + 2 * max_shift)
        pad = (self.max_shift, self.max_shift)
        x = torch.nn.functional.pad(x, pad)

        # num_windows x batch x channels x time
        x = unfold_windows(x, dim, 1)

        # now compute the correlation between each window
        # of x and the single window of y. Start by de-meaning
        x = x - x.mean(-1, keepdims=True)
        y = y - y.mean(-1, keepdims=True)

        # apply formula and sum along time dimension to give final shape
        # num_windows x batch x channels
        corr = (x * y).sum(axis=-1)
        norm = (x**2).sum(-1) * (y**2).sum(-1)

        return corr / norm**0.5
