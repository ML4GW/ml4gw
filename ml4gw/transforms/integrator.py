from typing import Literal

import torch


class TophatIntegrator(torch.nn.Module):
    r"""
    Applies a causal boxcar (moving-average) filter along the
    last dimension of the input tensor. Each output sample
    represents the average of the previous `integration_length`
    seconds of data. Zero-padding is applied on the left so that
    the output has the same length as the input. As a result, the
    first few samples are computed from partial windows and
    therefore have smaller magnitude.

    Args:
        sample_rate (int):
            Sampling rate (Hz) of the input timeseries.
        integration_length (int):
            Integration window length in seconds.

    Shape:
        - Input: `(..., T)`
        - Output: `(..., T)`

    Returns:
        torch.Tensor:
            Integrated timeseries with the same length as the input.
    """

    def __init__(self, sample_rate: int, integration_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.integration_length = integration_length
        self.window_size = int(self.sample_rate * self.integration_length) + 1
        self.register_buffer(
            "window", torch.ones((1, 1, self.window_size)) / self.window_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        L = shape[-1]

        x = x.reshape(-1, 1, L)
        x = torch.nn.functional.pad(x, (self.window_size - 1, 0))
        x = torch.nn.functional.conv1d(x, self.window)

        return x.reshape(shape)


class LeakyIntegrator(torch.nn.Module):
    r"""
    This integrator accumulates evidence when the input exceeds
    a threshold and decays linearly. The accumulator can either
    increment by a constant value (event counting) or by the
    input score itself.

    Args:
        threshold (float):
            Minimum value required to contribute to the accumulator.
        decay (float):
            Amount subtracted per timestep when the threshold
            condition is not met.
        lower_bound (float):
            Lowest allowed value of the cumulative accumulator. The
            output is clipped so it never falls below this value.
        integrate_value (Literal["count", "score"]):
            Integration mode. Must be one of:
                - ``"count"``: increment by 1 per threshold crossing
                - ``"score"``: increment by the input value per
                               threshold crossing

    Shape:
        - Input: `(..., T)`
        - Output: `(..., T)`

    Returns:
        torch.Tensor:
            Cumulative leaky integral of the input sequence.
    """

    def __init__(
        self,
        threshold: float,
        decay: float,
        lower_bound: float,
        integrate_value: Literal["count", "score"],
    ):
        super().__init__()

        if integrate_value not in ["count", "score"]:
            raise ValueError(
                "Invalid integrate_value: "
                f"{integrate_value}. Must be 'count' or 'score'."
            )

        self.integrate_value = integrate_value
        self.threshold = threshold
        self.decay = decay
        self.lower_bound = lower_bound

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine the increment
        increment = (
            torch.ones_like(x) if self.integrate_value == "count" else x
        )
        # Compute how much the output changes at each step
        score_delta = torch.where(x >= self.threshold, increment, -self.decay)
        # Calculate the raw sum, not accounting for the lower bound
        raw_sum = torch.cumsum(score_delta, dim=-1)
        # Figure out the furthest we've gone below the lower bound
        offset = torch.cummin(raw_sum - self.lower_bound, dim=-1)[0]
        # Account for the case of the raw sum starting above the lower bound
        offset = torch.minimum(
            offset, torch.full_like(offset, self.lower_bound)
        )

        return raw_sum - offset
