from typing import Literal

import torch


class TophatIntegrator(torch.nn.Module):
    r"""
    Convolve predictions with boxcar filter to get local
    integration, slicing off of the last values so that
    timeseries represents integration of _past_ data only.
    "Full" convolution means first few samples are integrated
    with 0s, so will have a lower magnitude than they
    technically should.

    Args:
        inference_sample_rate (int):
            Sampling rate (Hz) of the input timeseries.
        integration_length (int):
            Integration window length in seconds.

    Shape:
        - Input: `(T,)`
        - Output: `(T,)`

    Returns:
        torch.Tensor:
            Integrated timeseries with the same length as the input.
    """

    def __init__(
        self,
        inference_sample_rate: int,
        integration_length: int,
    ):
        super().__init__()
        self.inference_sample_rate = inference_sample_rate
        self.integration_length = integration_length
        self.window_size = (
            int(self.inference_sample_rate * self.integration_length) + 1
        )
        self.register_buffer(
            "window", torch.ones((1, 1, self.window_size)) / self.window_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 1:
            raise ValueError("TophatIntegrator expects input shape (T,)")

        x = x.view(1, 1, -1)
        x = torch.nn.functional.pad(x, (self.window_size - 1, 0))
        integrated = torch.nn.functional.conv1d(x, self.window)

        return integrated[0, 0, :]


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
            Amount subtracted from the accumulator when the
            threshold is not exceeded.
        integrate_value (str):
            Integration mode. Must be one of:
                - ``"count"``: increment by 1 per threshold crossing
                - ``"score"``: increment by the input value
        lower_bound (float):
            Minimum value of the accumulator.
        detection_threshold (float, optional):
            If provided, indicates a detection threshold for the
            accumulated statistic and resets the accumulator to the
            lower bound upon detection.

    Shape:
        - Input: `(T,)`
        - Output: `(T,)`

    Returns:
        torch.Tensor:
            Leaky integrated timeseries.
    """

    def __init__(
        self,
        threshold: float,
        decay: float,
        lower_bound: float,
        integrate_value: Literal["count", "score"],
    ):
        super().__init__()

        integrate_value = integrate_value.lower()
        if integrate_value not in ["count", "score"]:
            raise ValueError(
                "Invalid integrate_value: "
                f"{integrate_value}. Must be 'count' or 'score'."
            )

        self.integrate_value = integrate_value
        self.register_buffer("threshold", torch.tensor(threshold))
        self.register_buffer("decay", torch.tensor(decay))
        self.register_buffer("lower_bound", torch.tensor(lower_bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 1:
            raise ValueError("LeakyIntegrator expects input shape (T,)")

        threshold = self.threshold.to(x.dtype)
        decay = self.decay.to(x.dtype)
        lower_bound = self.lower_bound.to(x.dtype)

        output = torch.empty_like(x)
        score = lower_bound

        for i, data in enumerate(x):
            if data >= threshold:
                if self.integrate_value == "count":
                    score = score + 1.0
                else:
                    score = score + data
            else:
                score = torch.maximum(lower_bound, score - decay)

            output[i] = score

        return output
