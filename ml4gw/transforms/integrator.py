import numpy as np


class TophatIntegrator:
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
        np.ndarray:
            Integrated timeseries with the same length as the input.
    """

    def __init__(
        self,
        inference_sample_rate: int,
        integration_length: int,
    ):
        self.inference_sample_rate = inference_sample_rate
        self.integration_length = integration_length
        self.window_size = int(
            self.inference_sample_rate * self.integration_length
        )
        self.window = np.ones((self.window_size,)) / self.window_size

    def __call__(self, y: np.ndarray) -> np.ndarray:
        integrated = np.convolve(y, self.window, mode="full")
        return integrated[: -self.window_size + 1]


class LeakyIntegrator:
    r"""
    This integrator accumulates evidence when the input exceeds
    a threshold and decays otherwise. The accumulator can either
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
        np.ndarray:
            Leaky integrated timeseries.
    """

    def __init__(
        self,
        threshold: float,
        decay: float,
        integrate_value: str,
        lower_bound: float,
        detection_threshold: float | None = None,
    ):
        integrate_value = integrate_value.lower()
        self.integrate_value = integrate_value

        self.threshold = threshold
        self.decay = decay
        self.lower_bound = lower_bound
        self.detection_threshold = detection_threshold

    def __call__(self, y: np.ndarray) -> np.ndarray:
        output = []
        score = self.lower_bound
        for data in y:
            data = float(data)
            if data >= self.threshold:
                if self.integrate_value == "count":
                    score += 1.0
                elif self.integrate_value == "score":
                    score += data
                else:
                    raise ValueError(
                        "Invalid integrate_value: "
                        f"{self.integrate_value}. Must be 'count' or 'score'."
                    )
            else:
                score = max(self.lower_bound, score - self.decay)

            if (
                self.detection_threshold is not None
                and score >= self.detection_threshold
            ):
                output.append(score)
                score = self.lower_bound
            else:
                output.append(score)
        return np.asarray(output)
