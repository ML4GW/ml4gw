from typing import List

import torch
from torch import Tensor
from torchaudio.transforms import Resample


class MultiRateResample(torch.nn.Module):
    """
    Resample a time series to multiple different sample rates

    Args:
        original_sample_rate:
            The sample rate of the original time series in Hz
        duration:
            The duration of the original time series in seconds
        new_sample_rates:
            A list of new sample rates that different portions
            of the time series will be resampled to
        breakpoints:
            The time at which there is a transition from one
            sample rate to another

    Returns:
        A time series Tensor with each of the resampled segments
        concatenated together
    """

    def __init__(
        self,
        original_sample_rate: int,
        duration: float,
        new_sample_rates: List[int],
        breakpoints: List[float],
    ):
        super().__init__()
        self.original_sample_rate = original_sample_rate
        self.duration = duration
        self.new_sample_rates = new_sample_rates
        self.breakpoints = breakpoints
        self._validate_inputs()

        # Add endpoints to breakpoint list
        self.breakpoints.append(duration)
        self.breakpoints.insert(0, 0)

        self.resamplers = torch.nn.ModuleList(
            [Resample(original_sample_rate, new) for new in new_sample_rates]
        )
        idxs = [
            [int(breakpoints[i] * new), int(breakpoints[i + 1] * new)]
            for i, new in enumerate(self.new_sample_rates)
        ]
        self.register_buffer("idxs", torch.Tensor(idxs).int())

    def _validate_inputs(self):
        if len(self.new_sample_rates) != len(self.breakpoints) + 1:
            raise ValueError(
                "There are too many/few breakpoints given "
                "for the number of frequencies"
            )
        if max(self.breakpoints) >= self.duration:
            raise ValueError(
                "At least one breakpoint was greater than the given duration"
            )
        if not self.breakpoints[1:] > self.breakpoints[:-1]:
            raise ValueError("Breakpoints must be sorted in ascending order")

    def forward(self, X: Tensor):
        return torch.cat(
            [
                resample(X)[..., idx[0] : idx[1]]
                for resample, idx in zip(self.resamplers, self.idxs)
            ],
            dim=-1,
        )
