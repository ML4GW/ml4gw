from typing import Callable

import torch

from ml4gw.distributions import ParameterSampler


class WaveformGenerator(torch.nn.Module):
    def __init__(
        self, waveform: Callable, parameter_sampler: ParameterSampler
    ):
        """
        A torch module that generates waveforms from a given waveform function
        and a parameter sampler.
        """

        super().__init__()
        self.waveform = waveform
        self.parameter_sampler = parameter_sampler

    def forward(self, N: int):
        parameters = self.parameter_sampler(N)
        return self.waveform(**parameters)
