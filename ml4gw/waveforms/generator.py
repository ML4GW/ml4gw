from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from ml4gw.distributions import ParameterSampler


class WaveformGenerator(torch.nn.Module):
    def __init__(
        self, waveform: Callable, parameter_sampler: "ParameterSampler"
    ):
        """
        A torch module that generates waveforms from a given waveform function
        and a parameter sampler.

        Args:
            waveform:
                A callable that returns hplus and hcross polarizations
                given a set of parameters.
            parameter_sampler:
                A ParameterSampler object
        """

        super().__init__()
        self.waveform = waveform
        self.parameter_sampler = parameter_sampler

    def forward(self, N: int):
        parameters = self.parameter_sampler(N)
        return self.waveform(**parameters), parameters
