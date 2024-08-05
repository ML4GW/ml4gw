from typing import Callable, Dict, Tuple

import torch
from jaxtyping import Float
from torch import Tensor


class ParameterSampler(torch.nn.Module):
    def __init__(self, **parameters: Callable) -> None:
        super().__init__()
        self.parameters = parameters

    def forward(
        self,
        N: int,
    ) -> Dict[str, Float[Tensor, " {N}"]]:
        return {k: v.sample((N,)) for k, v in self.parameters.items()}


class WaveformGenerator(torch.nn.Module):
    def __init__(
        self, waveform: Callable, parameter_sampler: ParameterSampler
    ) -> None:
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

    def forward(
        self, N: int
    ) -> Tuple[Float[Tensor, "{N} samples"], Dict[str, Float[Tensor, " {N}"]]]:
        parameters = self.parameter_sampler(N)
        return self.waveform(**parameters), parameters
