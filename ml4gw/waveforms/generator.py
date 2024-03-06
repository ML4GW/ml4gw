from typing import Callable

import torch


class ParameterSampler(torch.nn.Module):
    def __init__(self, **parameters: Callable):
        super().__init__()
        self.parameters = parameters

    def forward(
        self,
        N: int,
    ):
        return {k: v.sample((N,)) for k, v in self.parameters.items()}

    def to_bilby_prior_dict(self):
        """
        Convert distributions supplied to parameter sampler to their bilby
        prior equivalents and return a PriorDict object. Raises an error
        if supplied distribution does not implement method
        ``.bilby_prior_equivalent``.
        """
        from ml4gw import distributions

        if not distributions._BILBY_INSTALLED:
            raise RuntimeError("Bilby should be installed to use this method")
        assert all(
            [
                hasattr(dist, "bilby_prior_equivalent")
                for dist in self.parameters.values()
            ]
        ), "Not all distributions have a bilby_prior_equivalent."
        from bilby import prior

        return prior.PriorDict(
            {k: v.bilby_prior_equivalent() for k, v in self.parameters.items()}
        )


class WaveformGenerator(torch.nn.Module):
    def __init__(
        self, waveform: Callable, parameter_sampler: ParameterSampler
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
