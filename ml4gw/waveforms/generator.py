from typing import Callable, Dict, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

from ml4gw.constants import MSUN
from ml4gw.waveforms.cbc import utils

EXTRA_TIME_FRACTION = (
    0.1  # fraction of waveform duration to add as extra time for tapering
)
EXTRA_CYCLES = 3.0


class ParameterSampler(torch.nn.Module):
    def __init__(self, **parameters: Callable) -> None:
        super().__init__()
        self.parameters = parameters

    def forward(
        self,
        N: int,
    ) -> Dict[str, Float[Tensor, " {N}"]]:
        return {k: v.sample((N,)) for k, v in self.parameters.items()}


class TimeDomainCBCWaveformGenerator(torch.nn.Module):
    """
    Waveform generator that generates time-domain waveforms.

    All relevant data conditioning for injection into real data will be applied
    """

    def __init__(
        self,
        approximant: Callable,
        sample_rate: float,
        duration: float,
        f_min: float,
        f_ref: float,
        right_pad: float,
        parameter_sampler: ParameterSampler,
    ) -> None:
        """
        A torch module that generates waveforms from a given waveform function
        and a parameter sampler.

        Args:
            waveform:
                A callable that returns hplus and hcross polarizations
                given a set of parameters.
            sample_rate:
                Rate at which waveform will be sampled in Hz.
                This also determines `f_max` for the waveforms
            f_min:
                Lower frequency bound for waveforms
            duration:
                Length of waveform in seconds
            right_pad:
                How far from the right edge of the window
                the waveforms coalescence will be placed in seconds
            f_min:
            parameter_sampler:
                A ParameterSampler object
        """
        super().__init__()
        self.approximant = approximant
        self.f_min = f_min
        self.sample_rate = sample_rate
        self.duration = duration
        self.right_pad = right_pad
        self.f_ref = f_ref
        self.parameter_sampler = parameter_sampler

    @property
    def nyquist(self):
        return int(self.sample_rate / 2)

    @property
    def num_freqs(self):
        return int(self.nyquist / self.delta_f)

    @property
    def frequencies(self):
        return torch.linspace(0, self.nyquist, self.num_freqs)

    @property
    def size(self):
        return int(self.duration * self.sample_rate)

    @property
    def delta_f(self):
        return 1 / self.duration

    def forward(
        self, N: int
    ) -> Tuple[Float[Tensor, "{N} samples"], Dict[str, Float[Tensor, " {N}"]]]:

        parameters = self.parameter_sampler(N)
        mass_1, mass_2 = (
            parameters["mass_1"].double() * MSUN,
            parameters["mass_2"].double() * MSUN,
        )

        s1z, s2z = parameters["s1z"], parameters["s2z"]
        total_mass = mass_1 + mass_2

        new_fmin = (
            self.f_min
        )  # torch.max(utils.frequency_isco(mass_1, mass_2)).max(self.f_min)

        s = utils.final_black_hole_spin_bound(s1z, s2z)
        tchirp = utils.chirp_time_bound(new_fmin, mass_1, mass_2, s1z, s2z)
        tmerge = utils.merge_time_bound(
            mass_1, mass_2
        ) + utils.ringdown_time_bound(total_mass, s)

        # textra = EXTRA_CYCLES / new_fmin
        fstart = utils.chirp_start_frequency_bound(
            (1.0 + EXTRA_TIME_FRACTION) * tchirp, mass_1, mass_2
        ).min()

        # chirplen = round((tchirp + tmerge + 2.0 * textra) * self.sample_rate)

        # TODO: validate duration and chirplen somehow

        mask = self.frequencies >= fstart
        frequencies = self.frequencies[mask]

        cross, plus = self.approximant(
            frequencies, **parameters, f_ref=self.f_ref
        )

        cross_old, plus_old = cross.copy(), plus.copy()
        taper_mask = frequencies < torch.min(fstart)
        taper_length = taper_mask.sum()
        taper = 0.5 - 0.5 * torch.cos(
            torch.pi * torch.arange(taper_length) / taper_length
        )
        cross[..., taper_mask] *= taper
        plus[..., taper_mask] *= taper

        tshift = (
            torch.round(torch.max(tmerge) * self.sample_rate)
            * self.sample_rate
        )
        phase = torch.exp(2.0 * torch.pi * 1j * self.delta_f * tshift)
        cross, plus = cross * phase, plus * phase

        return cross, plus, cross_old, plus_old
