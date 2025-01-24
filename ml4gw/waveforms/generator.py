import math
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


class TimeDomainCBCWaveformGenerator(torch.nn.Module):
    """
    Waveform generator that generates time-domain waveforms. Currently,
    only conversion from frequency domain approximants is implemented.

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
    ) -> None:
        """
        A torch module that generates waveforms from a given waveform function
        and a parameter sampler.

        Args:
            approximant:
                A callable that returns hplus and hcross polarizations
                given requested frequencies and relevant set of parameters.
            sample_rate:
                Rate at which returned time domain waveform will be
                sampled in Hz.This also determines `f_max` for the waveforms.
            f_min:
                Lower frequency bound for waveforms
            duration:
                Length of waveform in seconds
            right_pad:
                How far from the right edge of the window
                the returned waveform coalescence
                will be placed in seconds
            f_ref:
                Reference frequency for the waveform
        """
        super().__init__()
        self.approximant = approximant
        self.f_min = f_min
        self.sample_rate = sample_rate
        self.duration = duration
        self.right_pad = right_pad
        self.f_ref = f_ref

    def get_frequencies(self, df: float):
        """Get the frequencies from 0 to nyquist for corresponding df"""
        num_freqs = int(self.nyquist / df) + 1
        return torch.linspace(0, self.nyquist, num_freqs)

    @property
    def nyquist(self):
        return int(self.sample_rate / 2)

    @property
    def size(self):
        """Number of samples in the waveform"""
        return int(self.duration * self.sample_rate)

    @property
    def delta_f(self):
        return 1 / self.duration

    def forward(
        self,
        **parameters,
    ) -> Tuple[Float[Tensor, "{N} samples"], Dict[str, Float[Tensor, " {N}"]]]:
        """
        Heavily based on https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/python/lalsimulation/gwsignal/core/waveform_conditioning.py?ref_type=heads#L248 # noqa
        """
        # convert masses to kg, make sure
        # they are doubles so there is no
        # overflow in the calculations
        mass_1, mass_2 = (
            parameters["mass_1"].double() * MSUN,
            parameters["mass_2"].double() * MSUN,
        )

        device = mass_1.device

        s1z, s2z = parameters["s1z"], parameters["s2z"]
        total_mass = mass_1 + mass_2

        f_min = torch.minimum(
            utils.frequency_isco(mass_1, mass_2),
            torch.tensor(self.f_min, device=device),
        )
        s = utils.final_black_hole_spin_bound(s1z, s2z)
        tmerge = utils.merge_time_bound(
            mass_1, mass_2
        ) + utils.ringdown_time_bound(total_mass, s)

        tchirp = utils.chirp_time_bound(f_min, mass_1, mass_2, s1z, s2z)
        textra = EXTRA_CYCLES / f_min
        fstart = utils.chirp_start_frequency_bound(
            (1.0 + EXTRA_TIME_FRACTION) * tchirp, mass_1, mass_2
        )

        tchirp = utils.chirp_time_bound(fstart, mass_1, mass_2, s1z, s2z)
        chirplen = torch.round(
            (tchirp + tmerge + 2.0 * textra) * self.sample_rate
        )

        # pad to next power of 2
        chirplen = 2 ** torch.ceil(torch.log(chirplen) / math.log(2))

        # get smallest df corresponding to longest chirp length,
        # which will make sure there is no wrap around effects.
        df = min(1.0 / (chirplen.max() / self.sample_rate), self.delta_f)
        frequencies = self.get_frequencies(df).to(mass_1.device)

        # downselect to frequencies above fstart,
        # and generate the waveform at the specified frequencies
        freq_mask = frequencies >= fstart.min()
        frequencies = frequencies[freq_mask]

        cross, plus = self.approximant(
            frequencies, **parameters, f_ref=self.f_ref
        )
        batch_size = cross.size(0)

        # build a taper that is dependent on each
        # individual waveforms fstart;
        # since this means that the taper sizes
        # will be different for each waveform,
        # construct the tapers based on the maximum size
        # and then set the values outside of the individual
        # waveform taper regions to 1.0
        taper_mask = frequencies <= f_min[:, None]
        taper_size = taper_mask.sum(axis=1)
        max_taper_size = taper_size.max()
        taper = torch.nan_to_num(
            torch.arange(max_taper_size, device=device) / taper_size[:, None],
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        taper = 0.5 - 0.5 * torch.cos(torch.pi * taper)

        mask = torch.arange(max_taper_size, device=device).expand(
            batch_size, -1
        )
        mask = mask <= taper_size.unsqueeze(1)
        taper[~mask] = 1.0

        # finally, apply the taper to the waveforms
        cross[..., :max_taper_size] *= taper
        plus[..., :max_taper_size] *= taper

        # now, construct full spectrum of frequencies
        # from 0 to nyquist, and then fill in the
        # requested frequencies with the waveform values

        shape = (batch_size, int(self.nyquist / df) + 1)
        hc_spectrum = torch.zeros(shape, dtype=cross.dtype, device=device)
        hp_spectrum = torch.zeros(shape, dtype=plus.dtype, device=device)

        hc_spectrum[:, freq_mask] = cross
        hp_spectrum[:, freq_mask] = plus

        # set nyquist frequency to zero
        hc_spectrum[..., -1], hp_spectrum[..., -1] = 0.0, 0.0

        # perfom inverse fft to get time domain waveforms
        cross, plus = torch.fft.irfft(hc_spectrum), torch.fft.irfft(
            hp_spectrum
        )
        cross, plus = cross * self.sample_rate, plus * self.sample_rate

        # roll the waveforms so that the coalescence
        # is `right_pad` seconds from the end
        cross = torch.roll(
            cross, -int(self.right_pad * self.sample_rate), dims=-1
        )
        plus = torch.roll(
            plus, -int(self.right_pad * self.sample_rate), dims=-1
        )

        # lastly, slice the waveforms based on the requested duration
        cross = cross[..., -self.size :]
        plus = plus[..., -self.size :]

        # TODO: implement and apply butterworth
        # highpass filter in torch

        return cross, plus
