import math
from typing import Callable, Tuple

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from ..constants import MSUN
from ..transforms import IIRFilter
from ..types import BatchTensor
from .cbc import utils

EXTRA_TIME_FRACTION = (
    0.1  # fraction of waveform duration to add as extra time for tapering
)
EXTRA_CYCLES = 3.0


class TimeDomainCBCWaveformGenerator(torch.nn.Module):
    """
    Waveform generator that generates time-domain waveforms from frequency-domain approximants.

    Frequency domain waveforms are conditioned as done by lalsimulation.
    Specifically, waveforms are generated with a starting frequency `fstart`
    slightly below the requested `f_min`, so that they can be tapered from
    `fstart` to `f_min` using a cosine window.

    Please see https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_inspiral__c.html#gac9f16dab2cbca5a431738ee7d2505969 # noqa
    for more information

    Args:
        approximant:
            A callable that returns hplus and hcross polarizations
            given requested frequencies and relevant set of parameters.
            See `ml4gw.waveforms.cbc` for implemented approximants.
        sample_rate:
            Rate at which returned time domain waveform will be
            sampled in Hz. This also specifies `f_max` for generating
            waveforms via the nyquist frequency: `f_max = sample_rate // 2`.
        f_min:
            Lower frequency bound for waveforms
        duration:
            Length of waveform in seconds.
            Waveforms will be left padded with zeros
            appropiately to fill the requested duration
        right_pad:
            How far from the right edge of the window
            in seconds the returned waveform coalescence
            will be placed.
        f_ref:
            Reference frequency for the waveform
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

        super().__init__()
        self.approximant = approximant
        self.f_min = f_min
        self.sample_rate = sample_rate
        self.duration = duration
        self.right_pad = right_pad
        self.f_ref = f_ref

        self.highpass = self.build_highpass_filter()

    @property
    def delta_t(self):
        return 1 / self.sample_rate

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

    def build_highpass_filter(self):
        """
        Builds highpass filter object.

        See https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/python/lalsimulation/gwsignal/core/conditioning_subroutines.py?ref_type=heads#L10 # noqa
        """
        order = 8.0
        w1 = np.tan(np.pi * (self.f_min) / self.sample_rate)
        attenuation = 0.99
        wc = w1 * (1.0 / attenuation**0.5 - 1) ** (1.0 / (2.0 * order))
        fc = self.sample_rate * np.arctan(wc) / np.pi

        return IIRFilter(
            order,
            fc / (self.sample_rate / 2),
            btype="highpass",
            ftype="butterworth",
        )

    def get_frequencies(self, df: float):
        """Get the frequencies from 0 to nyquist for corresponding df"""
        num_freqs = int(self.nyquist / df) + 1
        return torch.linspace(0, self.nyquist, num_freqs)

    def generate_conditioned_fd_waveform(
        self, **parameters: dict[str, BatchTensor]
    ) -> Tuple[Float[Tensor, "{N} samples"], Float[Tensor, "{N} samples"]]:
        """
        Generate a conditioned frequency domain waveform from a frequency domain approximant.

        Based on https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/python/lalsimulation/gwsignal/core/waveform_conditioning.py?ref_type=heads#L248 # noqa

        Args:
            **parameters:
                Dictionary of parameters for waveform generation
                where key is the parameter name and value is a tensor of parameters.
                It is required that `parameters` contains `mass_1`, `mass_2`, `s1z`, and `s2z`
                keys, which are used for determining parameters of data conditioning.
                If the specified approximant takes other parameters for waveform generation,
                like `chirp_mass` and `mass_ratio`, the utility functions in `ml4gw.waveforms.conversion`
                may be useful for populating the parameters dictionary with these additional parameters.
                Note that, if using an approximant from `ml4gw.waveforms.cbc`, any additional keys in `parameters`
                not ingested by the approximant will be ignored.
        """
        # convert masses to kg, make sure
        # they are doubles so there is no
        # overflow in the calculations
        mass_1, mass_2 = (
            parameters["mass_1"].double() * MSUN,
            parameters["mass_2"].double() * MSUN,
        )
        total_mass = mass_1 + mass_2
        s1z, s2z = parameters["s1z"], parameters["s2z"]
        device = mass_1.device

        f_isco = utils.frequency_isco(mass_1, mass_2)
        f_min = torch.minimum(
            f_isco,
            torch.tensor(self.f_min, device=device),
        )

        # upper bound on chirp time
        tchirp = utils.chirp_time_bound(f_min, mass_1, mass_2, s1z, s2z)

        # upper bound on final black hole spin
        s = utils.final_black_hole_spin_bound(s1z, s2z)

        # upper bound on the final plunge, merger, and ringdown time
        tmerge = utils.merge_time_bound(
            mass_1, mass_2
        ) + utils.ringdown_time_bound(total_mass, s)

        # extra time to include for all waveforms to take care of situations
        # where the frequency is close to merger (and is sweeping rapidly):
        # this is a few cycles at the low frequency
        textra = EXTRA_CYCLES / f_min

        # lower bound on chirpt frequency start used for
        # conditioning the frequency domain waveform
        fstart = utils.chirp_start_frequency_bound(
            (1.0 + EXTRA_TIME_FRACTION) * tchirp, mass_1, mass_2
        )

        # revised chirp time estimate based on fstart
        tchirp_fstart = utils.chirp_time_bound(
            fstart, mass_1, mass_2, s1z, s2z
        )

        # chirp length in samples
        chirplen = torch.round(
            (tchirp_fstart + tmerge + 2.0 * textra) * self.sample_rate
        )

        # pad to next power of 2
        chirplen = 2 ** torch.ceil(torch.log(chirplen) / math.log(2))

        # get smallest df corresponding to longest chirp length,
        # which will make sure there is no wrap around effects.
        df = 1.0 / (chirplen.max() / self.sample_rate)

        # generate frequency array from 0 to nyquist based on df
        frequencies = self.get_frequencies(df).to(mass_1.device)

        # downselect to frequencies above fstart,
        # and generate the waveform at the specified frequencies
        freq_mask = frequencies >= fstart.min()
        waveform_frequencies = frequencies[freq_mask]

        # generate the waveform at specified frequencies
        cross, plus = self.approximant(
            waveform_frequencies, **parameters, f_ref=self.f_ref
        )
        batch_size = cross.size(0)

        # create tensors to hold the full spectrum
        # of frequencies from 0 to nyquist, and then
        # fill in the requested frequencies with the waveform values
        shape = (batch_size, frequencies.size(0))
        hc_spectrum = torch.zeros(shape, dtype=cross.dtype, device=device)
        hp_spectrum = torch.zeros(shape, dtype=plus.dtype, device=device)

        hc_spectrum[:, freq_mask] = cross
        hp_spectrum[:, freq_mask] = plus

        # build a taper that is dependent on each
        # individual waveforms fstart;
        # since this means that the taper sizes
        # will be different for each waveform,
        # construct the tapers based on the maximum size
        # and then set the values outside of the individual
        # waveform taper regions to 1.0
        k0s = torch.round(fstart / df)
        k1s = torch.round(f_min / df)

        num_freqs = frequencies.size(0)
        frequency_indices = torch.arange(num_freqs)
        taper_mask = frequency_indices <= k1s[:, None]
        taper_mask &= frequency_indices >= k0s[:, None]

        indices = frequency_indices.expand(batch_size, -1)

        kvals = indices[taper_mask]
        k0s_expanded = k0s.unsqueeze(1).expand(-1, num_freqs)[taper_mask]
        k1s_expanded = k1s.unsqueeze(1).expand(-1, num_freqs)[taper_mask]

        windows = 0.5 - 0.5 * torch.cos(
            torch.pi * (kvals - k0s_expanded) / (k1s_expanded - k0s_expanded)
        )

        hc_spectrum[taper_mask] *= windows
        hp_spectrum[taper_mask] *= windows

        # zero out frequencies below fstart
        zero_mask = frequencies < fstart[:, None]
        hc_spectrum[zero_mask] = 0
        hp_spectrum[zero_mask] = 0

        # set nyquist frequency to zero
        hc_spectrum[..., -1], hp_spectrum[..., -1] = 0.0, 0.0

        # apply time translation in (i.e. phase shift in frequency domain)
        # that will translate the coalescense time such that it is `right_pad`
        # seconds from the right edge of the window
        tshift = round(self.right_pad * self.sample_rate) / self.sample_rate
        kvals = torch.arange(num_freqs)
        phase_shift = torch.exp(1j * 2 * torch.pi * df * tshift * kvals)

        hc_spectrum *= phase_shift
        hp_spectrum *= phase_shift

        return hc_spectrum, hp_spectrum

    def forward(
        self,
        **parameters,
    ) -> Tuple[Float[Tensor, "{N} samples"], Float[Tensor, "{N} samples"]]:
        """
        Generates a time-domain waveform from a frequency domain approximant.
        Conditioning is based onhttps://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/python/lalsimulation/gwsignal/core/waveform_conditioning.py?ref_type=heads#L248 # noqa

        A frequency domain waveform is generated, conditioned (see `generate_conditioned_fd_waveform`)
        and fftdd into the time-domain

        **parameters:
            Dictionary of parameters for waveform generation
            where key is the parameter name and value is a tensor of parameters.
            It is required that `parameters` contains `mass_1`, `mass_2`, `s1z`, and `s2z`
            keys, which are used for determining parameters of data conditioning.
            If the specified approximant takes other parameters for waveform generation,
            like `chirp_mass` and `mass_ratio`, the utility functions in `ml4gw.waveforms.conversion`
            may be useful for populating the parameters dictionary with these additional parameters.
            Note that, if using an approximant from `ml4gw.waveforms.cbc`, any additional keys in `parameters`
            not ingested by the approximant will be ignored.
        """

        hc, hp = self.generate_conditioned_fd_waveform(**parameters)

        # fft to time domain and apply appropiate scaling
        hc = torch.fft.irfft(hc) * self.sample_rate
        hp = torch.fft.irfft(hp) * self.sample_rate

        # TODO: some additional tapering in the time
        # domain is performed in lalsimulation

        # pad waveforms on left up to requested duration
        pad = int((self.duration * self.sample_rate) - hp.shape[-1])
        hc = torch.nn.functional.pad(hc, (pad, 0))
        hp = torch.nn.functional.pad(hp, (pad, 0))

        # finally, highpass the waveforms,
        # going to double precision
        hp = self.highpass(hp.double())
        hc = self.highpass(hc.double())

        return hc, hp
