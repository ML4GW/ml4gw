from typing import Optional, Union

import torch

from ml4gw import gw
from ml4gw.gw import compute_network_snr
from ml4gw.transforms.transform import FittableSpectralTransform
from ml4gw.types import BatchTensor, TimeSeries2d, WaveformTensor


class SnrRescaler(FittableSpectralTransform):
    def __init__(
        self,
        num_channels: int,
        sample_rate: float,
        waveform_duration: float,
        highpass: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.highpass = highpass
        self.sample_rate = sample_rate
        self.num_channels = num_channels

        waveform_size = int(waveform_duration * sample_rate)
        num_freqs = int(waveform_size // 2 + 1)

        buff = torch.zeros((num_channels, num_freqs), dtype=dtype)
        self.register_buffer("background", buff)

        if highpass is not None:
            freqs = torch.fft.rfftfreq(waveform_size, 1 / sample_rate)
            self.register_buffer("mask", freqs >= highpass, persistent=False)
        else:
            self.mask = None

    def fit(
        self,
        *background: TimeSeries2d,
        fftlength: Optional[float] = None,
        overlap: Optional[float] = None,
    ):
        if len(background) != self.num_channels:
            raise ValueError(
                "Expected to fit whitening transform on {} background "
                "timeseries, but was passed {}".format(
                    self.num_channels, len(background)
                )
            )

        num_freqs = self.background.size(1)
        psds = []
        for x in background:
            psd = self.normalize_psd(
                x, self.sample_rate, num_freqs, fftlength, overlap
            )
            psds.append(psd)
        background = torch.stack(psds)
        super().build(background=background)

    def forward(
        self,
        responses: WaveformTensor,
        target_snrs: Optional[BatchTensor] = None,
    ):
        snrs = compute_network_snr(
            responses, self.background, self.sample_rate, self.mask
        )
        if target_snrs is None:
            idx = torch.randperm(len(snrs))
            target_snrs = snrs[idx]

        weights = target_snrs / snrs
        rescaled_responses = responses * weights.view(-1, 1, 1)

        return rescaled_responses, target_snrs

class SnrRescaler_Online(torch.nn.Module):
    """
    Module that calculates SNRs of injections relative
    to a given ASD and performs augmentation of the waveform
    dataset by rescaling injections such that they have SNRs
    given by `target_snrs`. If this argument is `None`, each
    injection is randomly matched with and scaled to the SNR
    of a different injection from the batch.
    """

    def __init__(
        self,
        sample_rate: float,
        highpass: Optional[float] = None,
        # lowpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.highpass = highpass

    def forward(
        self,
        responses: gw.WaveformTensor,
        psds: torch.Tensor,
        target_snrs: Union[BatchTensor, float, None],
    ) -> gw.WaveformTensor:
        # we can either specify one PSD for all batch
        # elements, or a PSD for each batch element
        if psds.ndim > 2 and len(psds) != len(responses):
            raise ValueError(
                "Background PSDs must either be two dimensional "
                "or have a PSD specified for every element in the "
                "batch. Expected {}, found {}".format(
                    len(responses), len(psds)
                )
            )

        # interpolate the number of PSD frequency bins down
        # to the value expected by the shape of the waveforms
        num_freqs = responses.size(-1) // 2 + 1
        if psds.size(-1) != num_freqs:
            if psds.ndim == 2:
                psds = psds[None]
                reshape = True
            else:
                reshape = False

            psds = torch.nn.functional.interpolate(psds, size=(num_freqs,))
            if reshape:
                psds = psds.view(-1, num_freqs)

        # compute the SNRs of the existing signals
        snrs = compute_network_snr(
            responses,
            psds,
            self.sample_rate,
            self.highpass,
        )

        if target_snrs is None:
            # if we didn't specify any target SNRs, then shuffle
            # the existing SNRs of the waveforms as they stand
            idx = torch.randperm(len(snrs))
            target_snrs = snrs[idx]
        elif not isinstance(target_snrs, torch.Tensor):
            # otherwise if we provided just a float, assume
            # that it's a lower bound on the desired SNR levels
            target_snrs = snrs.clamp(target_snrs, 1000)

        # reweight the amplitude of the IFO responses
        # in order to achieve the target SNRs
        target_snrs.to(snrs.device)
        weights = target_snrs / snrs
        return responses * weights.view(-1, 1, 1)

