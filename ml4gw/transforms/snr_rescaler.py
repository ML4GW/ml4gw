from typing import Optional

import torch

from ml4gw import gw
from ml4gw.transforms.transform import FittableSpectralTransform


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
        *background: torch.Tensor,
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
        responses: gw.WaveformTensor,
        target_snrs: Optional[gw.ScalarTensor] = None,
    ):
        snrs = gw.compute_network_snr(
            responses, self.background, self.sample_rate, self.mask
        )
        if target_snrs is None:
            idx = torch.randperm(len(snrs))
            target_snrs = snrs[idx]

        weights = target_snrs / snrs
        rescaled_responses = responses * weights.view(-1, 1, 1)

        return rescaled_responses, target_snrs
