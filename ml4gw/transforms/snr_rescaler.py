from typing import Optional

import numpy as np
import torch

from ml4gw import gw
from ml4gw.spectral import Background, normalize_psd
from ml4gw.transforms.transform import FittableTransform


class SnrRescaler(FittableTransform):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        waveform_duration: float,
        highpass: Optional[float] = None,
    ):
        super().__init__()
        self.highpass = highpass
        self.sample_rate = sample_rate
        self.df = 1 / waveform_duration
        waveform_size = int(waveform_duration * sample_rate)
        num_freqs = int(waveform_size // 2 + 1)
        buff = torch.zeros((num_ifos, num_freqs), dtype=torch.float64)
        self.register_buffer("background", buff)

        if highpass is not None:
            freqs = torch.fft.rfftfreq(waveform_size, 1 / sample_rate)
            self.register_buffer("mask", freqs >= highpass, persistent=False)
        else:
            self.mask = None

    def fit(
        self,
        *backgrounds: Background,
        sample_rate: Optional[float] = None,
        fftlength: float = 2
    ):
        psds = []
        for background in backgrounds:
            psd = normalize_psd(
                background, self.df, self.sample_rate, sample_rate, fftlength
            )
            psds.append(psd)

        background = torch.tensor(np.stack(psds), dtype=torch.float64)
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
